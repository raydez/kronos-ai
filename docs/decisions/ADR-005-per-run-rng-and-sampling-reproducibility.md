# ADR-005: Per-run RNG and Sampling Reproducibility

- 状态：Accepted
- 日期：2026-09-27
- 对应任务：RX-KAI-010（采样路径的 generator 化，交付于 `sampler.py`）；平台面隔离加固见 RX-KAI-011
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §9、§10、§15、§54.2、§36（ADR-005）

## 背景

上游 `sample_from_logits`（`vendor/kronos.py:370`）用 `torch.multinomial(probs, num_samples=1)`
采样，**不接受 generator**——随机性来自进程全局 RNG。若 v2 沿用，风险有三：

```text
1. 并发 run 共享全局状态：两个 run 交错时互改采样流，seed 不再决定结果
2. run 的结果依赖「此前还有谁动过全局 RNG」：同 seed 不保证同结果
3. 全局状态被采样污染：调用方无法预期自己的 RNG 流
```

`torch.manual_seed(seed)` 作为最终方案不足以解决第 1 条（§9 明确不推荐）。
同时 §9 要求：same input + same model + same seed + same sampling config
= same raw samples（同硬件/runtime），且必须记录 torch version / device / dtype，
因为跨设备 bitwise 一致**不应被默认承诺**。

自我强化测试：`top_k_top_p_filtering` 会产生含 0 概率的分布，`multinomial` 在
generator 版与全局版下对同 seed 抽出一致的结果（已在 RX-KAI-010 实测），
因此「换成 generator」不改变上游随机语义，只改变随机性的所有权。

## 决策

### 1. 每个 run 一个显式 generator

```python
generator = torch.Generator(device=device_class)   # device 不支持时 ConfigurationError
generator.manual_seed(seed)                        # seed 来自 SamplingConfig，契约限定 [0, 2**63-1]
```

`build_generator(seed, device)` 是唯一构造点；generator 沿调用链显式传入
（`decode_raw_samples` → `_decode_tokens` → `_sample`），不存全局、不用线程局部、
不依赖调用方先设 seed。

### 2. 覆盖全部采样调用点

每个自回归 step 有两次采样（s1 先验、s2 后验），**两次都**使用同一 generator；
step 之间也共享同一 generator，因此一个 run 的随机流是
「`horizon × 2` 次 `multinomial` 的确定序列」。任何新增采样调用点必须显式接收
generator——用全局 RNG 的采样调用属实现缺陷，而非风格问题。

### 3. 不读写全局 RNG（可执行断言）

- `torch.manual_seed` 只在测试内出现，用于**证明**隔离：调用前后
  `torch.random.get_rng_state()` 必须逐位相同（`tests/unit/test_sampler.py`、
  `tests/regression/test_sampler_equivalence.py`）。
- 构造 generator 本身也不触碰全局状态（有测试）。
- 上游 `sample_from_logits` 保持原样（vendored，ADR-020）；generator 化发生在受控
  adapter 的 `_sample` 中，`temperature` / `top_k` / `top_p` 的处理顺序与上游逐点一致。

### 4. 可复现性的承诺边界

```text
承诺：同 input + 同模型内容 + 同 seed + 同 sampling config + 同 device-class/dtype/torch 版本
      → raw samples 逐位一致
不承诺：跨 device（CPU/MPS/CUDA）、跨 dtype、跨 torch 版本、跨 vendored 上游 commit 的一致性
```

不承诺的部分必须**可追溯**而不是可忽略：`runtime_version` 编码
runtime 契约版本 + vendored 上游 commit + torch 版本，`device_class` / `dtype` 取
实态（§9「要记录」的落点），三者随 metadata 落盘并进入 §15 的 ForecastArtifactKey
（`KronosRuntime.artifact_identity()`）。因此「在别的环境上算出来的结果」不会因为
key 相同而被当作同一 artifact。

### 5. seed 进入 artifact 身份

`SamplingConfig.seed` 与 `sample_count` / `temperature` / `top_k` / `top_p` 一起进入
采样维哈希（RX-KAI-008 的 `hashing_payload`，§15）：同 seed 的重复运行命中同一
artifact，换 seed 必产生新 artifact——重放与缓存一致性因此不依赖文件名约定。

## 后果

- 并发 run 之间采样互不干扰；run 的结果只由 (input, 模型, seed, config, runtime) 决定。
- 违反隔离（新增采样点漏传 generator、或某处改回全局 RNG）会被全局状态断言检出。
- 复现一个历史 artifact 需要 device-class / dtype / torch 版本三者同时匹配；
  这三项都在 metadata 里，不需要重新猜测当时的环境。
- 若未来某采样点无法 generator 化（上游接口硬限制），必须在本 ADR 中显式登记为
  例外并说明隔离损失，不能静默回退到全局 RNG。
