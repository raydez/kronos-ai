"""BaoStock 数据源能力 spike（RX-KAI-006 前置，基线文档 §6.0）。

问题清单（基线文档 §6.0）：

1. ``query_trade_dates``         → 交易日历覆盖范围（是否包含未来日期）
2. ``query_adjust_factor``       → 复权因子的 PIT 语义（历史记录是否被重算）
3. ``query_hs300_stocks(date)``  → 历史成分股回溯深度与日期粒度
4. ``query_history_k_data_plus`` → tradestatus 可用性、停牌日是否有 bar

外加稳定性前提：数据源能力报告必须区分三类结论——

- ``ok``         服务端确实返回了可用数据；
- ``error``      检查自身抛异常（含客户端返回 None），或跑完但**未确认**任何能力；
- ``unverified`` 在硬超时内没有任何响应。

「无响应」绝不能被当成「能力确认」：脚本以非零退出码结束并分别列出 ``errors`` /
``unverified``。退出码：0 = 全部 ``ok``；1 = 存在 ``error``；2 = 存在 ``unverified``
且无 ``error``；3 = 两者都有（细节见 ``docs/spike/baostock-capability.md``）。

每个检查必须在 payload 里给出 ``confirmed``（能力是否被确认），且除
``error_semantics`` 外的检查不允许出现任何 ``error_code != "0"`` 的探针结果；
否则 ``main`` 把该检查降级为 ``error``（RX-KAI-006 评审 M5）——「所有查询都失败但
恰好跑完」不得被计为 ``ok``。

用法::

    uv run python experiments/spike_baostock_capability.py [--timeout 15] [--json out.json]

客户端 / 端点：baostock 0.8.x 的端点是 ``www.baostock.com:10030``，0.9.x 起改为
``public-api.baostock.com:10030``（``vip-api.baostock.com`` 用于 API key 登录）。
旧客户端配旧端点在本环境下表现为「TCP 连接成功、服务端立即关闭、客户端在
``send_msg`` 的 ``while True: recv`` 中空转」，因此每个检查都在 daemon 线程内
执行并施加硬超时。0.9.4 仍未修复该空转逻辑（EOF 与超时都不被当作错误）。

报告中的每一条事实都必须能从 ``--json`` 落盘的 artifact 里读出来，不得只存在于
叙述文字里（RX-KAI-006 评审 Major 2）。
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import inspect
import json
import os
import socket
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

TIMEOUT_S = 15.0
FUTURE_END = "2027-12-31"
CALENDAR_START = "1990-01-01"
FACTOR_START = "2000-01-01"
# 因子窗口截断探针的右端点：与整段窗口共享的 ex-date 因子值必须完全相同
TRUNCATED_FACTOR_END = "2022-12-31"
BEYOND_COVERAGE_START = "2027-01-01"
BEYOND_COVERAGE_END = "2027-12-31"
CONSTITUENT_FUTURE_DATE = "2027-06-30"
# 中秋 + 国庆窗口：含星期五休市（2026-09-25）与 10-01..10-07 长假，
# 是「工作日规则不可替代日历」的完整可核对证据（ADR-009 §6）。
HOLIDAY_WINDOW = ("2026-09-24", "2026-10-12")
# 成分股回溯深度的边界探针：HS300 在 2005-12-30/2006-01-04 之间出现，
# ZZ500 在 2007-01-04/2007-01-31 之间出现（两侧都探，避免再次把「窗口外空」
# 误读成「标的已退市」一类的错误结论）。
HISTORY_DEPTH_PROBES = (
    "2005-06-30",
    "2005-12-30",
    "2006-01-04",
    "2006-06-20",
    "2006-06-30",
    "2007-01-04",
    "2007-01-31",
    "2008-06-30",
    "2020-06-30",
)
INDEX_QUERIES = ("hs300", "zz500")
INDEX_EXPECTED_SIZE = {"hs300": 300, "zz500": 500}
ADJUST_FACTOR_CODE = "sh.600000"
# 对照标的：仍在市，用于证明 query_stock_basic 能区分「退市」与「在册」
LISTED_REFERENCE_CODE = "sh.600000"
# golden 标的日：测试锚定 raw/hfq/qfq 与适用因子（评审 Major 2）
MAPPING_SAMPLE_CODE = "sh.600000"
MAPPING_SAMPLE_DAY = "2023-01-03"
# 退市标的：用 query_stock_basic 的 outDate/status 证明退市，而非靠「因子窗口为空」
DELISTED_CODE = "sz.000003"
DELISTED_HISTORY_WINDOW = ("2002-06-01", "2002-06-30")
# 旧端点：0.8.9 写死的地址。留着是为了让「端点迁移」这一前置结论可核对（评审 M4）。
LEGACY_HOST = "www.baostock.com"
PUBLICATION_LAG_CODES = ("sh.600000", "sz.000001", "sz.300750")
PUBLICATION_LAG_DAYS = 10
FACTOR_NORMALIZATION_CODES = (
    "sh.600000",
    "sz.000001",
    "sh.600519",
    "sz.300750",
    "sz.000002",
    "sh.601318",
)
SUSPENSION_CODES = ("sh.600000", "sh.600816", "sz.000010")
FACTOR_MAPPING_PROBES = (
    ("sh.600000", "2000-01-04", "2000-06-30"),
    ("sz.000002", "2000-01-04", "2001-12-31"),
    ("sh.600519", "2001-09-03", "2002-12-31"),
    ("sz.300750", "2018-06-11", "2019-12-31"),
    ("sh.601318", "2007-03-02", "2008-12-31"),
)

CheckFn = Callable[[], dict[str, Any]]


def _bounded(fn: CheckFn, timeout: float) -> tuple[str, dict[str, Any]]:
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # spike 必须记录任何失败
            box["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=target, daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(timeout)
    elapsed = round(time.monotonic() - started, 2)
    if worker.is_alive():
        return "unverified", {"reason": "no server response within timeout", "elapsed_s": elapsed}
    if "error" in box:
        return "error", {"reason": box["error"], "elapsed_s": elapsed}
    return "ok", {**box["value"], "elapsed_s": elapsed}


def _page_rows() -> int:
    """库单页行数（读不到则 0）：识别「整页游标」截断痕迹用。

    页长必须来自库常量而不是硬编码——否则库改页长后，静默翻页失败会被记成
    「短但自洽」的结果，spike 会把截断当成数据结论（评审回归项 3）。
    """
    try:
        import baostock
    except ImportError:
        return 0
    contants = getattr(getattr(baostock, "common", None), "contants", None)
    value = getattr(contants, "BAOSTOCK_PER_PAGE_COUNT", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _reject_truncation(result: Any, collected: int) -> None:
    """整页 + 游标到头 = ``send_msg`` 静默失败，spike 必须显式失败。

    0.9.4 的 ``next()`` 在翻页请求静默失败时返回 False 且不置 ``error_code``，
    唯一痕迹是游标停在整页；若不检测，「少了一批行」会被当成完整结果写入 artifact。
    正常收尾只可能是空结果、非整页收尾、或整页后再取到空页（``data`` 清空）。
    """
    page_rows = _page_rows()
    data = getattr(result, "data", None)
    cursor = getattr(result, "cur_row_num", None)
    if not page_rows or not isinstance(data, list) or not isinstance(cursor, int):
        return
    if len(data) == page_rows and cursor >= page_rows:
        raise RuntimeError(
            f"paging appears truncated after {collected} rows: a full page was followed "
            "by an empty response with no error code"
        )


def _rows(result: Any) -> tuple[str, str, list[str], list[list[str]]]:
    """(error_code, error_msg, fields, rows)；客户端参数校验失败时 result 为 None。

    ``CLIENT_RETURNED_NONE`` 是**失败**状态而非数据结论：调用方必须把它当错误处理，
    ``_enforce_confirmation`` 会拦下任何含非 ``"0"`` ``error_code`` 的检查。
    翻页静默截断（无 error_code）由 ``_reject_truncation`` 直接抛错。
    """
    if result is None:
        return "CLIENT_RETURNED_NONE", "client-side validation rejected the request", [], []
    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    _reject_truncation(result, len(rows))
    return result.error_code, result.error_msg, list(result.fields), rows


def _walk_error_codes(node: Any, prefix: str = "") -> list[str]:
    """递归收集 payload 里所有非 ``"0"`` 的 ``error_code``（含嵌套探针）。"""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key == "error_code" and value != "0":
                found.append(f"{path}={value!r}")
            else:
                found.extend(_walk_error_codes(value, path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk_error_codes(value, f"{prefix}[{index}]"))
    return found


def _verdict(confirmed: bool, criterion: str) -> dict[str, Any]:
    """确认位与判定标准；``confirmed`` 非真时 ``main`` 把该检查降级为 ``error``。"""
    return {"confirmed": confirmed, "confirm_criterion": criterion}


def _enforce_confirmation(
    name: str, status: str, evidence: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """把「跑完了但没确认能力」降级为 ``error``（RX-KAI-006 评审 M5）。

    ``error_semantics`` 的唯一豁免是「非零 error_code」——它记录的就是错误语义本身；
    它仍必须通过 ``confirmed``（每个探针都符合预期行为）。
    """
    if status != "ok":
        return status, evidence
    if name != "error_semantics":
        failures = sorted(set(_walk_error_codes(evidence)))
        if failures:
            return "error", {
                **evidence,
                "reason": "probes reported errors: " + "; ".join(failures),
            }
    if evidence.get("confirmed") is not True:
        criterion = evidence.get("confirm_criterion") or f"{name} did not confirm its capability"
        return "error", {**evidence, "reason": f"not confirmed: {criterion}"}
    return status, evidence


def check_trade_dates(bs: Any) -> dict[str, Any]:
    error_code, error_msg, fields, rows = _rows(
        bs.query_trade_dates(start_date=CALENDAR_START, end_date=FUTURE_END)
    )
    days = [row[0] for row in rows]
    sessions = [row[0] for row in rows if row[1] == "1"]
    today = date.today().isoformat()

    beyond_code, beyond_msg, _, beyond_rows = _rows(
        bs.query_trade_dates(
            start_date=BEYOND_COVERAGE_START,
            end_date=BEYOND_COVERAGE_END,
        )
    )
    _, _, _, holiday_rows = _rows(
        bs.query_trade_dates(start_date=HOLIDAY_WINDOW[0], end_date=HOLIDAY_WINDOW[1])
    )
    confirmed = bool(days) and bool(sessions) and bool(holiday_rows) and not beyond_rows
    return {
        **_verdict(
            confirmed,
            "coverage rows + sessions + holiday window non-empty, beyond-coverage window empty",
        ),
        "error_code": error_code,
        "error_msg": error_msg,
        "fields": fields,
        "rows": len(rows),
        "coverage": [min(days), max(days)] if days else None,
        "requested_end": FUTURE_END,
        "sessions": len(sessions),
        "covers_future_sessions": bool(days) and max(days) > today,
        "beyond_coverage": {
            "window": [BEYOND_COVERAGE_START, BEYOND_COVERAGE_END],
            "error_code": beyond_code,
            "error_msg": beyond_msg,
            "rows": len(beyond_rows),
            "silent_empty": beyond_code == "0" and not beyond_rows,
        },
        "holiday_window": {
            "window": list(HOLIDAY_WINDOW),
            "rows": holiday_rows,
            "sessions": [row[0] for row in holiday_rows if row[1] == "1"],
        },
        "sample": rows[:3],
    }


def check_adjust_factor(bs: Any) -> dict[str, Any]:
    full_window = (FACTOR_START, FUTURE_END)
    truncated_window = (FACTOR_START, TRUNCATED_FACTOR_END)
    full_code, full_msg, fields, full_rows = _rows(
        bs.query_adjust_factor(
            code=ADJUST_FACTOR_CODE, start_date=full_window[0], end_date=full_window[1]
        )
    )
    _, _, _, truncated_rows = _rows(
        bs.query_adjust_factor(
            code=ADJUST_FACTOR_CODE, start_date=truncated_window[0], end_date=truncated_window[1]
        )
    )
    date_index = fields.index("dividOperateDate") if "dividOperateDate" in fields else 1
    back_index = fields.index("backAdjustFactor") if "backAdjustFactor" in fields else 3
    adjust_index = fields.index("adjustFactor") if "adjustFactor" in fields else 4
    full_map = {row[date_index]: row[back_index] for row in full_rows}
    truncated_map = {row[date_index]: row[back_index] for row in truncated_rows}
    shared = sorted(set(full_map) & set(truncated_map))
    rewritten = [key for key in shared if full_map[key] != truncated_map[key]]
    ex_dates = sorted(full_map)
    confirmed = bool(full_rows) and bool(shared) and not rewritten
    return {
        **_verdict(
            confirmed,
            "factor table non-empty and identical to the truncated window on shared ex-dates",
        ),
        "code": ADJUST_FACTOR_CODE,
        "error_code": full_code,
        "error_msg": full_msg,
        "fields": fields,
        # 两个请求窗口落盘：报告引用的窗口必须能从 artifact 读出（评审回归项 4）
        "windows": {"full": list(full_window), "truncated": list(truncated_window)},
        "rows_full": len(full_rows),
        "rows_truncated": len(truncated_rows),
        "coverage": [ex_dates[0], ex_dates[-1]] if ex_dates else None,
        "shared_ex_dates": len(shared),
        "history_rewritten_when_window_changes": bool(rewritten),
        "rewritten_ex_dates": rewritten[:5],
        "adjust_and_back_columns_identical": (
            all(row[adjust_index] == row[back_index] for row in full_rows) if fields else None
        ),
        # 整张因子表落盘：测试 fixture 与报告数字都必须能从这里逐行核对（评审 Major 2）
        "factor_table": full_rows,
    }


def check_symbol_lifecycle(bs: Any) -> dict[str, Any]:
    """退市标的的可用证据：靠 ``query_stock_basic`` 的 outDate/status，而不是「窗口为空」。

    教训（评审 Major 2）：先前把「``start_date=2000-01-01`` 的因子窗口为空」误读成
    「已退市标的静默返回空」。实际 ``sz.000003`` 有 1991–1996 共 15 条因子记录，
    空只是窗口造成的；退市必须由 ``query_stock_basic`` 的 ``outDate`` / ``status`` 判定。
    两种窗口与在册对照标的都落盘，使这条纠正本身可核对（评审 M3）。
    """
    basic_code, basic_msg, basic_fields, basic_rows = _rows(
        bs.query_stock_basic(code=DELISTED_CODE)
    )
    listed_code, listed_msg, _, listed_rows = _rows(
        bs.query_stock_basic(code=LISTED_REFERENCE_CODE)
    )
    factor_code, factor_msg, factor_fields, factor_rows = _rows(
        bs.query_adjust_factor(code=DELISTED_CODE, start_date="1990-01-01", end_date=FUTURE_END)
    )
    window_code, window_msg, _, window_rows = _rows(
        bs.query_adjust_factor(code=DELISTED_CODE, start_date=FACTOR_START, end_date=FUTURE_END)
    )
    window_start, window_end = DELISTED_HISTORY_WINDOW
    hist_code, hist_msg, hist_fields, hist_rows = _rows(
        bs.query_history_k_data_plus(
            DELISTED_CODE,
            "date,open,high,low,close,volume,amount,tradestatus",
            start_date=window_start,
            end_date=window_end,
            frequency="d",
            adjustflag="3",
        )
    )
    column = {name: hist_fields.index(name) for name in hist_fields}
    suspended_rows = [row for row in hist_rows if row[column["tradestatus"]] == "0"]
    flat_ohlc = sum(
        1
        for row in suspended_rows
        if row[column["open"]] == row[column["high"]] == row[column["low"]] == row[column["close"]]
    )
    volume_zero = sum(1 for row in suspended_rows if row[column["volume"]] in ("0", "0.0"))
    empty_numeric_field_rows = [
        {
            "row": row,
            "empty_volume": row[column["volume"]] == "",
            "empty_amount": row[column["amount"]] == "",
        }
        for row in hist_rows
        if row[column["volume"]] == "" or row[column["amount"]] == ""
    ]
    delisted_out_dates = [row[basic_fields.index("outDate")] for row in basic_rows]
    confirmed = (
        bool(basic_rows)
        and all(day for day in delisted_out_dates)  # 退市判定必须由 outDate 给出
        and bool(factor_rows)
        and not window_rows  # 2000 起为空纯属窗口效应，与退市无关
        and bool(hist_rows)
        and len(suspended_rows) == len(hist_rows)
    )
    return {
        **_verdict(
            confirmed,
            "query_stock_basic gives non-empty outDate; factor rows exist from 1990 but not "
            "from 2000; history rows in the delisting window are all tradestatus=0",
        ),
        "code": DELISTED_CODE,
        "stock_basic": {
            "error_code": basic_code,
            "error_msg": basic_msg,
            "fields": basic_fields,
            "rows": basic_rows,
        },
        "listed_reference": {
            "code": LISTED_REFERENCE_CODE,
            "error_code": listed_code,
            "error_msg": listed_msg,
            "rows": listed_rows,
        },
        "adjust_factor": {
            "error_code": factor_code,
            "error_msg": factor_msg,
            "fields": factor_fields,
            "from_1990": {
                "window": ["1990-01-01", FUTURE_END],
                "rows": len(factor_rows),
                "coverage": (
                    [
                        factor_rows[0][
                            factor_fields.index("dividOperateDate")
                            if "dividOperateDate" in factor_fields
                            else 1
                        ],
                        factor_rows[-1][
                            factor_fields.index("dividOperateDate")
                            if "dividOperateDate" in factor_fields
                            else 1
                        ],
                    ]
                    if factor_rows
                    else None
                ),
            },
            "from_2000": {
                "window": [FACTOR_START, FUTURE_END],
                "error_code": window_code,
                "error_msg": window_msg,
                "rows": len(window_rows),
                "note": "empty window is an artifact of the window, not a delisting signal",
            },
        },
        "history": {
            "window": list(DELISTED_HISTORY_WINDOW),
            "error_code": hist_code,
            "error_msg": hist_msg,
            "fields": hist_fields,
            "rows": len(hist_rows),
            "table": hist_rows,
            "suspended_rows": len(suspended_rows),
            "flat_ohlc_rows": flat_ohlc,
            "volume_zero_rows": volume_zero,
            "empty_numeric_field_rows": empty_numeric_field_rows,
        },
    }


def check_publication_lag(bs: Any) -> dict[str, Any]:
    """已完成 session 的 bar 何时可取：观测发布滞后，不验证具体钟点。

    ``BAO_STOCK_PUBLISHED_AT = 18:00`` 是**乐观**假设（若真实发布更晚，同日晚间 run
    会用到尚未发布的 bar），本检查只能给出「哪一天为止已可取」的下界证据：
    记录最近窗口的 calendar session、每个标的最后一条 bar 是哪天、哪些 session 尚无
    bar。bar 缺行既可能是发布滞后也可能是休市，必须与 calendar 对照。
    """
    today = date.today()
    window_start = (today - timedelta(days=PUBLICATION_LAG_DAYS)).isoformat()
    end = today.isoformat()
    _, _, cal_fields, cal_rows = _rows(bs.query_trade_dates(start_date=window_start, end_date=end))
    day_index = cal_fields.index("calendar_date") if "calendar_date" in cal_fields else 0
    flag_index = cal_fields.index("is_trading_day") if "is_trading_day" in cal_fields else 1
    sessions = [row[day_index] for row in cal_rows if row[flag_index] == "1"]

    per_code: dict[str, Any] = {}
    for code in PUBLICATION_LAG_CODES:
        error_code, _, fields, rows = _rows(
            bs.query_history_k_data_plus(
                code,
                "date,close,tradestatus",
                start_date=window_start,
                end_date=end,
                frequency="d",
                adjustflag="3",
            )
        )
        bar_days = [row[fields.index("date")] for row in rows]
        per_code[code] = {
            "error_code": error_code,
            "rows": len(rows),
            "bar_days": bar_days,
            "last_bar_day": bar_days[-1] if bar_days else None,
        }
    last_bar_days = {entry["last_bar_day"] for entry in per_code.values() if entry["last_bar_day"]}
    confirmed = bool(sessions) and len(last_bar_days) == 1
    return {
        **_verdict(
            confirmed,
            "calendar sessions non-empty and all codes agree on the same last bar day",
        ),
        "today": today.isoformat(),
        "window": [window_start, end],
        "calendar_sessions": sessions,
        "codes": per_code,
        "sessions_without_bars": [
            day for day in sessions if last_bar_days and day > max(last_bar_days)
        ],
        "last_bar_day_consistent": len(last_bar_days) == 1,
    }


def _probe_constituent_dates(
    bs: Any,
    query_name: str,
    code_index: int,
    update_index: int,
    latest_codes: set[str],
) -> dict[str, Any]:
    """同一组日期上探测某个指数的历史名单（HS300 与 ZZ500 深度不同，必须分别探）。"""
    query = getattr(bs, query_name)
    out: dict[str, Any] = {}
    for probe in HISTORY_DEPTH_PROBES:
        error_code, _, _, rows = _rows(query(date=probe))
        codes = {row[code_index] for row in rows}
        out[probe] = {
            "error_code": error_code,
            "rows": len(rows),
            "update_dates": sorted({row[update_index] for row in rows}),
            "diff_vs_latest": len(codes ^ latest_codes),
            "empty": not rows,
        }
    return out


def check_index_constituents(bs: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, query_name in (("hs300", "query_hs300_stocks"), ("zz500", "query_zz500_stocks")):
        query = getattr(bs, query_name)
        latest_code, latest_msg, fields, latest_rows = _rows(query())
        code_index = fields.index("code") if "code" in fields else 1
        update_index = fields.index("updateDate") if "updateDate" in fields else 0
        latest_codes = {row[code_index] for row in latest_rows}
        future_code, _, _, future_rows = _rows(query(date=CONSTITUENT_FUTURE_DATE))
        future_codes = {row[code_index] for row in future_rows}
        result[label] = {
            "error_code": latest_code,
            "error_msg": latest_msg,
            "fields": fields,
            "latest_rows": len(latest_rows),
            "latest_update_dates": sorted({row[update_index] for row in latest_rows}),
            "by_probe_date": _probe_constituent_dates(
                bs, query_name, code_index, update_index, latest_codes
            ),
            "future_date_probe": {
                "date": CONSTITUENT_FUTURE_DATE,
                "error_code": future_code,
                "rows": len(future_rows),
                "update_dates": sorted({row[update_index] for row in future_rows}),
                "same_set_as_latest": future_codes == latest_codes,
            },
            "sample": latest_rows[:2],
        }
    sizes_correct = all(
        result[label]["latest_rows"] == INDEX_EXPECTED_SIZE[label] for label in result
    )
    history_nonempty = all(
        any(not probe["empty"] for probe in result[label]["by_probe_date"].values())
        for label in result
    )
    future_leak = all(result[label]["future_date_probe"]["same_set_as_latest"] for label in result)
    confirmed = sizes_correct and history_nonempty and future_leak
    return {
        **_verdict(
            confirmed,
            "latest lists match index sizes, at least one historical probe is non-empty, and "
            "the future-date probe returns the latest list unchanged",
        ),
        **result,
    }


def check_suspension(bs: Any) -> dict[str, Any]:
    fields = "date,open,high,low,close,volume,amount,tradestatus,isST"
    per_code: dict[str, Any] = {}
    for code in SUSPENSION_CODES:
        error_code, _, result_fields, rows = _rows(
            bs.query_history_k_data_plus(
                code,
                fields,
                start_date="2015-01-01",
                end_date="2024-12-31",
                frequency="d",
                adjustflag="3",
            )
        )
        # 按返回的 fields 定位列，不依赖列序（评审 m2）
        column = {name: result_fields.index(name) for name in result_fields}
        statuses = Counter(row[column["tradestatus"]] for row in rows)
        suspended_indices = [
            index for index, row in enumerate(rows) if row[column["tradestatus"]] == "0"
        ]
        suspended = [rows[index] for index in suspended_indices]
        st_rows = [row for row in rows if row[column["isST"]] == "1"]
        per_code[code] = {
            "error_code": error_code,
            "fields": result_fields,
            "rows": len(rows),
            "tradestatus_values": dict(statuses),
            "suspended_rows": len(suspended),
            "suspended_flat_ohlc": sum(
                1
                for row in suspended
                if row[column["open"]]
                == row[column["high"]]
                == row[column["low"]]
                == row[column["close"]]
            ),
            "suspended_repeats_prev_close": sum(
                1
                for index in suspended_indices
                if index > 0 and rows[index][column["close"]] == rows[index - 1][column["close"]]
            ),
            "suspended_volume_zero": sum(
                1 for row in suspended if row[column["volume"]] in ("0", "0.0")
            ),
            "suspended_empty_volume": sum(1 for row in suspended if row[column["volume"]] == ""),
            "suspended_amount_zero": sum(
                1 for row in suspended if float(row[column["amount"]] or 0) == 0.0
            ),
            "suspended_sample": suspended[:2],
            "isST_rows": len(st_rows),
            "isST_sample": st_rows[:2],
        }
    totals = {
        "total_suspended_rows": sum(entry["suspended_rows"] for entry in per_code.values()),
        "total_flat_ohlc": sum(entry["suspended_flat_ohlc"] for entry in per_code.values()),
        "total_repeats_prev_close": sum(
            entry["suspended_repeats_prev_close"] for entry in per_code.values()
        ),
        "total_volume_zero": sum(entry["suspended_volume_zero"] for entry in per_code.values()),
        "total_amount_zero": sum(entry["suspended_amount_zero"] for entry in per_code.values()),
        "total_empty_volume": sum(entry["suspended_empty_volume"] for entry in per_code.values()),
    }
    confirmed = (
        totals["total_suspended_rows"] > 0
        and totals["total_flat_ohlc"] == totals["total_suspended_rows"]
    )
    return {
        **_verdict(
            confirmed,
            "at least one suspended row exists and every suspended row carries flat OHLC",
        ),
        "codes": per_code,
        **totals,
    }


def check_factor_normalization(bs: Any) -> dict[str, Any]:
    """复权因子表是否以「最新公司行动」为基准归一化。

    若最新记录的 foreAdjustFactor 恒为 1.0，说明整张表是相对查询时刻重算的，
    由它派生的复权序列即随查询时刻变化，无法 PIT 复现——这是 ADR-007 的核心证据。
    """
    per_code: dict[str, Any] = {}
    for code in FACTOR_NORMALIZATION_CODES:
        error_code, _, fields, rows = _rows(
            bs.query_adjust_factor(code=code, start_date="2000-01-01", end_date=FUTURE_END)
        )
        latest = rows[-1] if rows else None
        fore_index = fields.index("foreAdjustFactor") if "foreAdjustFactor" in fields else 2
        date_index = fields.index("dividOperateDate") if "dividOperateDate" in fields else 1
        per_code[code] = {
            "error_code": error_code,
            "records": len(rows),
            "latest_ex_date": latest[date_index] if latest else None,
            "latest_fore_factor": latest[fore_index] if latest else None,
            "latest_fore_is_unity": float(latest[fore_index]) == 1.0 if latest else None,
        }
    values = [entry["latest_fore_is_unity"] for entry in per_code.values()]
    all_unity = all(values) if values else None
    return {
        **_verdict(
            all_unity is True,
            "every symbol's latest foreAdjustFactor equals 1.0",
        ),
        "codes": per_code,
        "all_latest_fore_unity": all_unity,
        "interpretation": (
            "latest foreAdjustFactor == 1.0 across symbols → the factor table is normalized "
            "to the most recent corporate action at query time; adjusted series derived from "
            "it are query-time dependent and not PIT-reproducible without snapshots"
        ),
    }


def _factor_mapping_sample(bs: Any) -> dict[str, Any]:
    """单个标的日的 raw/hfq/qfq 收盘与适用因子：golden 值，供测试锚定。

    产出 ``hfq_over_raw`` / ``qfq_over_raw``，应与 ``applicable_back_factor`` /
    ``applicable_fore_factor`` 相等——这是映射规则的单点可读证据。
    """
    _, _, factor_fields, factor_rows = _rows(
        bs.query_adjust_factor(
            code=MAPPING_SAMPLE_CODE, start_date="1990-01-01", end_date=FUTURE_END
        )
    )
    date_index = factor_fields.index("dividOperateDate") if factor_fields else 1
    fore_index = factor_fields.index("foreAdjustFactor") if factor_fields else 2
    back_index = factor_fields.index("backAdjustFactor") if factor_fields else 3
    ex_dates = [row[date_index] for row in factor_rows]
    position = bisect.bisect_right(ex_dates, MAPPING_SAMPLE_DAY)
    applicable = factor_rows[position - 1] if position else None

    closes: dict[str, Any] = {}
    for label, flag in (("raw", "3"), ("hfq", "1"), ("qfq", "2")):
        _, _, fields, rows = _rows(
            bs.query_history_k_data_plus(
                MAPPING_SAMPLE_CODE,
                "date,close",
                start_date=MAPPING_SAMPLE_DAY,
                end_date=MAPPING_SAMPLE_DAY,
                frequency="d",
                adjustflag=flag,
            )
        )
        close_index = fields.index("close") if "close" in fields else 1
        closes[label] = float(rows[0][close_index]) if rows else None

    raw = closes["raw"]
    return {
        "code": MAPPING_SAMPLE_CODE,
        "day": MAPPING_SAMPLE_DAY,
        "closes": closes,
        "applicable_ex_date": applicable[date_index] if applicable else None,
        "applicable_fore_factor": applicable[fore_index] if applicable else None,
        "applicable_back_factor": applicable[back_index] if applicable else None,
        "hfq_over_raw": closes["hfq"] / raw if raw else None,
        "qfq_over_raw": closes["qfq"] / raw if raw else None,
    }


def check_factor_price_mapping(bs: Any) -> dict[str, Any]:
    """复权因子与 adjustflag 复权序列的一致性（hfq/qfq 映射规则）。

    对每个被测标的的全部重叠交易日验证：

    - ``hfq(D) == raw(D) * backAdjustFactor(最近一个 ex_date <= D)``
    - ``qfq(D) == raw(D) * foreAdjustFactor(最近一个 ex_date <= D)``

    规则一旦成立，复权序列就完全由「原始价格 + 因子表」决定；因此因子表若被
    快照留痕，复权序列即可 PIT 复现（ADR-007）。
    """
    tolerance = 1e-6
    per_probe: dict[str, Any] = {}
    for code, start, end in FACTOR_MAPPING_PROBES:
        _, _, factor_fields, factor_rows = _rows(
            bs.query_adjust_factor(code=code, start_date="1990-01-01", end_date=FUTURE_END)
        )
        date_index = factor_fields.index("dividOperateDate") if factor_fields else 1
        fore_index = factor_fields.index("foreAdjustFactor") if factor_fields else 2
        back_index = factor_fields.index("backAdjustFactor") if factor_fields else 3
        ex_dates = [row[date_index] for row in factor_rows]

        def closes(
            adjustflag: str, code: str = code, start: str = start, end: str = end
        ) -> dict[str, float]:
            _, _, _, rows = _rows(
                bs.query_history_k_data_plus(
                    code,
                    "date,close",
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag=adjustflag,
                )
            )
            return {row[0]: float(row[1]) for row in rows}

        raw, hfq, qfq = closes("3"), closes("1"), closes("2")
        days = sorted(set(raw) & set(hfq) & set(qfq))
        checked = hfq_mismatch = qfq_mismatch = pre_first_ex_days = 0
        for day in days:
            position = bisect.bisect_right(ex_dates, day)
            if position == 0:
                pre_first_ex_days += 1
                continue
            fore = float(factor_rows[position - 1][fore_index])
            back = float(factor_rows[position - 1][back_index])
            scale = max(1.0, abs(hfq[day]))
            checked += 1
            if abs(hfq[day] - raw[day] * back) > tolerance * scale:
                hfq_mismatch += 1
            if abs(qfq[day] - raw[day] * fore) > tolerance * scale:
                qfq_mismatch += 1
        per_probe[code] = {
            "window": [start, end],
            "first_factor_ex_date": ex_dates[0] if ex_dates else None,
            "days_in_window": len(days),
            "checked_days": checked,
            "days_before_first_ex_date": pre_first_ex_days,
            "hfq_mismatch": hfq_mismatch,
            "qfq_mismatch": qfq_mismatch,
        }
    total_checked = sum(entry["checked_days"] for entry in per_probe.values())
    total_mismatch = sum(
        entry["hfq_mismatch"] + entry["qfq_mismatch"] for entry in per_probe.values()
    )
    rule_holds = total_mismatch == 0 and total_checked > 0
    return {
        **_verdict(rule_holds, ">0 checked symbol-days and zero hfq/qfq mapping mismatches"),
        "rule": (
            "hfq(D) = raw(D) * backAdjustFactor(latest ex_date <= D); "
            "qfq(D) = raw(D) * foreAdjustFactor(latest ex_date <= D)"
        ),
        "probes": per_probe,
        "total_checked_days": total_checked,
        "total_mismatches": total_mismatch,
        "rule_holds": rule_holds,
        "sample_day": _factor_mapping_sample(bs),
    }


def check_error_semantics(bs: Any) -> dict[str, Any]:
    """错误语义探针：每个探针都带预期行为，全部命中才算确认。

    ``error_semantics`` 是唯一允许出现非零 ``error_code`` 的检查（它记录的就是错误
    本身），但「记录到」不等于「符合预期」——``confirmed`` 要求每个探针的表现与文档
    一致，否则同样失败（评审 M5）。
    """
    probes: dict[str, dict[str, Any]] = {}

    def record(label: str, result: Any, expectation: str) -> None:
        error_code, error_msg, _, rows = _rows(result)
        if expectation == "silent_empty":
            matches = error_code == "0" and not rows
        elif expectation == "client_none":
            matches = error_code == "CLIENT_RETURNED_NONE"
        elif expectation == "server_error":
            matches = error_code not in ("0", "CLIENT_RETURNED_NONE")
        elif expectation == "ok_non_trading_row":
            # 单日节假日查询是**正常**返回：1 行且 is_trading_day=0（query_trade_dates
            # 的 fields 固定为 calendar_date,is_trading_day → 列 1）
            matches = error_code == "0" and len(rows) == 1 and rows[0][1].strip() == "0"
        else:  # pragma: no cover - 常量拼写错误在自检中暴露
            raise ValueError(f"unknown expectation: {expectation}")
        probes[label] = {
            "expectation": expectation,
            "matches_expectation": matches,
            "error_code": error_code,
            "error_msg": error_msg,
            "rows": len(rows),
            "silent_empty": error_code == "0" and not rows,
            "sample_rows": rows[:3],
        }

    record(
        "unknown_symbol",
        bs.query_history_k_data_plus(
            "sh.999999",
            "date,close",
            start_date="2020-01-01",
            end_date="2020-01-10",
            frequency="d",
            adjustflag="3",
        ),
        "silent_empty",
    )
    record(
        "malformed_start_date",
        bs.query_history_k_data_plus(
            "sh.600000",
            "date,close",
            start_date="2020/01/01",
            end_date="2020-01-10",
            frequency="d",
            adjustflag="3",
        ),
        "client_none",
    )
    record(
        "start_after_end",
        bs.query_history_k_data_plus(
            "sh.600000",
            "date,close",
            start_date="2021-01-01",
            end_date="2020-01-10",
            frequency="d",
            adjustflag="3",
        ),
        "server_error",
    )
    record(
        "unknown_field",
        bs.query_history_k_data_plus(
            "sh.600000",
            "date,nosuchfield",
            start_date="2020-01-01",
            end_date="2020-01-10",
            frequency="d",
            adjustflag="3",
        ),
        "server_error",
    )
    record(
        "unknown_adjustflag",
        bs.query_history_k_data_plus(
            "sh.600000",
            "date,close",
            start_date="2020-01-01",
            end_date="2020-01-10",
            frequency="d",
            adjustflag="9",
        ),
        "server_error",
    )
    record(
        "fully_future_window",
        bs.query_history_k_data_plus(
            "sh.600000",
            "date,close",
            start_date=BEYOND_COVERAGE_START,
            end_date=BEYOND_COVERAGE_END,
            frequency="d",
            adjustflag="3",
        ),
        "silent_empty",
    )
    record(
        "adjustfactor_unknown_symbol",
        bs.query_adjust_factor(code="sh.999999", start_date="2000-01-01", end_date=FUTURE_END),
        "silent_empty",
    )
    record(
        "trade_dates_holiday_single_day",
        bs.query_trade_dates(start_date="2026-10-01", end_date="2026-10-01"),
        "ok_non_trading_row",
    )
    record(
        "trade_dates_mid_autumn_single_day",
        bs.query_trade_dates(start_date="2026-09-25", end_date="2026-09-25"),
        "ok_non_trading_row",
    )
    mismatches = sorted(
        label for label, probe in probes.items() if not probe["matches_expectation"]
    )
    confirmed = not mismatches
    return {
        **_verdict(confirmed, "every probe matches its declared expectation"),
        "probes": probes,
        "mismatches": mismatches,
    }


def check_client_info(bs: Any) -> dict[str, Any]:
    cons = __import__("baostock.common.contants", fromlist=["contants"])
    host = cons.BAOSTOCK_SERVER_IP
    port = cons.BAOSTOCK_SERVER_PORT
    active_probe = _tcp_probe(host, port)
    legacy_probe = _tcp_probe(LEGACY_HOST, port)
    confirmed = bool(active_probe.get("connect_ok"))
    return {
        **_verdict(confirmed, "the active endpoint accepts a TCP connection"),
        "client_version": cons.BAOSTOCK_CLIENT_VERSION,
        "server_host": host,
        "server_port": port,
        "vip_server_host": getattr(cons, "BAOSTOCK_VIP_SERVER_IP", None),
        "api_key": _api_key_evidence(bs, cons),
        "tcp_probe": active_probe,
        "legacy_endpoint_probe": legacy_probe,
    }


def _api_key_evidence(bs: Any, cons: Any) -> dict[str, Any]:
    """API key 的客户端校验事实（评审 m6）。

    ``set_API_key`` 自身只做 ``setattr``（源码落盘）；
    但 ``login`` 在发出任何 I/O 之前调用 ``valid_API_key`` 做长度/前缀/字符集/校验和
    检查，格式不符直接返回 ``BSERR_APIKey_FORMAT_INCORRECT``。只读 setter 会得出
    「客户端完全不校验」的错误结论，因此校验器源码与 login 中的调用点一并落盘。
    """
    loginout = __import__("baostock.login.loginout", fromlist=["loginout"])
    return {
        "setter_exists": hasattr(bs, "set_API_key"),
        "setter_source": _source_of(bs, "set_API_key"),
        "validator_source": _source_of(loginout, "valid_API_key"),
        "validator_call_site": _source_containing(loginout, "login", "valid_API_key"),
        "server_error_code_for_bad_format": getattr(cons, "BSERR_APIKey_FORMAT_INCORRECT", None),
    }


def _source_of(obj: Any, name: str) -> str | None:
    """取属性源码文本（源码不可得则返回 None）：让「未做校验」可被直接核对。"""
    try:
        return inspect.getsource(getattr(obj, name))
    except (OSError, TypeError):
        return None


def _source_containing(module: Any, func_name: str, needle: str) -> list[str] | None:
    """函数源码中包含 needle 的行（去缩进）：证明调用点存在且可核对。"""
    func = getattr(module, func_name, None)
    if func is None:
        return None
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return None
    return [line.strip() for line in source.splitlines() if needle in line]


def _tcp_probe(host: str, port: int) -> dict[str, Any]:
    try:
        started = time.monotonic()
        with socket.create_connection((host, port), timeout=5):
            return {
                "host": host,
                "connect_ok": True,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
    except Exception as exc:
        return {"host": host, "connect_ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def _login(bs: Any) -> dict[str, Any]:
    result = bs.login()
    if result is None:
        return {
            **_verdict(False, "login returns a ResultData with error_code == '0'"),
            "error_code": "CLIENT_RETURNED_NONE",
            "error_msg": "login returned None",
        }
    return {
        **_verdict(
            result.error_code == "0",
            "login returns a ResultData with error_code == '0'",
        ),
        "error_code": result.error_code,
        "error_msg": result.error_msg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BaoStock capability spike")
    parser.add_argument(
        "--timeout", type=float, default=TIMEOUT_S, help="per-check timeout seconds"
    )
    parser.add_argument("--json", type=str, default=None, help="also write the report to this path")
    args = parser.parse_args()

    # spike 自身的双保险：即使 daemon 线程泄漏，进程也不会永久挂住（检查线程另有硬超时）
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(args.timeout)
    try:
        return _run_checks(args)
    finally:
        socket.setdefaulttimeout(previous_timeout)


def _run_checks(args: argparse.Namespace) -> int:
    try:
        import baostock as bs
    except ImportError as exc:
        # 失败也必须落盘成 artifact，否则「为什么没有产物」无从追溯（评审 m8）
        report: dict[str, Any] = {
            "spike": "baostock-capability",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "timeout_s": args.timeout,
            "status": "error",
            "checks": {},
            "checks_total": 0,
            "checks_ok": 0,
            "errors": ["import"],
            "unverified": [],
            "reason": f"baostock not importable: {exc}",
        }
        _emit(report, args.json)
        return 1

    checks: list[tuple[str, CheckFn]] = [
        ("client_info", lambda: check_client_info(bs)),
        ("login", lambda: _login(bs)),
        ("trade_dates", lambda: check_trade_dates(bs)),
        ("adjust_factor", lambda: check_adjust_factor(bs)),
        ("factor_normalization", lambda: check_factor_normalization(bs)),
        ("factor_price_mapping", lambda: check_factor_price_mapping(bs)),
        ("index_constituents", lambda: check_index_constituents(bs)),
        ("suspension", lambda: check_suspension(bs)),
        ("symbol_lifecycle", lambda: check_symbol_lifecycle(bs)),
        ("publication_lag", lambda: check_publication_lag(bs)),
        ("error_semantics", lambda: check_error_semantics(bs)),
    ]

    report = {
        "spike": "baostock-capability",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timeout_s": args.timeout,
        "checks": {},
    }

    for name, fn in checks:
        status, evidence = _bounded(fn, args.timeout)
        status, evidence = _enforce_confirmation(name, status, evidence)
        report["checks"][name] = {"status": status, **evidence}
        print(f"[{status}] {name}: {json.dumps(evidence, ensure_ascii=False)[:300]}", flush=True)

    report["errors"] = [
        name for name, payload in report["checks"].items() if payload["status"] == "error"
    ]
    report["unverified"] = [
        name for name, payload in report["checks"].items() if payload["status"] == "unverified"
    ]
    report["checks_total"] = len(report["checks"])
    report["checks_ok"] = sum(
        1 for payload in report["checks"].values() if payload["status"] == "ok"
    )
    report["status"] = (
        "ok"
        if not report["errors"] and not report["unverified"]
        else "+".join(
            label
            for label, present in (
                ("error", report["errors"]),
                ("unverified", report["unverified"]),
            )
            if present
        )
    )
    _emit(report, args.json)

    if report["errors"] and report["unverified"]:
        return 3
    if report["errors"]:
        return 1
    if report["unverified"]:
        return 2
    return 0


def _emit(report: dict[str, Any], json_path: str | None) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if json_path:
        _write_atomic(json_path, payload + "\n")
    print(payload, flush=True)


def _write_atomic(path: str, text: str) -> None:
    """先写同目录临时文件再 rename，避免中途失败留下半截 artifact。"""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".spike-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


if __name__ == "__main__":
    sys.exit(main())
