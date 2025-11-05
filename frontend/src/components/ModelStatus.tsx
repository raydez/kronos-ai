import React, { useState, useEffect, useCallback } from 'react';
import { Card, Statistic, Row, Col, Tag, Button, Space, message, Select, Modal } from 'antd';
import { 
  ReloadOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  SwapOutlined 
} from '@ant-design/icons';
import { ModelInfo } from '../types';
import StockAPI from '../services/api';

const { Option } = Select;

interface ModelStatusProps {
  onModelUpdate?: (modelInfo: ModelInfo) => void;
}

interface ModelOption {
  id: string;
  name: string;
  description: string;
  params: string;
  context_length: number;
  available: boolean;
}

const ModelStatus: React.FC<ModelStatusProps> = ({ onModelUpdate }) => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
  const [currentModel, setCurrentModel] = useState<string>('');
  const [switchModalVisible, setSwitchModalVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [switching, setSwitching] = useState(false);

  // 模型中文描述映射
  const modelDescriptions: Record<string, string> = {
    'kronos-mini': '轻量级模型，适合快速预测和资源受限环境',
    'kronos-small': '平衡性能和速度的标准模型，适合日常使用',
    'kronos-base': '高精度基础模型，适合深度分析和精确预测'
  };

  const fetchModelInfo = useCallback(async () => {
    try {
      setLoading(true);
      const response = await StockAPI.getModelInfo();
      if (response.success) {
        setModelInfo(response.data);
        onModelUpdate?.(response.data);
      }
    } catch (error) {
      console.error('获取模型信息失败:', error);
    } finally {
      setLoading(false);
    }
  }, [onModelUpdate]);

  const fetchAvailableModels = useCallback(async () => {
    try {
      const response = await StockAPI.getAvailableModels();
      if (response.success) {
        setAvailableModels(response.data.models || []);
        setCurrentModel(response.data.current_model || '');
      }
    } catch (error) {
      console.error('获取可用模型列表失败:', error);
    }
  }, []);

  useEffect(() => {
    fetchModelInfo();
    fetchAvailableModels();
    // 每30秒刷新一次模型状态
    const interval = setInterval(fetchModelInfo, 30000);
    return () => clearInterval(interval);
  }, [fetchModelInfo, fetchAvailableModels]);

  const handleReloadModel = async () => {
    try {
      setReloading(true);
      
      // 显示加载中的提示
      const hideLoading = message.loading('正在加载模型，请稍候...', 0);
      
      const response = await StockAPI.reloadModel();
      
      hideLoading();
      
      if (response.success) {
        const downloadInfo = response.data?.download_info;
        let successMsg = '模型重新加载成功';
        
        // 如果有下载信息，显示详细消息
        if (downloadInfo) {
          if (downloadInfo.tokenizer_downloaded || downloadInfo.model_downloaded) {
            const downloaded = [];
            if (downloadInfo.tokenizer_downloaded) downloaded.push('Tokenizer');
            if (downloadInfo.model_downloaded) downloaded.push('模型');
            successMsg += `（已从HuggingFace下载: ${downloaded.join('、')}）`;
          } else {
            successMsg += '（从本地缓存加载）';
          }
        }
        
        message.success(successMsg, 5);
        await fetchModelInfo();
      } else {
        // 显示详细的错误信息
        const errorMsg = response.data?.error || response.message || '模型重新加载失败';
        message.error(errorMsg, 10);
      }
    } catch (error: any) {
      console.error('重新加载模型失败:', error);
      
      // 显示更详细的错误信息
      let errorMsg = '重新加载模型失败';
      if (error.response?.data?.detail) {
        errorMsg = error.response.data.detail;
      } else if (error.message) {
        errorMsg += ': ' + error.message;
      }
      
      message.error(errorMsg, 10);
    } finally {
      setReloading(false);
    }
  };

  const getStatusIcon = () => {
    if (!modelInfo) return <ExclamationCircleOutlined />;
    if (modelInfo.model_loaded) return <CheckCircleOutlined />;
    return <CloseCircleOutlined />;
  };

  const getStatusColor = () => {
    if (!modelInfo) return 'warning';
    if (modelInfo.model_loaded) return 'success';
    return 'error';
  };

  const getStatusText = () => {
    if (!modelInfo) return '状态未知';
    if (modelInfo.model_loaded) return '模型已加载';
    return '模型未加载';
  };

  const getAccuracyColor = () => {
    if (!modelInfo) return 'default';
    if (modelInfo.accuracy >= 0.8) return 'success';
    if (modelInfo.accuracy >= 0.6) return 'warning';
    return 'error';
  };

  const getAccuracyText = () => {
    if (!modelInfo) return '未知';
    if (modelInfo.accuracy >= 0.8) return '高精度';
    if (modelInfo.accuracy >= 0.6) return '中等精度';
    return '低精度';
  };

  const handleShowSwitchModal = () => {
    setSelectedModel(currentModel);
    setSwitchModalVisible(true);
  };

  const handleSwitchModel = async () => {
    if (!selectedModel) {
      message.warning('请选择要切换的模型');
      return;
    }

    if (selectedModel === currentModel) {
      message.info('该模型已经加载');
      setSwitchModalVisible(false);
      return;
    }

    try {
      setSwitching(true);
      
      // 显示加载中的提示
      const hideLoading = message.loading('正在切换模型，请稍候...', 0);
      
      const response = await StockAPI.switchModel(selectedModel);
      
      hideLoading();
      
      if (response.success) {
        const downloadInfo = response.data?.download_info;
        const unloadedModel = response.data?.unloaded_model;
        
        let successMsg = '模型切换成功';
        
        // 显示卸载信息
        if (unloadedModel) {
          successMsg = `已从 ${unloadedModel} 切换到 ${selectedModel}`;
        }
        
        // 如果有下载信息，显示详细消息
        if (downloadInfo) {
          if (downloadInfo.tokenizer_downloaded || downloadInfo.model_downloaded) {
            const downloaded = [];
            if (downloadInfo.tokenizer_downloaded) downloaded.push('Tokenizer');
            if (downloadInfo.model_downloaded) downloaded.push('模型');
            successMsg += `（已从HuggingFace下载: ${downloaded.join('、')}）`;
          } else {
            successMsg += '（从本地缓存加载）';
          }
        }
        
        message.success(successMsg, 5);
        await fetchModelInfo();
        await fetchAvailableModels();
        setSwitchModalVisible(false);
      } else {
        // 显示详细的错误信息
        const errorMsg = response.data?.error || response.message || '模型切换失败';
        message.error(errorMsg, 10);
      }
    } catch (error: any) {
      console.error('切换模型失败:', error);
      
      // 显示更详细的错误信息
      let errorMsg = '切换模型失败';
      if (error.response?.data?.detail) {
        errorMsg = error.response.data.detail;
      } else if (error.message) {
        errorMsg += ': ' + error.message;
      }
      
      message.error(errorMsg, 10);
    } finally {
      setSwitching(false);
    }
  };

  const handleCancelSwitch = () => {
    setSwitchModalVisible(false);
    setSelectedModel(currentModel);
  };

  return (
    <Card
      title={
        <Space>
          模型状态
          <Tag color={getStatusColor()} icon={getStatusIcon()}>
            {getStatusText()}
          </Tag>
        </Space>
      }
      extra={
        <Space>
          <Button
            icon={<SwapOutlined />}
            onClick={handleShowSwitchModal}
            size="small"
          >
            切换模型
          </Button>
          <Button
            type="primary"
            icon={<ReloadOutlined />}
            onClick={handleReloadModel}
            loading={reloading}
            size="small"
          >
            重新加载
          </Button>
        </Space>
      }
      loading={loading}
    >
      {modelInfo && (
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="当前模型"
              value={modelInfo.model}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="预测精度"
              value={modelInfo.accuracy * 100}
              precision={1}
              suffix="%"
              valueStyle={{ 
                fontSize: '16px',
                color: modelInfo.accuracy >= 0.8 ? '#52c41a' : 
                       modelInfo.accuracy >= 0.6 ? '#faad14' : '#ff4d4f'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="处理时间"
              value={modelInfo.processing_time}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
          <Col span={6}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ marginBottom: '8px', fontSize: '14px', color: '#666' }}>
                精度等级
              </div>
              <Tag color={getAccuracyColor()}>
                {getAccuracyText()}
              </Tag>
            </div>
          </Col>
        </Row>
      )}
      
      {modelInfo && !modelInfo.model_loaded && (
        <div style={{ 
          marginTop: 16, 
          padding: 12, 
          backgroundColor: '#fff2e8', 
          border: '1px solid #ffbb96',
          borderRadius: 6
        }}>
          <Space>
            <ExclamationCircleOutlined style={{ color: '#fa8c16' }} />
            <span style={{ color: '#d46b08' }}>
              模型当前未加载，预测功能可能不可用。点击"重新加载"按钮尝试加载模型。
            </span>
          </Space>
        </div>
      )}

      {modelInfo && modelInfo.model_loaded && modelInfo.accuracy < 0.6 && (
        <div style={{ 
          marginTop: 16, 
          padding: 12, 
          backgroundColor: '#fff1f0', 
          border: '1px solid #ffccc7',
          borderRadius: 6
        }}>
          <Space>
            <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
            <span style={{ color: '#cf1322' }}>
              模型预测精度较低，建议谨慎使用预测结果。
            </span>
          </Space>
        </div>
      )}

      <Modal
        title="切换模型"
        open={switchModalVisible}
        onOk={handleSwitchModel}
        onCancel={handleCancelSwitch}
        confirmLoading={switching}
        okText="切换"
        cancelText="取消"
        width={600}
      >
        <div style={{ marginBottom: 16 }}>
          <p style={{ color: '#666', marginBottom: 16 }}>
            选择要切换的模型。切换时会自动卸载当前模型以节省资源。
            如果本地没有模型文件，会自动从 HuggingFace 下载。
          </p>
          <Select
            value={selectedModel}
            onChange={setSelectedModel}
            style={{ width: '100%' }}
            placeholder="选择模型"
          >
            {availableModels.map((model) => (
              <Option key={model.id} value={model.id}>
                <div style={{ padding: '8px 0' }}>
                  <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
                    {model.name}
                    {model.id === currentModel && (
                      <Tag color="blue" style={{ marginLeft: 8 }}>当前</Tag>
                    )}
                  </div>
                  <div style={{ fontSize: '12px', color: '#666' }}>
                    {modelDescriptions[model.id] || model.description}
                  </div>
                  <div style={{ fontSize: '12px', color: '#999', marginTop: 4 }}>
                    参数量: {model.params} | 上下文长度: {model.context_length}
                  </div>
                </div>
              </Option>
            ))}
          </Select>
        </div>

        {selectedModel && selectedModel !== currentModel && (
          <div style={{ 
            padding: 12, 
            backgroundColor: '#e6f7ff', 
            border: '1px solid #91d5ff',
            borderRadius: 6,
            marginTop: 16
          }}>
            <Space>
              <ExclamationCircleOutlined style={{ color: '#1890ff' }} />
              <span style={{ color: '#096dd9' }}>
                切换模型会卸载当前模型并加载新模型，可能需要一些时间。
              </span>
            </Space>
          </div>
        )}
      </Modal>
    </Card>
  );
};

export default ModelStatus;