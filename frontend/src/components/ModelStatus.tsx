import React, { useState, useEffect, useCallback } from 'react';
import { Card, Statistic, Row, Col, Tag, Button, Space, message } from 'antd';
import { 
  ReloadOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  CloseCircleOutlined 
} from '@ant-design/icons';
import { ModelInfo } from '../types';
import StockAPI from '../services/api';

interface ModelStatusProps {
  onModelUpdate?: (modelInfo: ModelInfo) => void;
}

const ModelStatus: React.FC<ModelStatusProps> = ({ onModelUpdate }) => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [reloading, setReloading] = useState(false);

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

  useEffect(() => {
    fetchModelInfo();
    // 每30秒刷新一次模型状态
    const interval = setInterval(fetchModelInfo, 30000);
    return () => clearInterval(interval);
  }, [fetchModelInfo]);

  const handleReloadModel = async () => {
    try {
      setReloading(true);
      const response = await StockAPI.reloadModel();
      if (response.success) {
        message.success('模型重新加载成功');
        await fetchModelInfo();
      } else {
        message.error('模型重新加载失败');
      }
    } catch (error) {
      console.error('重新加载模型失败:', error);
      message.error('重新加载模型失败');
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
        <Button
          type="primary"
          icon={<ReloadOutlined />}
          onClick={handleReloadModel}
          loading={reloading}
          size="small"
        >
          重新加载
        </Button>
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
    </Card>
  );
};

export default ModelStatus;