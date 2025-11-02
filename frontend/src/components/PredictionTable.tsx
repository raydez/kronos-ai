import React from 'react';
import { Card, Table, Tag, Progress } from 'antd';
import { PredictionData, ModelInfo } from '../types';
import type { ColumnsType } from 'antd/es/table';

interface PredictionTableProps {
  stockName: string;
  stockCode: string;
  predictions: PredictionData[];
  modelInfo: ModelInfo;
  loading?: boolean;
}

const PredictionTable: React.FC<PredictionTableProps> = ({
  stockName,
  stockCode,
  predictions,
  modelInfo,
  loading = false
}) => {
  const columns: ColumnsType<PredictionData> = [
    {
      title: '日期',
      dataIndex: 'date',
      key: 'date',
      render: (date: string) => {
        const dateObj = new Date(date);
        return dateObj.toLocaleDateString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit'
        });
      },
      sorter: (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime(),
    },
    {
      title: '开盘价',
      dataIndex: 'open',
      key: 'open',
      render: (value: number) => `¥${value.toFixed(2)}`,
      sorter: (a, b) => a.open - b.open,
    },
    {
      title: '最高价',
      dataIndex: 'high',
      key: 'high',
      render: (value: number) => `¥${value.toFixed(2)}`,
      sorter: (a, b) => a.high - b.high,
    },
    {
      title: '最低价',
      dataIndex: 'low',
      key: 'low',
      render: (value: number) => `¥${value.toFixed(2)}`,
      sorter: (a, b) => a.low - b.low,
    },
    {
      title: '收盘价',
      dataIndex: 'close',
      key: 'close',
      render: (value: number) => (
        <span style={{ fontWeight: 'bold', color: '#1890ff' }}>
          ¥{value.toFixed(2)}
        </span>
      ),
      sorter: (a, b) => a.close - b.close,
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Progress
            percent={Math.round(confidence * 100)}
            size="small"
            style={{ width: 60 }}
            strokeColor={
              confidence >= 0.8 ? '#52c41a' : 
              confidence >= 0.6 ? '#faad14' : '#ff4d4f'
            }
          />
          <Tag color={
            confidence >= 0.8 ? 'success' : 
            confidence >= 0.6 ? 'warning' : 'error'
          }>
            {(confidence * 100).toFixed(0)}%
          </Tag>
        </div>
      ),
      sorter: (a, b) => a.confidence - b.confidence,
    },
    {
      title: '涨跌幅',
      key: 'change',
      render: (_, record, index) => {
        if (index === 0) return '-';
        const prevClose = predictions[index - 1].close;
        const currentClose = record.close;
        const change = ((currentClose - prevClose) / prevClose) * 100;
        const color = change >= 0 ? '#f5222d' : '#52c41a';
        const sign = change >= 0 ? '+' : '';
        return (
          <span style={{ color, fontWeight: 'bold' }}>
            {sign}{change.toFixed(2)}%
          </span>
        );
      },
    },
  ];

  const getModelStatusTag = () => {
    if (!modelInfo.model_loaded) {
      return <Tag color="error">模型未加载</Tag>;
    }
    if (modelInfo.accuracy >= 0.8) {
      return <Tag color="success">高精度</Tag>;
    } else if (modelInfo.accuracy >= 0.6) {
      return <Tag color="warning">中等精度</Tag>;
    }
    return <Tag color="error">低精度</Tag>;
  };

  return (
    <Card 
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>
            预测结果 - {stockName} ({stockCode})
          </span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {getModelStatusTag()}
            <Tag color="blue">{modelInfo.model}</Tag>
            <Tag color="default">处理时间: {modelInfo.processing_time}</Tag>
          </div>
        </div>
      }
      loading={loading}
    >
      <Table
        columns={columns}
        dataSource={predictions}
        rowKey="date"
        pagination={false}
        size="middle"
        scroll={{ x: 800 }}
        summary={() => (
          <Table.Summary fixed>
            <Table.Summary.Row>
              <Table.Summary.Cell index={0} colSpan={4}>
                <strong>统计信息</strong>
              </Table.Summary.Cell>
              <Table.Summary.Cell index={4}>
                <strong>
                  平均: ¥{(predictions.reduce((sum, p) => sum + p.close, 0) / predictions.length).toFixed(2)}
                </strong>
              </Table.Summary.Cell>
              <Table.Summary.Cell index={5}>
                <strong>
                  平均: {(predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length * 100).toFixed(1)}%
                </strong>
              </Table.Summary.Cell>
              <Table.Summary.Cell index={6}>
                <strong>
                  总波动: {predictions.length > 1 ? 
                    ((predictions[predictions.length - 1].close - predictions[0].close) / predictions[0].close * 100).toFixed(2) : 0
                  }%
                </strong>
              </Table.Summary.Cell>
            </Table.Summary.Row>
          </Table.Summary>
        )}
      />
    </Card>
  );
};

export default PredictionTable;