import React from 'react';
import { Form, Input, Select, Button, Card, message, Space, DatePicker } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { PredictionRequest } from '../types';
import dayjs from 'dayjs';

const { Option } = Select;

interface PredictionFormProps {
  onPredict: (request: PredictionRequest) => void;
  loading?: boolean;
}

const PredictionForm: React.FC<PredictionFormProps> = ({ onPredict, loading = false }) => {
  const [form] = Form.useForm();

  const handleSubmit = (values: any) => {
    const { code, predictionDays, startDate } = values;
    
    // 验证股票代码格式
    if (!/^\d{6}$/.test(code)) {
      message.error('股票代码必须是6位数字');
      return;
    }

    onPredict({
      code: code.trim(),
      prediction_days: predictionDays,
      start_date: startDate ? startDate.format('YYYY-MM-DD') : undefined
    });
  };



  return (
    <Card title="股票预测" style={{ marginBottom: 24 }}>
      <Form
        form={form}
        layout="inline"
        onFinish={handleSubmit}
        initialValues={{
          predictionDays: 5,
          startDate: dayjs()
        }}
      >
        <Form.Item
          name="code"
          rules={[
            { required: true, message: '请输入股票代码' },
            { len: 6, message: '股票代码必须是6位数字' }
          ]}
        >
          <Input
            placeholder="请输入6位股票代码"
            style={{ width: 200 }}
            maxLength={6}
          />
        </Form.Item>

        <Form.Item
          name="startDate"
          label="开始预测日期"
        >
          <DatePicker 
            style={{ width: 150 }} 
            format="YYYY-MM-DD"
            placeholder="选择开始日期"
            disabledDate={(current) => {
              // 禁用未来日期
              return current && current > dayjs().endOf('day');
            }}
          />
        </Form.Item>

        <Form.Item
          name="predictionDays"
          label="预测天数"
        >
          <Select style={{ width: 120 }}>
            <Option value={3}>3天</Option>
            <Option value={5}>5天</Option>
            <Option value={7}>7天</Option>
            <Option value={10}>10天</Option>
            <Option value={14}>14天</Option>
            <Option value={20}>20天</Option>
          </Select>
        </Form.Item>

        <Form.Item>
          <Button
            type="primary"
            htmlType="submit"
            icon={<SearchOutlined />}
            loading={loading}
          >
            开始预测
          </Button>
        </Form.Item>
      </Form>

      <div style={{ marginTop: 16, color: '#666', fontSize: '12px' }}>
        <Space>
          <span>示例代码：</span>
          <Button 
            type="link" 
            size="small" 
            onClick={() => form.setFieldsValue({ code: '600000' })}
          >
            600000 (浦发银行)
          </Button>
          <Button 
            type="link" 
            size="small" 
            onClick={() => form.setFieldsValue({ code: '000001' })}
          >
            000001 (平安银行)
          </Button>
          <Button 
            type="link" 
            size="small" 
            onClick={() => form.setFieldsValue({ code: '600036' })}
          >
            600036 (招商银行)
          </Button>
        </Space>
      </div>
    </Card>
  );
};

export default PredictionForm;