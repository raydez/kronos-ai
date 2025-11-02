import axios from 'axios';
import {
  APIResponse,
  PredictionRequest,
  PredictionResponse,
  StockHistoryResponse,
  StockInfo,
  ModelInfo
} from '../types';

// 创建axios实例
const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('API Error:', error.message);
    return Promise.reject(error);
  }
);

// API服务类
export class StockAPI {
  // 健康检查
  static async healthCheck(): Promise<APIResponse<{ status: string; model: ModelInfo; timestamp: number }>> {
    const response = await api.get('/health');
    return response.data;
  }

  // 获取模型信息
  static async getModelInfo(): Promise<APIResponse<ModelInfo>> {
    const response = await api.get('/model/info');
    return response.data;
  }

  // 获取股票信息
  static async getStockInfo(code: string): Promise<APIResponse<StockInfo>> {
    const response = await api.get(`/stock/${code}`);
    return response.data;
  }

  // 获取股票历史数据
  static async getStockHistory(code: string, days: number = 30): Promise<APIResponse<StockHistoryResponse>> {
    const response = await api.get(`/stock/${code}/history`, {
      params: { days }
    });
    return response.data;
  }

  // 股票预测
  static async predictStock(request: PredictionRequest): Promise<APIResponse<PredictionResponse>> {
    const response = await api.post('/predict', request);
    return response.data;
  }

  // 批量预测
  static async batchPredict(codes: string[], predictionDays: number = 5): Promise<APIResponse<Record<string, any>>> {
    const response = await api.post('/batch-predict', codes, {
      params: { prediction_days: predictionDays }
    });
    return response.data;
  }

  // 重新加载模型
  static async reloadModel(): Promise<APIResponse<any>> {
    const response = await api.post('/model/reload');
    return response.data;
  }

  // 清空缓存
  static async clearCache(): Promise<APIResponse<any>> {
    const response = await api.delete('/cache');
    return response.data;
  }

  // 获取实际股票数据（用于对比预测结果）
  static async getStockActualData(code: string, startDate: string, endDate: string): Promise<APIResponse<{ code: string; start_date: string; end_date: string; actual: any[] }>> {
    const response = await api.get(`/stock/${code}/actual`, {
      params: { start_date: startDate, end_date: endDate }
    });
    return response.data;
  }
}

export default StockAPI;