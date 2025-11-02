// 股票数据类型定义
export interface StockData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// 预测结果类型定义
export interface PredictionData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  confidence: number;
}

// 模型信息类型定义
export interface ModelInfo {
  model: string;
  accuracy: number;
  model_loaded: boolean;
  processing_time: string;
  available?: boolean;
}

// API响应类型定义
export interface APIResponse<T = any> {
  success: boolean;
  data: T;
  message: string;
}

// 预测请求类型定义
export interface PredictionRequest {
  code: string;
  prediction_days: number;
}

// 预测响应类型定义
export interface PredictionResponse {
  code: string;
  name: string;
  predictions: PredictionData[];
  model_info: ModelInfo;
}

// 股票历史数据响应类型定义
export interface StockHistoryResponse {
  code: string;
  name: string;
  history: StockData[];
}

// 股票信息类型定义
export interface StockInfo {
  code: string;
  name: string;
  market?: string;
}

// K线图数据类型定义
export interface KLineData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// 图表配置类型定义
export interface ChartOption {
  title: {
    text: string;
    left: 'center';
  };
  tooltip: {
    trigger: 'axis';
    axisPointer: {
      type: 'cross';
    };
  };
  legend: {
    data: string[];
    top: 30;
  };
  grid: {
    left: '10%';
    right: '10%';
    bottom: '15%';
  };
  xAxis: {
    type: 'category';
    data: string[];
    scale: true;
  };
  yAxis: {
    type: 'value';
    scale: true;
  };
  series: any[];
}