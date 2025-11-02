import React from 'react';
import { Card } from 'antd';
import { StockData, PredictionData } from '../types';
import Plot from 'react-plotly.js';
import { Data, Layout, Config } from 'plotly.js';

interface PlotlyKLineChartProps {
  title: string;
  historicalData: StockData[];
  predictionData: PredictionData[];
  loading?: boolean;
}

const PlotlyKLineChart: React.FC<PlotlyKLineChartProps> = ({
  title,
  historicalData,
  predictionData,
  loading = false
}) => {

  // 准备图表数据
  const prepareChartData = (): Data[] => {
    const traces: Data[] = [];

    // 历史数据
    const historicalDates = historicalData.map((item: any) => item.date);
    const historicalOpen = historicalData.map((item: any) => item.open);
    const historicalHigh = historicalData.map((item: any) => item.high);
    const historicalLow = historicalData.map((item: any) => item.low);
    const historicalClose = historicalData.map((item: any) => item.close);

    // 预测数据
    const predictionDates = predictionData.map((item: any) => item.date);
    const predictionOpen = predictionData.map((item: any) => item.open);
    const predictionHigh = predictionData.map((item: any) => item.high);
    const predictionLow = predictionData.map((item: any) => item.low);
    const predictionClose = predictionData.map((item: any) => item.close);

    // 历史数据K线图 - 红涨绿跌
    traces.push({
      x: historicalDates,
      open: historicalOpen,
      high: historicalHigh,
      low: historicalLow,
      close: historicalClose,
      type: 'candlestick',
      name: '历史数据',
      increasing: { line: { color: '#ff0000' }, fillcolor: '#ff0000' }, // 红色上涨
      decreasing: { line: { color: '#00ff00' }, fillcolor: '#00ff00' }  // 绿色下跌
    } as Data);

    // 预测数据K线图 - 单一蓝色，虚线外框
    if (predictionData.length > 0) {
      traces.push({
        x: predictionDates,
        open: predictionOpen,
        high: predictionHigh,
        low: predictionLow,
        close: predictionClose,
        type: 'candlestick',
        name: '预测数据',
        increasing: { 
          line: { color: '#1890ff', width: 2, dash: 'dash' }, 
          fillcolor: '#1890ff' 
        }, // 蓝色上涨，虚线外框
        decreasing: { 
          line: { color: '#1890ff', width: 2, dash: 'dash' }, 
          fillcolor: '#1890ff' 
        }  // 蓝色下跌，虚线外框
      } as Data);
    }

    return traces;
  };

  const chartData: Data[] = prepareChartData();

  const layout: Partial<Layout> = {
    title: {
      text: title,
      font: { size: 16 }
    },
    xaxis: {
      title: { text: '日期' },
      type: 'category',
      rangeslider: { visible: false }
    },
    yaxis: {
      title: { text: '价格 (¥)' },
      autorange: true
    },
    legend: {
      x: 1,
      y: 1,
      xanchor: 'right',
      yanchor: 'top'
    },
    dragmode: 'zoom',
    showlegend: true,
    height: 500
  };

  const config: Partial<Config> = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
  };

  return (
    <Card 
      loading={loading} 
      style={{ marginBottom: 24 }}
      title={title}
    >
      <Plot
        data={chartData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '500px' }}
      />
    </Card>
  );
};

export default PlotlyKLineChart;