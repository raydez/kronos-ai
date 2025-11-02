import React from 'react';
import { Card } from 'antd';
import { StockData, PredictionData } from '../types';
import Plot from 'react-plotly.js';
import { Data, Layout, Config } from 'plotly.js';

interface PlotlyKLineChartProps {
  title: string;
  historicalData: StockData[];
  predictionData: PredictionData[];
  actualData?: StockData[]; // 预测时间段的实际数据
  loading?: boolean;
}

const PlotlyKLineChart: React.FC<PlotlyKLineChartProps> = ({
  title,
  historicalData,
  predictionData,
  actualData = [],
  loading = false
}) => {
  
  // 调试信息
  console.log('📊 PlotlyKLineChart 数据:', {
    历史数据条数: historicalData.length,
    预测数据条数: predictionData.length,
    实际数据条数: actualData.length,
    历史数据日期范围: historicalData.length > 0 ? `${historicalData[0].date} 到 ${historicalData[historicalData.length-1].date}` : '无',
    预测数据日期范围: predictionData.length > 0 ? `${predictionData[0].date} 到 ${predictionData[predictionData.length-1].date}` : '无',
    实际数据日期范围: actualData.length > 0 ? `${actualData[0].date} 到 ${actualData[actualData.length-1].date}` : '无'
  });

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

    // 预测时间段的实际数据K线图 - 橙色边框，半透明填充
    if (actualData && actualData.length > 0) {
      const actualDates = actualData.map((item: any) => item.date);
      const actualOpen = actualData.map((item: any) => item.open);
      const actualHigh = actualData.map((item: any) => item.high);
      const actualLow = actualData.map((item: any) => item.low);
      const actualClose = actualData.map((item: any) => item.close);

      traces.push({
        x: actualDates,
        open: actualOpen,
        high: actualHigh,
        low: actualLow,
        close: actualClose,
        type: 'candlestick',
        name: '实际走势',
        increasing: { 
          line: { color: '#ff8c00', width: 2 }, 
          fillcolor: 'rgba(255, 140, 0, 0.3)' 
        }, // 橙色上涨，半透明填充
        decreasing: { 
          line: { color: '#ff8c00', width: 2 }, 
          fillcolor: 'rgba(255, 140, 0, 0.3)' 
        }  // 橙色下跌，半透明填充
      } as Data);
    }

    // 预测数据K线图 - 蓝色虚线
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
          fillcolor: 'rgba(24, 144, 255, 0.2)' 
        }, // 蓝色上涨，虚线外框，半透明填充
        decreasing: { 
          line: { color: '#1890ff', width: 2, dash: 'dash' }, 
          fillcolor: 'rgba(24, 144, 255, 0.2)' 
        }  // 蓝色下跌，虚线外框，半透明填充
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