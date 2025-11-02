import React, { useEffect, useRef, useCallback } from 'react';
import * as echarts from 'echarts';
import { Card } from 'antd';
import { StockData, PredictionData } from '../types';

interface KLineChartProps {
  title: string;
  historicalData: StockData[];
  predictionData: PredictionData[];
  loading?: boolean;
}

const KLineChart: React.FC<KLineChartProps> = ({
  title,
  historicalData,
  predictionData,
  loading = false
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (chartRef.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    return () => {
      if (chartInstance.current) {
        chartInstance.current.dispose();
      }
    };
  }, []);

  const generateChartOption = useCallback((historical: StockData[], predictions: PredictionData[]) => {
    // 数据验证和清理
    const cleanHistorical = historical.filter(item => 
      item && 
      typeof item.open === 'number' && 
      typeof item.high === 'number' && 
      typeof item.low === 'number' && 
      typeof item.close === 'number' &&
      !isNaN(item.open) && !isNaN(item.high) && !isNaN(item.low) && !isNaN(item.close)
    );

    const cleanPredictions = predictions.filter(item => 
      item && 
      typeof item.open === 'number' && 
      typeof item.high === 'number' && 
      typeof item.low === 'number' && 
      typeof item.close === 'number' &&
      !isNaN(item.open) && !isNaN(item.high) && !isNaN(item.low) && !isNaN(item.close)
    );

    // 准备历史数据 - ECharts candlestick格式: [open, close, low, high]
    const historicalDates = cleanHistorical.map(item => item.date);
    const historicalValues = cleanHistorical.map(item => [
      parseFloat(item.open.toFixed(2)),
      parseFloat(item.close.toFixed(2)),
      parseFloat(item.low.toFixed(2)),
      parseFloat(item.high.toFixed(2))
    ]);

    // 准备预测数据
    const predictionDates = cleanPredictions.map(item => item.date);
    const predictionValues = cleanPredictions.map(item => [
      parseFloat(item.open.toFixed(2)),
      parseFloat(item.close.toFixed(2)),
      parseFloat(item.low.toFixed(2)),
      parseFloat(item.high.toFixed(2))
    ]);

    // 验证预测数据有效性
    const validPredictionValues = predictionValues.filter(item => 
      item && 
      item.length === 4 && 
      item.every(val => typeof val === 'number' && !isNaN(val))
    );

    const allDates = [...historicalDates, ...predictionDates];

    return {
      title: {
        text: title,
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold'
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['历史数据', '预测数据'],
        top: 30
      },
      grid: {
        left: '10%',
        right: '10%',
        bottom: '15%'
      },
      xAxis: {
        type: 'category',
        data: allDates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        splitLine: { show: false },
        axisLabel: {
          rotate: 45,
          formatter: function(value: any) {
            if (typeof value === 'string') {
              return value.substring(0, 10); // 只显示日期部分
            }
            return value;
          }
        }
      },
      yAxis: {
        scale: true,
        splitArea: {
          show: true
        },
        axisLabel: {
          formatter: '¥{value}'
        }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 50,
          end: 100
        },
        {
          show: true,
          type: 'slider',
          top: '90%',
          start: 50,
          end: 100
        }
      ],
      series: [
        // 历史K线图
        {
          name: '历史数据',
          type: 'candlestick',
          data: historicalValues,
          itemStyle: {
            color: '#ec0000',
            color0: '#00da3c',
            borderColor: '#8A0000',
            borderColor0: '#008F28'
          }
        },
        // 预测K线图
        ...(validPredictionValues.length > 0 ? [{
          name: '预测数据',
          type: 'candlestick',
          data: new Array(historicalDates.length).fill(null).concat(validPredictionValues),
          itemStyle: {
            color: '#ff9800',
            color0: '#ff9800',
            borderColor: '#f57c00',
            borderColor0: '#f57c00'
          }
        }] : [])
      ]
    };
  }, [title]);

  useEffect(() => {
    if (chartInstance.current) {
      try {
        if (historicalData.length > 0 || predictionData.length > 0) {
          console.log('Chart data:', { 
            historicalCount: historicalData.length, 
            predictionCount: predictionData.length,
            historicalSample: historicalData.slice(0, 2),
            predictionSample: predictionData.slice(0, 2)
          });
          
          const option = generateChartOption(historicalData, predictionData);
          console.log('Chart option:', option);
          
          // 清空之前的数据
          chartInstance.current.clear();
          // 设置新选项
          chartInstance.current.setOption(option, true);
          
          console.log('Chart rendered successfully');
        } else {
          console.log('No data to render chart');
        }
      } catch (error) {
        console.error('Chart rendering error:', error);
        if (error instanceof Error) {
          console.error('Error details:', {
            message: error.message,
            stack: error.stack,
            historicalData: historicalData.slice(0, 3),
            predictionData: predictionData.slice(0, 3)
          });
        }
      }
    } else {
      console.log('Chart instance not available');
    }
  }, [historicalData, predictionData, generateChartOption]);

  return (
    <Card loading={loading} style={{ marginBottom: 24 }}>
      <div 
        ref={chartRef} 
        style={{ 
          width: '100%', 
          height: '500px',
          minHeight: '400px'
        }} 
      />
    </Card>
  );
};

export default KLineChart;