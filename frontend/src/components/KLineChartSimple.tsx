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

const KLineChartSimple: React.FC<KLineChartProps> = ({
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

  const renderChart = useCallback(() => {
    if (!chartInstance.current || (historicalData.length === 0 && predictionData.length === 0)) {
      console.log('No data or chart instance not ready');
      return;
    }

    console.log('Rendering chart with data:', {
      historicalCount: historicalData.length,
      predictionCount: predictionData.length,
      historicalSample: historicalData.slice(0, 2),
      predictionSample: predictionData.slice(0, 2)
    });

    try {
      // 创建历史数据 - 使用与debug_kline.html相同的格式
      const historicalValues = historicalData.map(item => [
        parseFloat(item.open.toFixed(2)),  // open
        parseFloat(item.close.toFixed(2)),  // close
        parseFloat(item.low.toFixed(2)),   // low
        parseFloat(item.high.toFixed(2))    // high
      ]);

      // 创建预测数据
      const predictionValues = predictionData.map(item => [
        parseFloat(item.open.toFixed(2)),  // open
        parseFloat(item.close.toFixed(2)),  // close
        parseFloat(item.low.toFixed(2)),   // low
        parseFloat(item.high.toFixed(2))    // high
      ]);

      // 准备日期数据
      const historicalDates = historicalData.map(item => item.date);
      const predictionDates = predictionData.map(item => item.date);
      const allDates = [...historicalDates, ...predictionDates];

      console.log('Processed data:', {
        historicalValues: historicalValues.length,
        predictionValues: predictionValues.length,
        allDates: allDates.length,
        sampleHistorical: historicalValues[0],
        samplePrediction: predictionValues[0]
      });

      // 预测数据需要对齐 - 不使用null，而是创建两个独立的系列
      const alignedPredictionData = predictionValues;

      const option = {
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
                return value.substring(0, 10);
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
          // 预测线图（避免K线图的null值问题）
          ...(predictionValues.length > 0 ? [{
            name: '预测收盘价',
            type: 'line',
            data: new Array(historicalDates.length).fill(null).concat(predictionValues.map(item => item[1])),
            itemStyle: {
              color: '#ff9800'
            },
            lineStyle: {
              color: '#ff9800',
              type: 'dashed'
            }
          }] : [])
        ]
      };

      console.log('Chart option prepared:', option);

      // 清空并设置新选项
      chartInstance.current.clear();
      chartInstance.current.setOption(option, true);
      
      console.log('Chart rendered successfully');

    } catch (error) {
      console.error('Chart rendering error:', error);
    }
  }, [historicalData, predictionData, title]);

  useEffect(() => {
    renderChart();
  }, [renderChart]);

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

export default KLineChartSimple;