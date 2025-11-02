import React, { useEffect, useRef, useCallback, useState } from 'react';
import * as echarts from 'echarts';
import { Card, Button, Space, message } from 'antd';
import { StockData, PredictionData } from '../types';
import StockAPI from '../services/api';

interface KLineChartProps {
  title: string;
  historicalData: StockData[];
  predictionData: PredictionData[];
  loading?: boolean;
}

const KLineChartAdvanced: React.FC<KLineChartProps> = ({
  title,
  historicalData,
  predictionData,
  loading = false
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const [actualData, setActualData] = useState<StockData[]>([]);
  const [showActual, setShowActual] = useState(false);
  const [loadingActual, setLoadingActual] = useState(false);

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

  // 获取实际数据用于对比
  const fetchActualData = useCallback(async () => {
    if (predictionData.length === 0) return;

    try {
      setLoadingActual(true);
      const startDate = predictionData[0].date;
      const endDate = predictionData[predictionData.length - 1].date;
      const stockCode = title.match(/\((\d+)\)/)?.[1] || '';

      if (stockCode) {
        const response = await StockAPI.getStockActualData(stockCode, startDate, endDate);
        if (response.success) {
          setActualData(response.data.actual);
          setShowActual(true);
          message.success('实际数据加载成功');
        } else {
          message.warning('无法获取实际数据，将仅显示预测数据');
        }
      }
    } catch (error) {
      console.error('Failed to fetch actual data:', error);
      message.warning('获取实际数据失败');
    } finally {
      setLoadingActual(false);
    }
  }, [predictionData, title]);

  const renderChart = useCallback(() => {
    if (!chartInstance.current || historicalData.length === 0) {
      console.log('No data or chart instance not ready');
      return;
    }

    console.log('Rendering advanced chart with data:', {
      historicalCount: historicalData.length,
      predictionCount: predictionData.length,
      actualCount: actualData.length,
      showActual
    });

    try {
      // 历史数据K线图格式: [open, close, low, high]
      const historicalValues = historicalData.map(item => [
        parseFloat(item.open.toFixed(2)),
        parseFloat(item.close.toFixed(2)),
        parseFloat(item.low.toFixed(2)),
        parseFloat(item.high.toFixed(2))
      ]);

      // 预测数据K线图格式
      const predictionValues = predictionData.map(item => [
        parseFloat(item.open.toFixed(2)),
        parseFloat(item.close.toFixed(2)),
        parseFloat(item.low.toFixed(2)),
        parseFloat(item.high.toFixed(2))
      ]);

      // 实际数据K线图格式
      const actualValues = actualData.map(item => [
        parseFloat(item.open.toFixed(2)),
        parseFloat(item.close.toFixed(2)),
        parseFloat(item.low.toFixed(2)),
        parseFloat(item.high.toFixed(2))
      ]);

      // 日期数据
      const historicalDates = historicalData.map(item => item.date);
      const predictionDates = predictionData.map(item => item.date);
      const actualDates = actualData.map(item => item.date);

      // 合并所有日期用于X轴
      const allDates = [...historicalDates, ...predictionDates];
      if (showActual && actualDates.length > 0) {
        // 确保实际数据日期与预测数据日期对齐
        allDates.push(...actualDates);
      }

      console.log('Processed chart data:', {
        historicalValues: historicalValues.length,
        predictionValues: predictionValues.length,
        actualValues: actualValues.length,
        allDates: allDates.length
      });

      // 构建系列数据
      const series = [];

      // 1. 历史数据K线图
      series.push({
        name: '历史数据',
        type: 'candlestick',
        data: historicalValues,
        itemStyle: {
          color: '#ec0000',
          color0: '#00da3c',
          borderColor: '#8A0000',
          borderColor0: '#008F28'
        }
      });

      // 2. 预测数据线图（避免K线图的null值问题）
      if (predictionValues.length > 0) {
        // 使用收盘价作为预测线图数据
        const predictionClosePrices = predictionValues.map(item => item[1]);
        const alignedPredictionData = new Array(historicalDates.length).fill(null).concat(predictionClosePrices);
        
        series.push({
          name: '预测收盘价',
          type: 'line',
          data: alignedPredictionData,
          itemStyle: {
            color: '#66BB6A'
          },
          lineStyle: {
            color: '#66BB6A',
            type: 'dashed',
            width: 2
          },
          symbol: 'circle',
          symbolSize: 4,
          connectNulls: false
        });
      }

      // 3. 实际数据线图（如果启用）
      if (showActual && actualValues.length > 0) {
        // 使用收盘价作为实际线图数据
        const actualClosePrices = actualValues.map(item => item[1]);
        const alignedActualData = new Array(historicalDates.length).fill(null).concat(actualClosePrices);
        
        series.push({
          name: '实际收盘价',
          type: 'line',
          data: alignedActualData,
          itemStyle: {
            color: '#FF9800'
          },
          lineStyle: {
            color: '#FF9800',
            width: 2
          },
          symbol: 'diamond',
          symbolSize: 4,
          connectNulls: false
        });
      }

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
          },
          formatter: function(params: any) {
            let result = params[0].axisValue + '<br/>';
            params.forEach((param: any) => {
              if (param.data && param.data !== null) {
                const data = param.data;
                result += `${param.marker}${param.seriesName}<br/>`;
                result += `开盘: ${data[0]}<br/>`;
                result += `收盘: ${data[1]}<br/>`;
                result += `最低: ${data[2]}<br/>`;
                result += `最高: ${data[3]}<br/><br/>`;
              }
            });
            return result;
          }
        },
        legend: {
          data: ['历史数据', '预测收盘价', ...(showActual ? ['实际收盘价'] : [])],
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
            start: Math.max(0, 100 - (historicalDates.length / allDates.length) * 100),
            end: 100
          },
          {
            show: true,
            type: 'slider',
            top: '90%',
            start: Math.max(0, 100 - (historicalDates.length / allDates.length) * 100),
            end: 100
          }
        ],
        series: series
      };

      console.log('Chart option prepared');

      // 清空并设置新选项
      chartInstance.current.clear();
      chartInstance.current.setOption(option, true);
      
      console.log('Advanced chart rendered successfully');

    } catch (error) {
      console.error('Chart rendering error:', error);
    }
  }, [historicalData, predictionData, actualData, showActual, title]);

  useEffect(() => {
    renderChart();
  }, [renderChart]);

  return (
    <Card 
      loading={loading} 
      style={{ marginBottom: 24 }}
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>{title}</span>
          <Space>
            {predictionData.length > 0 && !showActual && (
              <Button 
                type="primary" 
                size="small"
                loading={loadingActual}
                onClick={fetchActualData}
              >
                加载实际数据对比
              </Button>
            )}
            {showActual && (
              <Button 
                size="small"
                onClick={() => setShowActual(false)}
              >
                隐藏实际数据
              </Button>
            )}
          </Space>
        </div>
      }
    >
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

export default KLineChartAdvanced;