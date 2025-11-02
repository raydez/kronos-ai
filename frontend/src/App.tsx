import React, { useState } from 'react';
import { Layout, Typography, message, Spin, Row, Col } from 'antd';
import PredictionForm from './components/PredictionForm';
import PlotlyKLineChart from './components/PlotlyKLineChart';
import PredictionTable from './components/PredictionTable';
import ModelStatus from './components/ModelStatus';
import StockAPI from './services/api';
import { PredictionRequest, PredictionResponse, StockData } from './types';
import 'antd/dist/reset.css';

const { Header, Content } = Layout;
const { Title } = Typography;

const App: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [predictionData, setPredictionData] = useState<PredictionResponse | null>(null);
  const [historicalData, setHistoricalData] = useState<StockData[]>([]);
  const [actualData, setActualData] = useState<StockData[]>([]); // 预测时间段的实际数据


  const handlePredict = async (request: PredictionRequest) => {
    try {
      setLoading(true);
      message.loading('正在预测中，请稍候...', 0);

      // 获取历史数据
      const historyResponse = await StockAPI.getStockHistory(request.code, 90);
      if (historyResponse.success) {
        console.log('Historical data received:', historyResponse.data.history);
        // 处理日期格式
        let processedHistory = historyResponse.data.history.map((item: any) => ({
          ...item,
          date: item.date.split('T')[0] // 只保留日期部分
        }));
        
        // 如果有预测开始日期，只保留该日期之前的历史数据
        if (request.start_date) {
          processedHistory = processedHistory.filter((item: any) => item.date < request.start_date!);
          console.log('Filtered historical data to dates before', request.start_date, ':', processedHistory.length, 'records');
        }
        
        console.log('Processed historical data:', processedHistory);
        setHistoricalData(processedHistory);
      }

      // 进行预测
      const predictResponse = await StockAPI.predictStock(request);
      if (predictResponse.success) {
        console.log('Prediction data received:', predictResponse.data);
        // 处理预测数据日期格式
        const processedPredictions = predictResponse.data.predictions.map((item: any) => ({
          ...item,
          date: item.date.split('T')[0] // 只保留日期部分
        }));
        const processedPredictionData = {
          ...predictResponse.data,
          predictions: processedPredictions
        };
        console.log('Processed prediction data:', processedPredictionData);
        setPredictionData(processedPredictionData);

        // 获取预测时间段的实际数据（用于对比）
        if (processedPredictions.length > 0) {
          const firstPredictionDate = processedPredictions[0].date;
          
          // 检查预测开始时间是否早于今天
          const today = new Date().toISOString().split('T')[0];
          
          if (firstPredictionDate <= today) {
            try {
              console.log('🔍 获取实际数据用于对比...');
              
              // 计算需要获取的实际数据范围，确保有足够的交易日
              const targetDays = request.prediction_days;
              // 扩展结束日期，确保有足够的交易日（通常需要2-3倍的日历天数）
              const extendedEndDate = new Date(firstPredictionDate);
              extendedEndDate.setDate(extendedEndDate.getDate() + Math.ceil(targetDays * 2.5));
              
              const actualResponse = await StockAPI.getStockActualData(
                request.code, 
                firstPredictionDate, 
                extendedEndDate.toISOString().split('T')[0]
              );
              
              if (actualResponse.success && actualResponse.data.actual) {
                const processedActual = actualResponse.data.actual.map((item: any) => ({
                  ...item,
                  date: item.date.split('T')[0] // 只保留日期部分
                }));
                
                // 重新设计的对齐逻辑：只显示有预测和实际数据的交易日
                let alignedActual: any[] = [];
                let alignedPredictions: any[] = [];
                
                // 找出预测数据和实际数据中都存在的交易日
                const actualDates = processedActual.map((actual: any) => actual.date);
                const predictionDates = processedPredictions.map((prediction: any) => prediction.date);
                
                // 获取共同的交易日（既在预测中也在实际数据中）
                const commonTradingDays = actualDates.filter((date: string) => 
                  predictionDates.includes(date)
                );
                
                if (commonTradingDays.length > 0) {
                  // 使用共同的交易日作为对齐基准
                  alignedActual = processedActual.filter((actual: any) => 
                    commonTradingDays.includes(actual.date)
                  );
                  alignedPredictions = processedPredictions.filter((prediction: any) => 
                    commonTradingDays.includes(prediction.date)
                  );
                  
                  // 按日期排序
                  alignedActual.sort((a, b) => a.date.localeCompare(b.date));
                  alignedPredictions.sort((a, b) => a.date.localeCompare(b.date));
                  
                  console.log(`✅ 找到 ${commonTradingDays.length} 个共同交易日进行对比`);
                } else {
                  // 如果没有共同的交易日，显示所有可用的数据
                  console.log('⚠️ 没有找到共同的交易日，显示所有可用数据');
                  alignedActual = processedActual;
                  alignedPredictions = processedPredictions.filter((prediction: any) => 
                    actualDates.includes(prediction.date)
                  );
                }
                
                // 更新预测数据为交易日对齐的版本
                const alignedPredictionData = {
                  ...processedPredictionData,
                  predictions: alignedPredictions
                };
                setPredictionData(alignedPredictionData);
                setActualData(alignedActual);
                
                console.log('✅ 交易日对齐完成:');
                console.log('  用户选择预测天数:', targetDays);
                console.log('  原始预测天数:', processedPredictions.length);
                console.log('  获取实际天数:', processedActual.length);
                console.log('  可对比预测天数:', alignedPredictions.length);
                console.log('  可对比实际天数:', alignedActual.length);
                console.log('📅 可对比的交易日:', alignedActual.map(a => a.date));
              } else {
                console.log('⚠️ 没有找到实际数据');
                setActualData([]);
              }
            } catch (error) {
              console.warn('❌ 获取实际数据失败:', error);
              setActualData([]);
            }
          } else {
            console.log('🔮 预测日期在未来，无实际数据');
            setActualData([]);
          }
        }

        message.success('预测完成！');
      } else {
        message.error('预测失败');
      }
    } catch (error: any) {
      console.error('预测失败:', error);
      message.error(error.response?.data?.message || '预测失败，请检查网络连接');
    } finally {
      setLoading(false);
      message.destroy();
    }
  };



  return (
    <Layout style={{ minHeight: '100vh', backgroundColor: '#f0f2f5' }}>
      <Header style={{ 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        padding: '0 50px',
        display: 'flex',
        alignItems: 'center'
      }}>
        <Title level={3} style={{ color: 'white', margin: 0 }}>
          🚀 Kronos A股预测分析系统
        </Title>
      </Header>

      <Content style={{ padding: '24px' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
          {/* 模型状态和预测表单在同一行 */}
          <Row gutter={24} style={{ marginBottom: '6px' }}>
            <Col xs={24} lg={8}>
              <ModelStatus />
            </Col>
            <Col xs={24} lg={16}>
              <PredictionForm onPredict={handlePredict} loading={loading} />
            </Col>
          </Row>

          {/* 加载状态 */}
          {loading && (
            <div style={{ 
              textAlign: 'center', 
              padding: '50px',
              backgroundColor: 'white',
              borderRadius: '8px',
              marginBottom: '24px'
            }}>
              <Spin size="large" />
              <div style={{ marginTop: '16px', color: '#666' }}>
                正在获取数据并进行预测...
              </div>
            </div>
          )}

          {/* 图表展示 */}
          {predictionData && !loading && (
            <PlotlyKLineChart
              title={`${predictionData.name} (${predictionData.code}) - 股价走势预测`}
              historicalData={historicalData}
              predictionData={predictionData.predictions}
              actualData={actualData}
            />
          )}

          {/* 预测结果表格 */}
          {predictionData && !loading && (
            <PredictionTable
              stockName={predictionData.name}
              stockCode={predictionData.code}
              predictions={predictionData.predictions}
              modelInfo={predictionData.model_info}
            />
          )}

          {/* 空状态 */}
          {!predictionData && !loading && (
            <div style={{
              textAlign: 'center',
              padding: '100px 20px',
              backgroundColor: 'white',
              borderRadius: '8px',
              color: '#666'
            }}>
              <div style={{ fontSize: '48px', marginBottom: '16px' }}>📈</div>
              <Title level={4} type="secondary">
                欢迎使用Kronos A股预测分析系统
              </Title>
              <p style={{ fontSize: '16px', marginBottom: '24px' }}>
                输入股票代码开始预测，系统将基于Kronos金融大模型为您提供智能预测
              </p>
              <div style={{ color: '#999' }}>
                <p>💡 支持的股票代码示例：</p>
                <p>• 600000 (浦发银行)</p>
                <p>• 000001 (平安银行)</p>
                <p>• 600036 (招商银行)</p>
              </div>
            </div>
          )}
        </div>
      </Content>
    </Layout>
  );
};

export default App;