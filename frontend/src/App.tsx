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


  const handlePredict = async (request: PredictionRequest) => {
    try {
      setLoading(true);
      message.loading('正在预测中，请稍候...', 0);

      // 获取历史数据
      const historyResponse = await StockAPI.getStockHistory(request.code, 90);
      if (historyResponse.success) {
        console.log('Historical data received:', historyResponse.data.history);
        // 处理日期格式
        const processedHistory = historyResponse.data.history.map((item: any) => ({
          ...item,
          date: item.date.split('T')[0] // 只保留日期部分
        }));
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