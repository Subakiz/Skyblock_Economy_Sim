import { Row, Col, Card, Statistic, Typography, Spin, Alert } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api/services';

const { Title } = Typography;

export function Dashboard() {
  const {
    data: healthData,
    isLoading: healthLoading,
    error: healthError,
  } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiService.healthCheck(),
  });

  const {
    data: scenariosData,
    isLoading: scenariosLoading,
  } = useQuery({
    queryKey: ['scenarios'],
    queryFn: () => apiService.getAvailableScenarios(),
  });

  const {
    data: modelStatusData,
    isLoading: modelStatusLoading,
  } = useQuery({
    queryKey: ['model-status'],
    queryFn: () => apiService.getModelStatus(),
  });

  console.log('Model status:', modelStatusData, 'Loading:', modelStatusLoading);

  if (healthError) {
    return (
      <Alert
        message="Connection Error"
        description="Unable to connect to the SkyBlock Economy API. Please ensure the backend server is running on port 8000."
        type="error"
        showIcon
        style={{ marginBottom: 16 }}
      />
    );
  }

  return (
    <div>
      <Title level={2}>Dashboard Overview</Title>
      <p style={{ fontSize: '16px', color: '#666' }}>
        Welcome to the SkyBlock Economic Analysis Platform. Monitor market metrics, analyze trends, and manage your simulations.
      </p>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="API Status"
              value={healthLoading ? "Checking..." : healthData ? "Online" : "Offline"}
              prefix={
                healthLoading ? (
                  <Spin size="small" />
                ) : healthData ? (
                  <ArrowUpOutlined style={{ color: '#3f8600' }} />
                ) : (
                  <ArrowDownOutlined style={{ color: '#cf1322' }} />
                )
              }
              valueStyle={{
                color: healthLoading ? '#666' : healthData ? '#3f8600' : '#cf1322',
              }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Available Scenarios"
              value={scenariosLoading ? 0 : (scenariosData?.data?.length || 0)}
              suffix="scenarios"
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="ML Models"
              value={modelStatusLoading ? "Loading..." : "Ready"}
              prefix={<ArrowUpOutlined style={{ color: '#3f8600' }} />}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Market Volatility"
              value={2.4}
              precision={1}
              suffix="%"
              prefix={<ArrowUpOutlined style={{ color: '#cf1322' }} />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="Quick Actions" style={{ height: '300px' }}>
            <div style={{ padding: '20px 0' }}>
              <p>• <strong>Market Analysis:</strong> View current and historical price data with interactive charts</p>
              <p>• <strong>Craft Profitability:</strong> Analyze crafting opportunities and profit margins</p>
              <p>• <strong>Backtesting:</strong> Test trading strategies against historical data</p>
              <p>• <strong>Market Simulation:</strong> Run agent-based market scenarios and predictions</p>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="Recent Activity" style={{ height: '300px' }}>
            <div style={{ padding: '20px 0' }}>
              <p>• Connected to SkyBlock Economy API</p>
              <p>• {(scenariosData?.data?.length || 0)} simulation scenarios available</p>
              <p>• ML forecasting models ready</p>
              <p>• Real-time market data streaming</p>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
}