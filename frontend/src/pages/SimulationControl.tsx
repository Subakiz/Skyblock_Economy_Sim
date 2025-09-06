import { useState } from 'react';
import {
  Card,
  Form,
  Select,
  InputNumber,
  Button,
  Typography,
  Row,
  Col,
  Statistic,
  Alert,
  Table,
  Tag,
  Progress,
  Tabs,
  Spin,
} from 'antd';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useQuery, useMutation } from '@tanstack/react-query';
import { apiService } from '../api/services';
import type { MarketSimulationRequest } from '../api/types';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

const COLORS = ['#1890ff', '#52c41a', '#fa8c16', '#f5222d', '#722ed1'];

export function SimulationControl() {
  const [form] = Form.useForm();
  const [results, setResults] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('configure');

  // Get available scenarios
  const {
    data: scenariosData,
    isLoading: scenariosLoading,
  } = useQuery({
    queryKey: ['scenarios'],
    queryFn: () => apiService.getAvailableScenarios(),
  });

  // Market simulation mutation
  const simulationMutation = useMutation({
    mutationFn: (request: MarketSimulationRequest) => apiService.runMarketSimulation(request),
    onSuccess: (data) => {
      setResults(data.data);
      setActiveTab('results');
    },
    onError: (error) => {
      console.error('Simulation failed:', error);
    },
  });

  const handleSubmit = (values: any) => {
    const request: MarketSimulationRequest = {
      scenario: values.scenario,
      n_agents: values.n_agents,
      steps: values.steps,
      market_volatility: values.market_volatility / 100, // Convert percentage to decimal
    };

    simulationMutation.mutate(request);
  };

  // Mock agent performance data
  const mockAgentData = results ? [
    { name: 'Farmer', performance: 85, count: 20, avgProfit: 125000 },
    { name: 'Dungeon Runner', performance: 92, count: 15, avgProfit: 250000 },
    { name: 'Auction Flipper', performance: 78, count: 25, avgProfit: 95000 },
    { name: 'Crafter', performance: 88, count: 18, avgProfit: 180000 },
    { name: 'Merchant', performance: 95, count: 12, avgProfit: 320000 },
  ] : [];

  const mockScenarioComparison = [
    { scenario: 'Normal Market', avgReturn: 12.5, volatility: 8.2, successRate: 78 },
    { scenario: 'Volatile Market', avgReturn: 15.8, volatility: 24.6, successRate: 52 },
    { scenario: 'Stable Market', avgReturn: 8.1, volatility: 3.1, successRate: 89 },
    { scenario: 'Major Update', avgReturn: 22.3, volatility: 35.2, successRate: 45 },
  ];

  const agentColumns = [
    {
      title: 'Agent Type',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => <Tag color="blue">{name}</Tag>,
    },
    {
      title: 'Count',
      dataIndex: 'count',
      key: 'count',
    },
    {
      title: 'Performance',
      dataIndex: 'performance',
      key: 'performance',
      render: (perf: number) => (
        <div style={{ width: 100 }}>
          <Progress
            percent={perf}
            size="small"
            status={perf > 80 ? 'success' : perf > 60 ? 'normal' : 'exception'}
            showInfo={false}
          />
          <Text style={{ fontSize: '12px' }}>{perf}%</Text>
        </div>
      ),
    },
    {
      title: 'Avg Profit',
      dataIndex: 'avgProfit',
      key: 'avgProfit',
      render: (profit: number) => `${profit.toLocaleString()} coins`,
    },
  ];

  return (
    <div>
      <Title level={2}>Market Simulation Control Panel</Title>
      <p style={{ fontSize: '16px', color: '#666', marginBottom: 24 }}>
        Configure and run agent-based market simulations to predict market behavior and test scenarios.
      </p>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="Configure Simulation" key="configure">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="Simulation Parameters">
                <Form
                  form={form}
                  layout="vertical"
                  onFinish={handleSubmit}
                  initialValues={{
                    scenario: 'normal_market',
                    n_agents: 100,
                    steps: 500,
                    market_volatility: 2.0,
                  }}
                >
                  <Form.Item
                    name="scenario"
                    label="Market Scenario"
                    rules={[{ required: true, message: 'Please select a scenario' }]}
                  >
                    <Select
                      size="large"
                      loading={scenariosLoading}
                      placeholder="Select scenario"
                    >
                      {(scenariosData?.data || []).map?.((scenario: string) => (
                        <Option key={scenario} value={scenario}>
                          {scenario.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </Option>
                      )) || [
                        <Option key="normal_market" value="normal_market">Normal Market</Option>,
                        <Option key="volatile_market" value="volatile_market">Volatile Market</Option>,
                        <Option key="stable_market" value="stable_market">Stable Market</Option>,
                        <Option key="major_update" value="major_update">Major Update</Option>,
                      ]}
                    </Select>
                  </Form.Item>

                  <Row gutter={16}>
                    <Col xs={24} sm={12}>
                      <Form.Item
                        name="n_agents"
                        label="Number of Agents"
                        rules={[{ required: true, message: 'Please enter number of agents' }]}
                      >
                        <InputNumber
                          min={10}
                          max={500}
                          style={{ width: '100%' }}
                          size="large"
                        />
                      </Form.Item>
                    </Col>

                    <Col xs={24} sm={12}>
                      <Form.Item
                        name="steps"
                        label="Simulation Steps"
                        rules={[{ required: true, message: 'Please enter number of steps' }]}
                      >
                        <InputNumber
                          min={100}
                          max={5000}
                          style={{ width: '100%' }}
                          size="large"
                        />
                      </Form.Item>
                    </Col>
                  </Row>

                  <Form.Item
                    name="market_volatility"
                    label="Market Volatility (%)"
                    rules={[{ required: true, message: 'Please enter market volatility' }]}
                  >
                    <InputNumber
                      min={0.1}
                      max={50}
                      step={0.1}
                      style={{ width: '100%' }}
                      size="large"
                      formatter={(value) => `${value}%`}
                      parser={(value) => {
                        const num = parseFloat(value?.replace('%', '') || '0');
                        return Math.min(50, Math.max(0.1, num)) as 50 | 0.1;
                      }}
                    />
                  </Form.Item>

                  <Form.Item>
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={simulationMutation.isPending}
                      size="large"
                      block
                    >
                      {simulationMutation.isPending ? 'Running Simulation...' : 'Start Simulation'}
                    </Button>
                  </Form.Item>
                </Form>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="Scenario Information">
                <div style={{ marginBottom: 16 }}>
                  <Text strong>Available Scenarios:</Text>
                </div>
                
                <div style={{ marginBottom: 12 }}>
                  <Tag color="blue">Normal Market</Tag>
                  <span>Standard market conditions with regular trading activity</span>
                </div>
                
                <div style={{ marginBottom: 12 }}>
                  <Tag color="orange">Volatile Market</Tag>
                  <span>High volatility environment with rapid price changes</span>
                </div>
                
                <div style={{ marginBottom: 12 }}>
                  <Tag color="green">Stable Market</Tag>
                  <span>Low volatility, efficient market with predictable trends</span>
                </div>
                
                <div style={{ marginBottom: 12 }}>
                  <Tag color="red">Major Update</Tag>
                  <span>Simulates impact of large game updates on market</span>
                </div>

                {simulationMutation.isPending && (
                  <div style={{ textAlign: 'center', marginTop: 24, padding: '20px' }}>
                    <Spin size="large" />
                    <p style={{ marginTop: 16 }}>Simulation in progress...</p>
                    <Progress percent={45} status="active" />
                  </div>
                )}

                {simulationMutation.error && (
                  <Alert
                    message="Simulation Failed"
                    description="Unable to run market simulation. Please check your parameters."
                    type="error"
                    style={{ marginTop: 16 }}
                    showIcon
                  />
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="Results & Analysis" key="results" disabled={!results}>
          {results && (
            <>
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24} sm={12} lg={6}>
                  <Card>
                    <Statistic
                      title="Total Trades"
                      value={results.total_trades || 1250}
                      formatter={(value) => value?.toLocaleString()}
                    />
                  </Card>
                </Col>

                <Col xs={24} sm={12} lg={6}>
                  <Card>
                    <Statistic
                      title="Market Sentiment"
                      value={(results.market_sentiment || 0.65) * 100}
                      precision={1}
                      suffix="%"
                      valueStyle={{
                        color: results.market_sentiment > 0.5 ? '#3f8600' : '#cf1322',
                      }}
                    />
                  </Card>
                </Col>

                <Col xs={24} sm={12} lg={6}>
                  <Card>
                    <Statistic
                      title="Price Volatility"
                      value={8.2}
                      precision={1}
                      suffix="%"
                    />
                  </Card>
                </Col>

                <Col xs={24} sm={12} lg={6}>
                  <Card>
                    <Statistic
                      title="Scenario"
                      value={results.scenario?.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                    />
                  </Card>
                </Col>
              </Row>

              <Row gutter={[16, 16]}>
                <Col xs={24} lg={12}>
                  <Card title="Agent Performance Distribution">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={mockAgentData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                        <YAxis />
                        <Tooltip formatter={(value: number) => [`${value}%`, 'Performance']} />
                        <Legend />
                        <Bar dataKey="performance" fill="#1890ff" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>

                <Col xs={24} lg={12}>
                  <Card title="Agent Type Distribution">
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={mockAgentData}
                          dataKey="count"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          label={({ name, count }) => `${name}: ${count}`}
                        >
                          {mockAgentData.map((_, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
              </Row>

              <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
                <Col xs={24} lg={16}>
                  <Card title="Agent Performance Details">
                    <Table
                      columns={agentColumns}
                      dataSource={mockAgentData}
                      pagination={false}
                      size="middle"
                    />
                  </Card>
                </Col>

                <Col xs={24} lg={8}>
                  <Card title="Simulation Summary">
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>Most Successful Agent Type:</Text>
                      <p>Merchant (95% performance)</p>
                    </div>
                    
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>Market Efficiency:</Text>
                      <p>High - Prices converged quickly</p>
                    </div>
                    
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>Risk Level:</Text>
                      <p>Moderate volatility observed</p>
                    </div>
                    
                    <div>
                      <Text strong>Recommendation:</Text>
                      <p>Merchant strategies show highest potential returns</p>
                    </div>

                    {results.market_sentiment > 0.7 && (
                      <Alert
                        message="Bullish Market"
                        description="High market sentiment indicates good trading conditions"
                        type="success"
                        style={{ marginTop: 16 }}
                        showIcon
                      />
                    )}
                  </Card>
                </Col>
              </Row>
            </>
          )}
        </TabPane>

        <TabPane tab="Scenario Comparison" key="comparison">
          <Card title="Multi-Scenario Analysis">
            <Table
              dataSource={mockScenarioComparison}
              pagination={false}
              columns={[
                {
                  title: 'Scenario',
                  dataIndex: 'scenario',
                  key: 'scenario',
                  render: (scenario: string) => <Tag color="blue">{scenario}</Tag>,
                },
                {
                  title: 'Avg Return (%)',
                  dataIndex: 'avgReturn',
                  key: 'avgReturn',
                  render: (value: number) => (
                    <span style={{ color: value > 10 ? '#3f8600' : '#666' }}>
                      {value}%
                    </span>
                  ),
                },
                {
                  title: 'Volatility (%)',
                  dataIndex: 'volatility',
                  key: 'volatility',
                  render: (value: number) => (
                    <span style={{ color: value > 20 ? '#cf1322' : '#666' }}>
                      {value}%
                    </span>
                  ),
                },
                {
                  title: 'Success Rate (%)',
                  dataIndex: 'successRate',
                  key: 'successRate',
                  render: (value: number) => (
                    <Progress
                      percent={value}
                      size="small"
                      status={value > 70 ? 'success' : value > 50 ? 'normal' : 'exception'}
                    />
                  ),
                },
              ]}
            />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
}