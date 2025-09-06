import { useState } from 'react';
import {
  Card,
  Form,
  Input,
  Select,
  DatePicker,
  InputNumber,
  Button,
  Typography,
  Row,
  Col,
  Statistic,
  Alert,
  Table,
  Progress,
  Spin,
} from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useMutation } from '@tanstack/react-query';
import { apiService } from '../api/services';
import type { BacktestRequest } from '../api/types';
import dayjs from 'dayjs';

const { Title } = Typography;
const { Option } = Select;
const { RangePicker } = DatePicker;

export function Backtesting() {
  const [form] = Form.useForm();
  const [results, setResults] = useState<any>(null);

  const backtestMutation = useMutation({
    mutationFn: (request: BacktestRequest) => apiService.runBacktest(request),
    onSuccess: (data) => {
      setResults(data.data);
    },
    onError: (error) => {
      console.error('Backtest failed:', error);
    },
  });

  const handleSubmit = (values: any) => {
    const request: BacktestRequest = {
      strategy: values.strategy,
      params: {
        min_profit: values.min_profit,
        max_position: values.max_position,
      },
      start_date: values.date_range[0].format('YYYY-MM-DD'),
      end_date: values.date_range[1].format('YYYY-MM-DD'),
      capital: values.capital,
      item_id: values.item_id,
    };

    backtestMutation.mutate(request);
  };

  // Mock performance data for chart
  const mockPerformanceData = results ? [
    { date: '2025-08-01', value: results.capital },
    { date: '2025-08-05', value: results.capital + results.final_value * 0.2 },
    { date: '2025-08-10', value: results.capital + results.final_value * 0.4 },
    { date: '2025-08-15', value: results.capital + results.final_value * 0.6 },
    { date: '2025-08-20', value: results.capital + results.final_value * 0.8 },
    { date: '2025-08-25', value: results.final_value },
  ] : [];

  const tradeColumns = [
    {
      title: 'Date',
      dataIndex: 'date',
      key: 'date',
    },
    {
      title: 'Action',
      dataIndex: 'action',
      key: 'action',
      render: (action: string) => (
        <span style={{ color: action === 'BUY' ? '#cf1322' : '#3f8600' }}>
          {action}
        </span>
      ),
    },
    {
      title: 'Price',
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => `${price.toLocaleString()} coins`,
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
    },
    {
      title: 'P&L',
      dataIndex: 'pnl',
      key: 'pnl',
      render: (pnl: number) => (
        <span style={{ color: pnl > 0 ? '#3f8600' : '#cf1322' }}>
          {pnl > 0 ? '+' : ''}{pnl.toLocaleString()} coins
        </span>
      ),
    },
  ];

  const mockTrades = [
    { key: '1', date: '2025-08-02', action: 'BUY', price: 145000, quantity: 1, pnl: 0 },
    { key: '2', date: '2025-08-02', action: 'SELL', price: 152000, quantity: 1, pnl: 7000 },
    { key: '3', date: '2025-08-05', action: 'BUY', price: 148000, quantity: 1, pnl: 0 },
    { key: '4', date: '2025-08-05', action: 'SELL', price: 155000, quantity: 1, pnl: 7000 },
  ];

  return (
    <div>
      <Title level={2}>Strategy Backtesting</Title>
      <p style={{ fontSize: '16px', color: '#666', marginBottom: 24 }}>
        Test trading strategies against historical data to evaluate performance and risk metrics.
      </p>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="Backtest Configuration">
            <Form
              form={form}
              layout="vertical"
              onFinish={handleSubmit}
              initialValues={{
                strategy: 'flip_bin',
                min_profit: 50000,
                max_position: 10,
                capital: 10000000,
                item_id: 'ENCHANTED_LAPIS_BLOCK',
                date_range: [dayjs().subtract(30, 'days'), dayjs()],
              }}
            >
              <Form.Item
                name="strategy"
                label="Strategy"
                rules={[{ required: true, message: 'Please select a strategy' }]}
              >
                <Select size="large">
                  <Option value="flip_bin">Flip BIN</Option>
                  <Option value="craft_and_sell">Craft and Sell</Option>
                  <Option value="simple_arbitrage">Simple Arbitrage</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="item_id"
                label="Item ID"
                rules={[{ required: true, message: 'Please enter an item ID' }]}
              >
                <Input
                  placeholder="e.g., ENCHANTED_LAPIS_BLOCK"
                  size="large"
                />
              </Form.Item>

              <Row gutter={16}>
                <Col xs={24} sm={12}>
                  <Form.Item
                    name="min_profit"
                    label="Min Profit (coins)"
                    rules={[{ required: true, message: 'Please enter minimum profit' }]}
                  >
                    <InputNumber
                      placeholder="50000"
                      style={{ width: '100%' }}
                      size="large"
                      formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                    />
                  </Form.Item>
                </Col>

                <Col xs={24} sm={12}>
                  <Form.Item
                    name="max_position"
                    label="Max Position"
                    rules={[{ required: true, message: 'Please enter max position' }]}
                  >
                    <InputNumber
                      placeholder="10"
                      style={{ width: '100%' }}
                      size="large"
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                name="capital"
                label="Starting Capital (coins)"
                rules={[{ required: true, message: 'Please enter starting capital' }]}
              >
                <InputNumber
                  placeholder="10000000"
                  style={{ width: '100%' }}
                  size="large"
                  formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                />
              </Form.Item>

              <Form.Item
                name="date_range"
                label="Date Range"
                rules={[{ required: true, message: 'Please select a date range' }]}
              >
                <RangePicker
                  style={{ width: '100%' }}
                  size="large"
                />
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={backtestMutation.isPending}
                  size="large"
                  block
                >
                  {backtestMutation.isPending ? 'Running Backtest...' : 'Run Backtest'}
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          {backtestMutation.isPending && (
            <Card title="Backtest Progress">
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Spin size="large" />
                <p style={{ marginTop: 16 }}>Running backtest simulation...</p>
                <Progress percent={75} status="active" />
              </div>
            </Card>
          )}

          {backtestMutation.error && (
            <Card title="Error">
              <Alert
                message="Backtest Failed"
                description="Unable to run backtest. Please check your parameters and try again."
                type="error"
                showIcon
              />
            </Card>
          )}

          {results && (
            <Card title="Performance Summary">
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12}>
                  <Statistic
                    title="Total Return"
                    value={results.total_return_pct}
                    precision={1}
                    suffix="%"
                    valueStyle={{
                      color: results.total_return_pct > 0 ? '#3f8600' : '#cf1322',
                    }}
                  />
                </Col>

                <Col xs={24} sm={12}>
                  <Statistic
                    title="Sharpe Ratio"
                    value={results.sharpe_ratio}
                    precision={2}
                  />
                </Col>

                <Col xs={24} sm={12}>
                  <Statistic
                    title="Max Drawdown"
                    value={results.max_drawdown_pct}
                    precision={1}
                    suffix="%"
                    valueStyle={{ color: '#cf1322' }}
                  />
                </Col>

                <Col xs={24} sm={12}>
                  <Statistic
                    title="Win Rate"
                    value={results.win_rate}
                    precision={1}
                    suffix="%"
                  />
                </Col>

                <Col xs={24} sm={12}>
                  <Statistic
                    title="Total Trades"
                    value={results.total_trades}
                  />
                </Col>

                <Col xs={24} sm={12}>
                  <Statistic
                    title="Final Value"
                    value={results.final_value}
                    formatter={(value) => `${Number(value).toLocaleString()} coins`}
                  />
                </Col>
              </Row>
            </Card>
          )}
        </Col>
      </Row>

      {results && (
        <>
          <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
            <Col xs={24}>
              <Card title="Performance Chart">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={mockPerformanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis tickFormatter={(value: number) => `${(value / 1000000).toFixed(1)}M`} />
                    <Tooltip formatter={(value: number) => [`${value.toLocaleString()} coins`, 'Portfolio Value']} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="#1890ff"
                      strokeWidth={2}
                      dot={{ fill: '#1890ff' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          <Row style={{ marginTop: 16 }}>
            <Col xs={24}>
              <Card title="Trade History">
                <Table
                  columns={tradeColumns}
                  dataSource={mockTrades}
                  pagination={{ pageSize: 10 }}
                  scroll={{ x: 600 }}
                />
              </Card>
            </Col>
          </Row>
        </>
      )}
    </div>
  );
}