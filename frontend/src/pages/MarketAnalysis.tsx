import { useState } from 'react';
import { 
  Card, 
  Input, 
  Select, 
  Typography, 
  Row, 
  Col, 
  Table, 
  Tag,
  Spin,
  Alert
} from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api/services';

const { Title } = Typography;
const { Search } = Input;
const { Option } = Select;

// Sample popular items
const popularItems = [
  'ENCHANTED_LAPIS_BLOCK',
  'HYPERION',
  'NECRON_CHESTPLATE',
  'WITHER_SKULL',
  'ENCHANTED_CARROT',
  'ENCHANTED_POTATO',
];

export function MarketAnalysis() {
  const [selectedItem, setSelectedItem] = useState<string>('ENCHANTED_LAPIS_BLOCK');
  const [timeWindow, setTimeWindow] = useState<string>('1h');

  const {
    data: forecastData,
    isLoading: forecastLoading,
    error: forecastError,
  } = useQuery({
    queryKey: ['forecast', selectedItem],
    queryFn: () => apiService.getForecast(selectedItem, 60),
    enabled: !!selectedItem,
  });

  console.log('Forecast data:', forecastData, 'Loading:', forecastLoading, 'Error:', forecastError);

  const {
    data: ahPriceData,
    isLoading: ahLoading,
    error: ahError,
  } = useQuery({
    queryKey: ['ah-prices', selectedItem, timeWindow],
    queryFn: () => apiService.getAHPrices(selectedItem, timeWindow),
    enabled: !!selectedItem,
  });

  // Mock price history data for chart
  const mockPriceHistory = [
    { time: '00:00', price: 145000, volume: 24 },
    { time: '04:00', price: 148000, volume: 18 },
    { time: '08:00', price: 152000, volume: 31 },
    { time: '12:00', price: 149000, volume: 27 },
    { time: '16:00', price: 154000, volume: 22 },
    { time: '20:00', price: 151000, volume: 29 },
  ];

  const columns = [
    {
      title: 'Time',
      dataIndex: 'time',
      key: 'time',
    },
    {
      title: 'Price',
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => `${price.toLocaleString()} coins`,
    },
    {
      title: 'Volume',
      dataIndex: 'volume',
      key: 'volume',
    },
    {
      title: 'Status',
      key: 'status',
      render: () => <Tag color="green">Active</Tag>,
    },
  ];

  return (
    <div>
      <Title level={2}>Market Analysis</Title>
      <p style={{ fontSize: '16px', color: '#666', marginBottom: 24 }}>
        Analyze current and historical price data for SkyBlock items with interactive charts and detailed metrics.
      </p>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} md={12} lg={8}>
          <Card>
            <Search
              placeholder="Search for items..."
              value={selectedItem}
              onChange={(e) => setSelectedItem(e.target.value)}
              onSearch={setSelectedItem}
              enterButton="Search"
              size="large"
            />
            <div style={{ marginTop: 16 }}>
              <strong>Popular Items:</strong>
              <div style={{ marginTop: 8 }}>
                {popularItems.map((item) => (
                  <Tag
                    key={item}
                    style={{ cursor: 'pointer', marginBottom: 4 }}
                    color={selectedItem === item ? 'blue' : 'default'}
                    onClick={() => setSelectedItem(item)}
                  >
                    {item}
                  </Tag>
                ))}
              </div>
            </div>
          </Card>
        </Col>

        <Col xs={24} md={12} lg={8}>
          <Card>
            <div style={{ marginBottom: 16 }}>
              <strong>Time Window:</strong>
            </div>
            <Select
              value={timeWindow}
              onChange={setTimeWindow}
              style={{ width: '100%' }}
              size="large"
            >
              <Option value="15m">15 minutes</Option>
              <Option value="1h">1 hour</Option>
              <Option value="4h">4 hours</Option>
              <Option value="1d">1 day</Option>
            </Select>
          </Card>
        </Col>

        <Col xs={24} md={12} lg={8}>
          <Card>
            {ahLoading ? (
              <Spin />
            ) : ahError ? (
              <Alert message="Error loading price data" type="error" />
            ) : ahPriceData?.data ? (
              <div>
                <div><strong>Median Price:</strong> {ahPriceData.data.median_price.toLocaleString()} coins</div>
                <div><strong>Sales:</strong> {ahPriceData.data.sale_count}</div>
                <div><strong>Last Updated:</strong> {new Date(ahPriceData.data.last_updated).toLocaleTimeString()}</div>
              </div>
            ) : (
              <div>No price data available</div>
            )}
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card title={`Price Chart - ${selectedItem}`} style={{ height: 400 }}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={mockPriceHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip formatter={(value: number) => [`${value.toLocaleString()} coins`, 'Price']} />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#1890ff" 
                  strokeWidth={2}
                  dot={{ fill: '#1890ff' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="Recent Transactions" style={{ height: 400 }}>
            <Table
              columns={columns}
              dataSource={mockPriceHistory}
              size="small"
              pagination={false}
              scroll={{ y: 280 }}
            />
          </Card>
        </Col>
      </Row>

      {forecastData?.data && (
        <Row style={{ marginTop: 16 }}>
          <Col xs={24}>
            <Card title="Price Forecast">
              <Alert
                message={`Predicted price in 60 minutes: ${forecastData.data.forecast_price.toLocaleString()} coins`}
                type="info"
                showIcon
              />
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
}