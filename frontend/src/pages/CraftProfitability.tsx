import { useState } from 'react';
import {
  Card,
  Input,
  Button,
  Typography,
  Row,
  Col,
  Statistic,
  Alert,
  Table,
  Tag,
  Spin,
} from 'antd';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api/services';

const { Title } = Typography;
const { Search } = Input;

export function CraftProfitability() {
  const [selectedItem, setSelectedItem] = useState<string>('ENCHANTED_LAPIS_BLOCK');

  const {
    data: craftData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['craft-profit', selectedItem],
    queryFn: () => apiService.getCraftProfitability(selectedItem, '1h', 'median'),
    enabled: !!selectedItem,
  });

  const handleSearch = (value: string) => {
    setSelectedItem(value);
  };

  // Mock crafting tree data
  const craftingTree = [
    {
      key: '1',
      material: 'Lapis Lazuli',
      quantity: 160,
      unitPrice: 2,
      totalCost: 320,
      source: 'Bazaar',
    },
    {
      key: '2',
      material: 'Enchanted Lapis Lazuli',
      quantity: 1,
      unitPrice: 320,
      totalCost: 320,
      source: 'Craft',
    },
    {
      key: '3',
      material: 'Enchanted Lapis Block',
      quantity: 1,
      unitPrice: 150000,
      totalCost: 150000,
      source: 'Final Product',
    },
  ];

  const columns = [
    {
      title: 'Material',
      dataIndex: 'material',
      key: 'material',
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
    },
    {
      title: 'Unit Price',
      dataIndex: 'unitPrice',
      key: 'unitPrice',
      render: (price: number) => `${price.toLocaleString()} coins`,
    },
    {
      title: 'Total Cost',
      dataIndex: 'totalCost',
      key: 'totalCost',
      render: (cost: number) => `${cost.toLocaleString()} coins`,
    },
    {
      title: 'Source',
      dataIndex: 'source',
      key: 'source',
      render: (source: string) => {
        const color = source === 'Bazaar' ? 'green' : source === 'Craft' ? 'blue' : 'gold';
        return <Tag color={color}>{source}</Tag>;
      },
    },
  ];

  return (
    <div>
      <Title level={2}>Craft Profitability Analysis</Title>
      <p style={{ fontSize: '16px', color: '#666', marginBottom: 24 }}>
        Analyze crafting opportunities, material costs, and profit margins for SkyBlock items.
      </p>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} md={16}>
          <Card>
            <Search
              placeholder="Enter item name (e.g., ENCHANTED_LAPIS_BLOCK)"
              value={selectedItem}
              onChange={(e) => setSelectedItem(e.target.value)}
              onSearch={handleSearch}
              enterButton="Analyze"
              size="large"
              loading={isLoading}
            />
          </Card>
        </Col>
        
        <Col xs={24} md={8}>
          <Card>
            <Button
              type="primary"
              onClick={() => refetch()}
              loading={isLoading}
              block
              size="large"
            >
              Refresh Analysis
            </Button>
          </Card>
        </Col>
      </Row>

      {error && (
        <Alert
          message="Error"
          description="Could not fetch craft profitability data. The item might not exist or the API might be unavailable."
          type="error"
          style={{ marginBottom: 16 }}
          showIcon
        />
      )}

      {isLoading && (
        <Card>
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <Spin size="large" />
            <p style={{ marginTop: 16 }}>Analyzing crafting profitability...</p>
          </div>
        </Card>
      )}

      {craftData?.data && (
        <>
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={24} sm={12} lg={6}>
              <Card>
                <Statistic
                  title="Craft Cost"
                  value={craftData.data.craft_cost}
                  precision={0}
                  suffix="coins"
                  valueStyle={{ color: '#cf1322' }}
                />
              </Card>
            </Col>

            <Col xs={24} sm={12} lg={6}>
              <Card>
                <Statistic
                  title="Sale Price"
                  value={craftData.data.expected_sale_price}
                  precision={0}
                  suffix="coins"
                  valueStyle={{ color: '#3f8600' }}
                />
              </Card>
            </Col>

            <Col xs={24} sm={12} lg={6}>
              <Card>
                <Statistic
                  title="Net Profit"
                  value={craftData.data.net_margin}
                  precision={0}
                  suffix="coins"
                  valueStyle={{ 
                    color: craftData.data.net_margin > 0 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Card>
            </Col>

            <Col xs={24} sm={12} lg={6}>
              <Card>
                <Statistic
                  title="ROI"
                  value={craftData.data.roi_percent}
                  precision={1}
                  suffix="%"
                  valueStyle={{ 
                    color: craftData.data.roi_percent > 0 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card title="Crafting Tree & Material Costs">
                <Table
                  columns={columns}
                  dataSource={craftingTree}
                  pagination={false}
                  size="middle"
                />
              </Card>
            </Col>

            <Col xs={24} lg={8}>
              <Card title="Profitability Summary">
                <div style={{ marginBottom: 16 }}>
                  <strong>Best Path:</strong>
                  <p>{craftData.data.best_path}</p>
                </div>
                
                <div style={{ marginBottom: 16 }}>
                  <strong>Sales Volume:</strong>
                  <p>{craftData.data.sell_volume} units</p>
                </div>
                
                <div style={{ marginBottom: 16 }}>
                  <strong>Turnover Adjusted Profit:</strong>
                  <p>{craftData.data.turnover_adj_profit.toLocaleString()} coins</p>
                </div>
                
                <div>
                  <strong>Data Age:</strong>
                  <p>{craftData.data.data_age_minutes} minutes old</p>
                </div>
                
                {craftData.data.roi_percent > 10 && (
                  <Alert
                    message="High Profitability"
                    description="This item shows strong profit potential!"
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
    </div>
  );
}