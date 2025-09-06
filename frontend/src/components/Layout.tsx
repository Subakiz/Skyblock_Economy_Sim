import type { ReactNode } from 'react';
import { useState } from 'react';
import { Layout as AntLayout, Menu, Typography, theme, Drawer, Button } from 'antd';
import { Link, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  LineChartOutlined,
  ToolOutlined,
  BackwardOutlined,
  ExperimentOutlined,
  MenuOutlined,
} from '@ant-design/icons';

const { Header, Content, Sider } = AntLayout;
const { Title } = Typography;

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileMenuVisible, setMobileMenuVisible] = useState(false);
  const location = useLocation();
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined />,
      label: <Link to="/dashboard">Dashboard</Link>,
    },
    {
      key: '/market-analysis',
      icon: <LineChartOutlined />,
      label: <Link to="/market-analysis">Market Analysis</Link>,
    },
    {
      key: '/craft-profitability',
      icon: <ToolOutlined />,
      label: <Link to="/craft-profitability">Craft Profitability</Link>,
    },
    {
      key: '/backtesting',
      icon: <BackwardOutlined />,
      label: <Link to="/backtesting">Backtesting</Link>,
    },
    {
      key: '/simulation',
      icon: <ExperimentOutlined />,
      label: <Link to="/simulation">Market Simulation</Link>,
    },
  ];

  const selectedKey = location.pathname === '/' ? '/dashboard' : location.pathname;

  return (
    <AntLayout style={{ minHeight: '100vh' }}>
      {/* Desktop Sidebar */}
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        width={250}
        className="hidden md:block"
        style={{
          background: colorBgContainer,
          borderRight: '1px solid #f0f0f0',
        }}
      >
        <div style={{ padding: '16px', textAlign: 'center' }}>
          <Title level={4} style={{ margin: 0, color: '#1890ff' }}>
            {collapsed ? 'SE' : 'SkyBlock Economy'}
          </Title>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
          style={{ border: 'none' }}
        />
      </Sider>

      {/* Mobile Drawer */}
      <Drawer
        title="SkyBlock Economy"
        placement="left"
        onClose={() => setMobileMenuVisible(false)}
        open={mobileMenuVisible}
        bodyStyle={{ padding: 0 }}
        className="md:hidden"
      >
        <Menu
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
          onClick={() => setMobileMenuVisible(false)}
        />
      </Drawer>

      <AntLayout>
        <Header
          style={{
            padding: '0 16px',
            background: colorBgContainer,
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div className="flex items-center">
            <Button
              type="text"
              icon={<MenuOutlined />}
              onClick={() => setMobileMenuVisible(true)}
              className="md:hidden mr-2"
            />
            <Title level={3} style={{ margin: 0 }} className="hidden md:block">
              Hypixel SkyBlock Economic Analysis Platform
            </Title>
            <Title level={4} style={{ margin: 0 }} className="md:hidden">
              SkyBlock Economy
            </Title>
          </div>
        </Header>
        <Content
          style={{
            margin: '16px',
            padding: '24px',
            minHeight: 280,
            background: colorBgContainer,
            borderRadius: borderRadiusLG,
          }}
        >
          {children}
        </Content>
      </AntLayout>
    </AntLayout>
  );
}