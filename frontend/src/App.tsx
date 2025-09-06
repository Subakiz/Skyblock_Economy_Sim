import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider, App as AntApp } from 'antd';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { MarketAnalysis } from './pages/MarketAnalysis';
import { CraftProfitability } from './pages/CraftProfitability';
import { Backtesting } from './pages/Backtesting';
import { SimulationControl } from './pages/SimulationControl';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 6,
        },
      }}
    >
      <AntApp>
        <QueryClientProvider client={queryClient}>
          <Router>
            <Layout>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/market-analysis" element={<MarketAnalysis />} />
                <Route path="/craft-profitability" element={<CraftProfitability />} />
                <Route path="/backtesting" element={<Backtesting />} />
                <Route path="/simulation" element={<SimulationControl />} />
              </Routes>
            </Layout>
          </Router>
        </QueryClientProvider>
      </AntApp>
    </ConfigProvider>
  );
}

export default App;
