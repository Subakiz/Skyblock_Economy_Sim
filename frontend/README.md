# SkyBlock Economy Frontend

A modern React-based web interface for the SkyBlock Economic Analysis Platform.

## Features

- **Dashboard Overview**: Key market metrics and system status at a glance
- **Market Analysis**: Interactive price charts and data tables for SkyBlock items
- **Craft Profitability**: Real-time crafting analysis with profit calculations
- **Backtesting Engine**: Test trading strategies against historical data
- **Market Simulation**: Configure and run agent-based market simulations

## Technology Stack

- **React 18** with TypeScript
- **Vite** for development and building
- **Ant Design** for UI components
- **TanStack Query** for server state management
- **Recharts** for data visualization
- **React Router** for navigation
- **Axios** for API communication

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running on port 8000

### Installation

```bash
cd frontend
npm install
```

### Development

Start the development server:

```bash
npm run dev
```

The application will be available at http://localhost:3000 with API proxy to http://localhost:8000

### Building

Build for production:

```bash
npm run build
```

The built files will be in the `dist` directory.

## API Integration

The frontend communicates with the FastAPI backend through a proxy configuration. All API calls are made to `/api/*` which are proxied to the backend server.

### Available Endpoints

- `GET /api/healthz` - Health check
- `GET /api/forecast/{product_id}` - Price forecasts
- `GET /api/prices/ah/{product_id}` - Auction house prices
- `GET /api/profit/craft/{product_id}` - Craft profitability
- `POST /api/backtest/run` - Run backtests
- `POST /api/ml/train` - Train ML models
- `POST /api/simulation/market` - Market simulation
- `POST /api/analysis/predictive` - Predictive analysis

## Project Structure

```
frontend/
├── src/
│   ├── api/           # API client and types
│   ├── components/    # Reusable React components
│   ├── pages/         # Page components
│   ├── hooks/         # Custom React hooks
│   ├── stores/        # State management
│   └── utils/         # Utility functions
├── public/            # Static assets
└── dist/              # Built application
```

## Features Overview

### Dashboard
- System health monitoring
- Quick stats and metrics
- Navigation to other features

### Market Analysis
- Real-time price data
- Interactive price charts
- Historical data visualization
- Item search functionality

### Craft Profitability
- Crafting cost analysis
- Profit margin calculations
- Material requirement breakdown
- ROI calculations

### Backtesting
- Strategy configuration
- Historical performance testing
- Results visualization
- Trade history analysis

### Market Simulation
- Agent-based market modeling
- Scenario configuration
- Performance metrics
- Comparative analysis

## Responsive Design

The interface is fully responsive and works on:
- Desktop (1200px+)
- Tablet (768px - 1199px)
- Mobile (< 768px)

## Error Handling

- Network error handling with user-friendly messages
- Loading states for all async operations
- Graceful degradation when API is unavailable
- Form validation and error feedback
