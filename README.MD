# SkyBlock Economic Modeling (Blueprint + MVP)

This repository is a production-minded scaffold for building a comprehensive economic model of Hypixel SkyBlock.

## Core Components

- Ingestion
  - Poll Bazaar and Auctions (later) and persist time-series snapshots.
  - Store raw auctions in object storage and roll up signals in Postgres/Timescale.

- Item Ontology
  - JSON-based supply graph describing crafting relationships, sinks, taxes/fees.
  - Enables dependency-aware features and “upstream/downstream” propagation.

- Feature Engineering
  - Rolling MAs/EMAs, volatility, spreads, depth proxies, cross-item lags.
  - Event features (mayors, festivals, patch flags).

- Forecasting
  - Baseline per-series forecast; extend to multivariate ML.
  - Signals for strategy: undervaluation, spread mean-reversion, craft-path arbitrage.

- Simulation & Scenarios
  - Agent-based modeling (ABM) stubs for future: simulate player cohorts and reactions.
  - Scenario engine to test shocks and update assumptions.

- API & Dashboards
  - Serve predictions and craft-path profitability via FastAPI.
  - Hook up Grafana/Metabase for monitoring.

## Quickstart

1) Prereqs
- Python 3.10+
- Postgres 14+ (TimescaleDB optional but recommended)
- Env vars:
  - `HYPIXEL_API_KEY`
  - `DATABASE_URL` (e.g., `postgresql://user:pass@localhost:5432/skyblock`)

2) Initialize DB
- Run `psql "$DATABASE_URL" -f storage/schema.sql`

3) Configure
- Edit `config/config.yaml` as needed (poll interval, endpoints).

4) Ingest Bazaar
- `python -m ingestion.bazaar_collector`

5) Build Features
- `python -m modeling.features.feature_pipeline`

6) Train Model
- `python -m modeling.forecast.price_forecaster`

7) Serve API
- `uvicorn services.api.app:app --reload`

## Roadmap (MVP → Advanced)

- MVP
  - [x] Bazaar snapshots
  - [x] Basic features (MAs, spreads, vol)
  - [x] Baseline forecaster
  - [x] API endpoints for latest forecast

- Phase 2 ✅
  - [x] Auctions roll-ups and BIN analytics
  - [x] Item ontology coverage + craft-path PnL service
  - [x] Backtesting framework with fees, slippage, and fill logic
  - [x] Event feature ingestion (mayors, seasonal)
  - [x] **No-Database Mode** - NDJSON file storage alternative to PostgreSQL

- Phase 3 ✅
  - [x] Multivariate ML (LightGBM/XGBoost)
  - [x] Agent-based modeling (Mesa)
  - [x] Scenario engine for shock testing
  - [x] Proactive market prediction and insight generation

## Storage Options

The system supports two storage modes:

### Database Mode (Default)
Full-featured mode using PostgreSQL for complex queries, forecasting, and ML features.

### No-Database Mode 
Lightweight mode using NDJSON files for basic price tracking and profitability analysis. Perfect for development, testing, or lightweight deployments.

```yaml
# Enable in config/config.yaml
no_database_mode:
  enabled: true
  data_directory: "data"
```

See [docs/NO_DATABASE_MODE.md](docs/NO_DATABASE_MODE.md) for complete documentation.

## Testing & Development

### Quick Tests
Run the smoke tests to validate the installation:
```bash
python test_smoke.py
```

Test the no-database mode functionality:
```bash
python test_no_database_mode.py
```

### Component Demos  
Explore individual components with interactive demos:
```bash
python demo.py
```

### Individual Module Usage

**Rate-Limited API Client:**
```python
from ingestion.common.hypixel_client import HypixelClient
client = HypixelClient("https://api.hypixel.net", "your-api-key")
data = client.get_json("/skyblock/bazaar")
```

**Feature Engineering (standalone):**
```bash
python -m modeling.features.feature_pipeline
```

**Single Product Forecasting:**
```bash
python -m modeling.forecast.price_forecaster ENCHANTED_LAPIS_BLOCK
```

**Craft Profitability Analysis (Phase 2):**
```bash
python -m modeling.profitability.crafting_profit ENCHANTED_LAPIS_BLOCK --horizon 1h --pricing median
```

**Strategy Backtesting (Phase 2):**
```bash
python -m backtesting.run --strategy craft_and_sell --item ENCHANTED_LAPIS_BLOCK --start 2025-08-01 --end 2025-09-01 --capital 10000000
```

**Advanced ML Forecasting (Phase 3):**
```bash
python -m modeling.forecast.ml_forecaster ENCHANTED_LAPIS_BLOCK lightgbm
```

**Agent-Based Market Simulation (Phase 3):**
```bash
python phase3_cli.py simulate --scenario volatile_market --agents 100 --steps 500
```

**Predictive Market Analysis (Phase 3):**
```bash
python phase3_cli.py analyze "HYPERION,NECRON_CHESTPLATE" --model-type lightgbm
```

**Auction House Data Collection (Phase 2):**
```bash
python -m ingestion.auction_collector
```

## Notes

- Respect API limits; this scaffold includes a token-bucket rate limiter.
- The Hypixel API and game economy evolve. Keep ontology and assumptions versioned.
- For Auctions at scale, consider object storage + Parquet; don’t overload Postgres.
- All components are designed to be modular and can be used independently.
