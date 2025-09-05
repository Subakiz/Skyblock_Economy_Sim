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

- Phase 2
  - [ ] Auctions roll-ups and BIN analytics
  - [ ] Item ontology coverage + craft-path PnL service
  - [ ] Backtesting framework with fees, slippage, and fill logic
  - [ ] Event feature ingestion (mayors, seasonal)

- Phase 3
  - [ ] Multivariate ML (LightGBM/XGBoost)
  - [ ] Agent-based modeling (Mesa)
  - [ ] Scenario engine for shock testing
  - [ ] Portfolio and risk module, strategy orchestration

## Notes

- Respect API limits; this scaffold includes a token-bucket rate limiter.
- The Hypixel API and game economy evolve. Keep ontology and assumptions versioned.
- For Auctions at scale, consider object storage + Parquet; don’t overload Postgres.
