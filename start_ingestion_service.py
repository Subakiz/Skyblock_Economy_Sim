#!/usr/bin/env python3
"""
Script to start the standalone data ingestion service.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from ingestion.standalone_ingestion_service import main

if __name__ == "__main__":
    print("Starting Skyblock Economy Simulator - Standalone Data Ingestion Service")
    print("=" * 70)
    print("This service will continuously fetch data from the Hypixel API and")
    print("write it to partitioned Parquet datasets.")
    print()
    print("Press Ctrl+C to stop the service.")
    print("=" * 70)
    
    try:
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nService stopped by user.")
    except Exception as e:
        print(f"\nService error: {e}")
        sys.exit(1)