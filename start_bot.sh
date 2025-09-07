#!/bin/bash
#
# Master Launch Script for SkyBlock Economy Two-Process Application
# Implements the mandated two-process architecture:
# 1. Background Data Ingestion Service (data_ingestion/run_ingestion.py)
# 2. Foreground Discord Bot (bot.py)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID file for ingestion service
INGESTION_PID_FILE="logs/ingestion_service.pid"

echo -e "${GREEN}SkyBlock Economy Two-Process Application Launcher${NC}"
echo "================================================================"
echo -e "${BLUE}Production-Grade Hypixel Market Analysis & Sniping System${NC}"
echo "================================================================"

# Cleanup function to kill ingestion service on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down application...${NC}"
    
    # Kill ingestion service if running
    if [[ -f "$INGESTION_PID_FILE" ]]; then
        INGESTION_PID=$(cat "$INGESTION_PID_FILE")
        if ps -p "$INGESTION_PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping data ingestion service (PID: $INGESTION_PID)...${NC}"
            kill "$INGESTION_PID" 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if ps -p "$INGESTION_PID" > /dev/null 2>&1; then
                echo -e "${RED}Force killing ingestion service...${NC}"
                kill -9 "$INGESTION_PID" 2>/dev/null || true
            fi
        fi
        rm -f "$INGESTION_PID_FILE"
    fi
    
    echo -e "${GREEN}Application shutdown complete.${NC}"
}

# Set trap for clean shutdown
trap cleanup EXIT INT TERM

# Check if we're in the right directory
if [[ ! -f "bot.py" ]]; then
    echo -e "${RED}Error: bot.py not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
if ! python3 -c 'import sys; assert sys.version_info >= (3, 8)'; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${RED}Error: Python 3.8+ is required. You have: $PYTHON_VERSION${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo -e "Python version: ${GREEN}$PYTHON_VERSION${NC}"

# Check for Discord bot token
if [[ -f ".env" ]]; then
    echo "✅ .env file found"
    # Check if DISCORD_BOT_TOKEN exists in .env file
    if ! grep -q "^DISCORD_BOT_TOKEN=" .env; then
        echo -e "${YELLOW}Warning: DISCORD_BOT_TOKEN not found in .env file${NC}"
        echo "Please add your Discord bot token to .env file:"
        echo "DISCORD_BOT_TOKEN=your_token_here"
        echo ""
        echo "Continue anyway? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check for Hypixel API key (required for data ingestion)
    if ! grep -q "^HYPIXEL_API_KEY=" .env; then
        echo -e "${RED}Error: HYPIXEL_API_KEY not found in .env file${NC}"
        echo "The data ingestion service requires a Hypixel API key."
        echo "Please add your Hypixel API key to .env file:"
        echo "HYPIXEL_API_KEY=your_key_here"
        exit 1
    fi
else
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create a .env file with your secrets. You can use .env.example as a template:"
    echo "cp .env.example .env"
    echo "Then edit .env and add your actual tokens:"
    echo "- DISCORD_BOT_TOKEN=your_discord_bot_token"  
    echo "- HYPIXEL_API_KEY=your_hypixel_api_key"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "
import sys
required_packages = ['discord', 'pandas', 'yaml', 'numpy', 'pyarrow', 'dotenv']
missing = []

for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'Missing packages: {missing}')
    print('Run: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('✅ All required packages available')
"

# Create necessary directories
echo "Setting up directories..."
mkdir -p data logs data/auction_history data/bazaar_history

# Check configuration
if [[ ! -f "config/config.yaml" ]]; then
    echo -e "${RED}Error: config/config.yaml not found${NC}"
    exit 1
fi

# Check if data ingestion entry point exists
if [[ ! -f "data_ingestion/run_ingestion.py" ]]; then
    echo -e "${RED}Error: data_ingestion/run_ingestion.py not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pre-flight checks passed${NC}"
echo ""

# Phase 1: Start Data Ingestion Service (Background Process)
echo -e "${BLUE}Phase 1: Starting Data Ingestion Service...${NC}"
nohup python3 data_ingestion/run_ingestion.py > logs/ingestion.log 2>&1 &
INGESTION_PID=$!
echo $INGESTION_PID > "$INGESTION_PID_FILE"

echo -e "${GREEN}✅ Data ingestion service started (PID: $INGESTION_PID)${NC}"
echo -e "   Logs: ${YELLOW}logs/ingestion.log${NC}"
echo -e "   Features: Canonical item normalization, auction/bazaar loops, Parquet output"
echo ""

# Give ingestion service a moment to start
sleep 3

# Verify ingestion service is running
if ! ps -p "$INGESTION_PID" > /dev/null 2>&1; then
    echo -e "${RED}Error: Data ingestion service failed to start${NC}"
    echo "Check logs/ingestion.log for details"
    exit 1
fi

# Phase 2: Start Discord Bot (Foreground Process)
echo -e "${BLUE}Phase 2: Starting Discord Bot...${NC}"
echo -e "   Features: Auction sniping, market intelligence, read-only consumer"
echo -e "   Hunter task: Page 0 scanning every 2 seconds"
echo -e "   Analyst task: Market intelligence updates every 60 seconds"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop both processes${NC}"
echo "================================================================"

# Add timestamp to logs and run the bot
exec python3 bot.py 2>&1 | while IFS= read -r line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') [BOT] $line"
done
