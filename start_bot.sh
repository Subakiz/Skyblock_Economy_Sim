#!/bin/bash
#
# Launch script for SkyBlock Economy Discord Bot
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}SkyBlock Economy Discord Bot Launcher${NC}"
echo "========================================="

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
if [[ -z "$DISCORD_BOT_TOKEN" ]]; then
    echo -e "${YELLOW}Warning: DISCORD_BOT_TOKEN environment variable not set${NC}"
    echo "Please set your Discord bot token:"
    echo "export DISCORD_BOT_TOKEN=\"your_token_here\""
    echo ""
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for Hypixel API key
if [[ -z "$HYPIXEL_API_KEY" ]]; then
    echo -e "${YELLOW}Note: HYPIXEL_API_KEY not set - data collection will be limited${NC}"
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "
import sys
# --- CORRECTED PACKAGE NAME ---
required_packages = ['discord', 'pandas', 'yaml', 'numpy', 'lightgbm']
# --- END CORRECTION ---
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
mkdir -p data models logs

# Check configuration
if [[ ! -f "config/config.yaml" ]]; then
    echo -e "${RED}Error: config/config.yaml not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pre-flight checks passed${NC}"
echo ""

# Run the bot
echo -e "${GREEN}Starting Discord bot...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Add timestamp to logs
exec python3 bot.py 2>&1 | while IFS= read -r line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $line"
done
