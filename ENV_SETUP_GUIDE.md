# Environment Variable Setup Guide

This guide explains how to set up environment variables using `.env` files for the SkyBlock Economy Two-Process Application.

## Overview

The application has been updated to use `.env` files for managing secrets instead of shell environment variables. This resolves the issue where the background data ingestion service couldn't access environment variables when launched with `nohup`.

## Changes Made

- **Environment Variable Loading**: Both `bot.py` and `data_ingestion/run_ingestion.py` now load variables from `.env` file
- **Dependency**: Added `python-dotenv>=1.0.0` to requirements
- **Startup Script**: `start_bot.sh` now checks for `.env` file instead of shell environment variables

## Setup Instructions

### 1. Install Required Dependency

The `python-dotenv` dependency is included in `requirements.txt`. Install it with:

```bash
pip install -r requirements.txt
```

Or install just this dependency:

```bash
pip install python-dotenv
```

### 2. Create .env File

Copy the example file and customize it:

```bash
cp .env.example .env
```

### 3. Edit .env File

Open `.env` in your text editor and add your actual tokens:

```bash
# Environment variables for SkyBlock Economy Discord Bot
# Copy this to .env and fill in your actual values

# Discord Bot Token (Required)
# Get this from https://discord.com/developers/applications
DISCORD_BOT_TOKEN=your_actual_discord_bot_token_here

# Hypixel API Key (Required for data ingestion service)
# Get this from https://developer.hypixel.net/
HYPIXEL_API_KEY=your_actual_hypixel_api_key_here
```

Replace the placeholder values with your actual tokens:

- `DISCORD_BOT_TOKEN`: Your Discord bot token from https://discord.com/developers/applications
- `HYPIXEL_API_KEY`: Your Hypixel API key from https://developer.hypixel.net/

### 4. Secure Your .env File

The `.env` file is already excluded from version control via `.gitignore`. **Never commit your `.env` file to the repository.**

Set appropriate file permissions (recommended):

```bash
chmod 600 .env
```

### 5. Run the Application

Use the updated launch script:

```bash
./start_bot.sh
```

The script will automatically:
- Check for the existence of `.env` file
- Verify that both required environment variables are present
- Load the variables for both the Discord bot and data ingestion service

## How It Works

### Python Code Changes

Both Python entry points now include:

```python
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
```

This loads the `.env` file before any other imports that might need environment variables.

### Shell Script Changes

`start_bot.sh` now checks for the `.env` file and its contents:

- **File existence**: Ensures `.env` file exists
- **Required variables**: Verifies both `DISCORD_BOT_TOKEN` and `HYPIXEL_API_KEY` are defined
- **Helpful messages**: Provides clear instructions if setup is incomplete

### Background Process Fix

The original issue was that `nohup` doesn't inherit shell environment variables. Now both processes load variables directly from the `.env` file, so this is no longer a problem.

## Troubleshooting

### Error: .env file not found

```
Error: .env file not found
Please create a .env file with your secrets. You can use .env.example as a template:
cp .env.example .env
```

**Solution**: Create the `.env` file as described in the setup instructions.

### Error: HYPIXEL_API_KEY not found in .env file

```
Error: HYPIXEL_API_KEY not found in .env file
The data ingestion service requires a Hypixel API key.
```

**Solution**: Add the `HYPIXEL_API_KEY=your_key_here` line to your `.env` file.

### Error: Missing packages

```
Missing packages: ['dotenv']
Run: pip install -r requirements.txt
```

**Solution**: Install the updated requirements that include `python-dotenv`.

## Migration from Environment Variables

If you were previously using shell environment variables:

1. **Old way** (no longer recommended):
   ```bash
   export DISCORD_BOT_TOKEN="your_token"
   export HYPIXEL_API_KEY="your_key"
   ./start_bot.sh
   ```

2. **New way** (current):
   ```bash
   # Create .env file with your tokens
   echo "DISCORD_BOT_TOKEN=your_token" > .env
   echo "HYPIXEL_API_KEY=your_key" >> .env
   ./start_bot.sh
   ```

The application will work the same way, but now both processes can reliably access the environment variables.

## Security Notes

- The `.env` file contains sensitive information and should never be shared
- The `.gitignore` file excludes `.env` to prevent accidental commits
- Consider using restrictive file permissions (`chmod 600 .env`)
- Never hardcode secrets in source code
- Use different tokens for development and production environments

## Testing

To verify your setup works correctly:

1. Create a test `.env` file with placeholder values
2. Run `./start_bot.sh` 
3. Check that the pre-flight checks pass and both processes attempt to start
4. The only errors should be network-related (if using test tokens)