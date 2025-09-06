# Docker Quick Start Guide

## Using Docker Compose (Recommended)

1. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your Discord bot token and optional Hypixel API key
   ```

2. **Start the bot:**
   ```bash
   docker-compose up -d
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f skyblock-bot
   ```

4. **Stop the bot:**
   ```bash
   docker-compose down
   ```

## Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t skyblock-bot .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name skyblock-bot \
     -e DISCORD_BOT_TOKEN="your_token_here" \
     -e HYPIXEL_API_KEY="your_api_key_here" \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/logs:/app/logs \
     --restart unless-stopped \
     skyblock-bot
   ```

3. **View logs:**
   ```bash
   docker logs -f skyblock-bot
   ```

## Data persistence

The Docker setup includes volume mounts for:
- `./data` - Raw SkyBlock data (NDJSON files)
- `./models` - Trained ML models
- `./logs` - Application logs

This ensures your data persists across container restarts.

## Health checks

The container includes health checks that verify the bot configuration is valid. You can check the health status with:

```bash
docker ps  # Shows health status
docker inspect skyblock-bot | grep Health -A 10  # Detailed health info
```