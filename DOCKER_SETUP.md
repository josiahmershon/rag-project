# Docker Setup Guide for RAG Project

## Prerequisites

1. **Install Docker Desktop**
   - Windows/Mac: https://www.docker.com/products/docker-desktop
   - Linux: `sudo apt-get install docker.io docker-compose`

2. **Verify Installation**
   ```bash
   docker --version
   docker-compose --version
   ```

## Quick Start (3 Steps)

### Step 1: Create Your .env File
```bash
cd /home/user/rag-project
cp backend/env.example .env
```

Edit `.env` and set your real password:
```bash
DB_PASSWORD=your_actual_password
```

### Step 2: Start Everything
```bash
docker-compose up -d
```

This command:
- `-d` = detached mode (runs in background)
- Starts PostgreSQL, Backend, and Frontend
- Automatically creates database schema
- Connects everything together

### Step 3: Wait for Startup
```bash
# Watch the logs
docker-compose logs -f

# Check if services are ready
docker-compose ps
```

## Accessing Your Application

- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs
- **Frontend UI**: http://localhost:8002
- **Database**: localhost:5432

## Common Commands

### See What's Running
```bash
docker-compose ps
```

### View Logs
```bash
# All services
docker-compose logs -f

# Just backend
docker-compose logs -f backend

# Just database
docker-compose logs -f postgres
```

### Stop Everything
```bash
docker-compose down
```

### Stop and Delete Data
```bash
docker-compose down -v
```
**WARNING**: This deletes your database!

### Restart a Service
```bash
# Restart backend only
docker-compose restart backend

# Rebuild and restart backend
docker-compose up -d --build backend
```

### Execute Commands Inside Containers
```bash
# Access database
docker-compose exec postgres psql -U postgres

# Access backend shell
docker-compose exec backend bash

# Run database migrations
docker-compose exec backend python ingest.py
```

## Initial Data Setup

After first startup, load sample documents:
```bash
docker-compose exec backend python ingest.py
```

## Troubleshooting

### Container Won't Start?
```bash
# Check logs for errors
docker-compose logs backend

# Try rebuilding
docker-compose build --no-cache backend
docker-compose up -d backend
```

### Database Connection Errors?
```bash
# Check if postgres is healthy
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Manually test connection
docker-compose exec postgres psql -U postgres -c "SELECT 1;"
```

### Port Already in Use?
Edit `docker-compose.yml` and change ports:
```yaml
ports:
  - "8001:8001"  # Change 8001 to 8101 if needed
```

### vLLM Connection Issues?
The backend tries to connect to vLLM at `http://host.docker.internal:8000/v1`

If you're running vLLM locally:
- **Mac/Windows**: Should work automatically
- **Linux**: Add this to docker-compose.yml under backend:
  ```yaml
  extra_hosts:
    - "host.docker.internal:host-gateway"
  ```

Or if vLLM is in another container:
```yaml
environment:
  - VLLM_BASE_URL=http://vllm:8000/v1
```

## Production Considerations

### 1. Use Environment-Specific Settings
```yaml
# docker-compose.prod.yml
environment:
  - VLLM_BASE_URL=https://your-vllm-server.com/v1
```

### 2. Set Up Secrets Management
Don't use `.env` in production. Use:
- Docker secrets
- AWS Secrets Manager
- HashiCorp Vault

### 3. Add Nginx Reverse Proxy
```yaml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
```

### 4. Set Resource Limits
```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
```

## Understanding the Setup

### What Happens When You Run `docker-compose up`?

1. **Creates Network** (`rag_network`)
   - All containers can talk to each other by name
   - Example: backend reaches postgres via `postgres:5432`

2. **Starts PostgreSQL**
   - Pulls `pgvector/pgvector:pg16` image (if not present)
   - Runs `schema.sql` automatically on first start
   - Data saved in `postgres_data` volume (persists between restarts)

3. **Builds Backend**
   - Reads `backend/Dockerfile`
   - Installs Python dependencies
   - Waits for postgres to be healthy
   - Starts FastAPI on port 8001

4. **Builds Frontend**
   - Reads `frontend/Dockerfile`
   - Installs Chainlit
   - Connects to backend
   - Starts on port 8002

### File Structure After Docker Setup
```
rag-project/
├── docker-compose.yml       # Main orchestration file
├── .env                     # Your secrets (NOT in git)
├── .dockerignore           # Files to exclude from builds
├── backend/
│   ├── Dockerfile          # Backend container recipe
│   ├── schema.sql          # Database setup
│   └── ...
└── frontend/
    ├── Dockerfile          # Frontend container recipe
    └── ...
```

## Next Steps

1. **Test the Setup**
   ```bash
   curl http://localhost:8001/health
   ```

2. **Upload Documents**
   Visit: http://localhost:8001/upload

3. **Query via Frontend**
   Visit: http://localhost:8002

4. **Load Sample Data**
   ```bash
   docker-compose exec backend python ingest.py
   ```

## FAQ

**Q: Do I need to install Python/PostgreSQL on my machine?**
A: No! Docker handles everything.

**Q: Can I edit code while containers are running?**
A: Yes! The volumes in docker-compose.yml mount your local code.

**Q: How do I deploy this?**
A: Copy docker-compose.yml to your server and run `docker-compose up -d`.

**Q: Is data lost when I stop containers?**
A: No. PostgreSQL data is in a volume. Use `docker-compose down -v` to delete it.

**Q: Can I use this with Docker Swarm or Kubernetes?**
A: Yes! Convert docker-compose.yml to Kubernetes manifests with Kompose.
