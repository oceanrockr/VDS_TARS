# T.A.R.S. Phase 2 Quick Start Guide

## 🚀 What's New in Phase 2

Phase 2 adds **WebSocket Gateway** and **JWT Authentication** to T.A.R.S., enabling real-time streaming chat with your local LLM.

### Key Features
- ✅ JWT token-based authentication (24h expiry)
- ✅ WebSocket streaming at `ws://localhost:8000/ws/chat`
- ✅ Real-time token streaming from Ollama
- ✅ 10+ concurrent connections supported
- ✅ Automatic heartbeat (ping/pong every 30s)
- ✅ Comprehensive test suite (107 tests, 88% coverage)

---

## 🔧 Quick Setup

### 1. Start Services

```bash
cd VDS_TARS
docker-compose up -d
```

### 2. Verify Health

```bash
# Check all services
curl http://localhost:8000/health
curl http://localhost:8000/ready

# Check WebSocket service
curl http://localhost:8000/ws/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "websocket",
  "metrics": {
    "active_connections": 0,
    "total_connections": 0
  },
  "ollama_status": "healthy"
}
```

---

## 🔑 Authentication

### Get a Token

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "my-client",
    "device_name": "Terminal",
    "device_type": "bash"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Validate Token

```bash
TOKEN="your-access-token"

curl -X POST http://localhost:8000/auth/validate \
  -H "Authorization: Bearer $TOKEN"
```

---

## 💬 WebSocket Chat

### Using Python

```bash
cd docs/examples
pip install websockets httpx
python websocket_client_example.py
```

### Using wscat (Node.js)

```bash
# Install wscat
npm install -g wscat

# Connect (replace TOKEN)
wscat -c "ws://localhost:8000/ws/chat?token=YOUR_TOKEN_HERE"

# Send a message (after connection)
{"type":"chat","content":"Hello T.A.R.S.!","conversation_id":"test-1"}
```

### Expected Flow

```
# 1. You connect
> Connected

# 2. Server sends ack
< {"type":"connection_ack","client_id":"my-client","session_id":"..."}

# 3. You send message
> {"type":"chat","content":"Say hello","conversation_id":"conv-1"}

# 4. Server streams tokens
< {"type":"token","token":"Hello"}
< {"type":"token","token":"!"}
< {"type":"token","token":" How"}
< {"type":"token","token":" can"}
< {"type":"token","token":" I"}
< {"type":"token","token":" help"}
< {"type":"token","token":"?"}

# 5. Server sends completion
< {"type":"complete","total_tokens":7,"latency_ms":156}

# 6. Heartbeat (every 30s)
< {"type":"ping"}
> {"type":"pong"}
```

---

## 📊 Monitoring

### Active Sessions

```bash
curl http://localhost:8000/ws/sessions
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

### WebSocket Health

```bash
curl http://localhost:8000/ws/health
```

---

## 🧪 Testing

### Run All Tests

```bash
docker exec tars-backend pytest tests/ -v
```

### Run with Coverage

```bash
docker exec tars-backend pytest tests/ --cov=app --cov-report=term-missing
```

### Run Specific Tests

```bash
# Authentication tests
docker exec tars-backend pytest tests/test_auth.py -v

# WebSocket tests
docker exec tars-backend pytest tests/test_websocket.py -v

# Performance tests (manual)
docker exec tars-backend pytest tests/test_performance.py -v -s
```

---

## 🔍 API Endpoints

### Authentication
- `POST /auth/token` - Generate tokens
- `POST /auth/refresh` - Refresh access token
- `POST /auth/validate` - Validate token
- `POST /auth/revoke` - Revoke token
- `GET /auth/health` - Auth service health

### WebSocket
- `WS /ws/chat?token=<jwt>` - Chat endpoint
- `GET /ws/health` - WebSocket health
- `GET /ws/sessions` - Active sessions

### Health
- `GET /health` - Basic health check
- `GET /ready` - Readiness (includes Ollama)
- `GET /metrics` - Application metrics

### Documentation
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc
- `GET /openapi.json` - OpenAPI schema

---

## 📖 Message Types

### Client → Server

**Chat Message:**
```json
{
  "type": "chat",
  "content": "Your message here",
  "conversation_id": "optional-conv-id",
  "model": "optional-model-override",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Pong (heartbeat response):**
```json
{
  "type": "pong",
  "timestamp": "2025-11-07T12:00:00"
}
```

### Server → Client

**Connection Ack:**
```json
{
  "type": "connection_ack",
  "client_id": "your-client-id",
  "session_id": "unique-session-id"
}
```

**Token Stream:**
```json
{
  "type": "token",
  "token": "text",
  "conversation_id": "conv-id"
}
```

**Stream Complete:**
```json
{
  "type": "complete",
  "conversation_id": "conv-id",
  "total_tokens": 42,
  "latency_ms": 567.8
}
```

**Error:**
```json
{
  "type": "error",
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

**Ping (heartbeat):**
```json
{
  "type": "ping",
  "timestamp": "2025-11-07T12:00:00"
}
```

---

## 🛠️ Troubleshooting

### Issue: Cannot connect to WebSocket

**Check:**
```bash
# Is backend running?
docker ps | grep tars-backend

# Check logs
docker logs tars-backend

# Is token valid?
curl -X POST http://localhost:8000/auth/validate \
  -H "Authorization: Bearer $TOKEN"
```

### Issue: Ollama not responding

**Check:**
```bash
# Is Ollama running?
docker ps | grep tars-ollama

# Check Ollama health
curl http://localhost:11434/api/tags

# Check Ollama logs
docker logs tars-ollama

# Verify GPU
docker exec tars-ollama nvidia-smi
```

### Issue: Slow token streaming

**Check:**
- GPU available? (`nvidia-smi`)
- Model loaded? (first request loads model, takes ~30s)
- Network latency?
- CPU vs GPU inference? (GPU is 5-10x faster)

---

## 🔒 Security Notes

### Development (Current)
- ⚠️ HTTP only (no TLS)
- ⚠️ Default JWT secret
- ⚠️ CORS allows all origins

### Production (Required)
```bash
# 1. Generate secure JWT secret
openssl rand -hex 32

# 2. Update .env
JWT_SECRET_KEY=<your-secret-key>
HTTPS_ENABLED=true
CORS_ORIGINS=https://your-domain.com

# 3. Add TLS certificates
TLS_CERT_PATH=/etc/ssl/certs/tars.crt
TLS_KEY_PATH=/etc/ssl/private/tars.key
```

---

## 📚 Additional Resources

- **Full Report:** [PHASE2_IMPLEMENTATION_REPORT.md](PHASE2_IMPLEMENTATION_REPORT.md)
- **Python Example:** [docs/examples/websocket_client_example.py](docs/examples/websocket_client_example.py)
- **API Docs:** http://localhost:8000/docs
- **Main README:** [README.md](README.md)

---

## 🎯 Next Steps

Ready for **Phase 3: Document Indexing & RAG**

Phase 3 will add:
- ChromaDB vector database integration
- Document upload and indexing
- Semantic search
- RAG (Retrieval-Augmented Generation)
- NAS file monitoring
- Citation support

---

**Updated:** November 7, 2025
**Version:** v0.1.0-alpha
