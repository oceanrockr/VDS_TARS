# T.A.R.S. Phase 4 Quick Start Guide
## Get Started in 5 Minutes

**Version:** v0.2.0-alpha
**Phase:** Phase 4 - Client UI & NAS Monitoring
**Target:** Developers and DevOps Engineers

---

## Prerequisites

- ✅ Phase 3 backend running (see [PHASE3_QUICKSTART.md](PHASE3_QUICKSTART.md))
- ✅ Node.js 18+ installed
- ✅ npm or yarn package manager
- ✅ Modern web browser (Chrome, Firefox, Safari, Edge)

---

## 1. Start Backend Services

If Phase 3 backend is not running:

```bash
cd backend
docker-compose up -d

# Verify services
docker-compose ps

# Should see:
# - ollama (running)
# - chromadb (running)
# - backend (running)
```

Check backend health:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Expected response:
```json
{
  "status": "ready",
  "checks": {
    "ollama": "healthy",
    "chromadb": "healthy",
    "embedding_model": "healthy",
    "conversation_service": "healthy",
    "nas_watcher": "disabled"
  }
}
```

---

## 2. Install Frontend Dependencies

```bash
cd ui
npm install

# Or with yarn
yarn install
```

This will install:
- React 18.2.0
- TypeScript 5.2.2
- Vite 5.0.8
- TailwindCSS 3.3.6
- Axios, Recharts, and other dependencies

---

## 3. Configure Frontend

Create environment file:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# API Configuration
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# Feature Flags
VITE_ENABLE_METRICS=true
VITE_ENABLE_DEBUG=false

# UI Configuration
VITE_MAX_UPLOAD_SIZE_MB=50
VITE_CHAT_HISTORY_LIMIT=100
```

---

## 4. Start Development Server

```bash
npm run dev

# Or with yarn
yarn dev
```

Expected output:
```
  VITE v5.0.8  ready in 423 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h to show help
```

---

## 5. Access UI

Open your browser:
```
http://localhost:5173
```

You should see the T.A.R.S. login screen.

---

## 6. Authenticate

**Login:**
1. Enter a client ID (e.g., `test-client`)
2. Click "Connect"

The system will:
- Generate a JWT token
- Establish WebSocket connection
- Load conversation history
- Display the main interface

---

## 7. Test RAG Chat

**Steps:**
1. Ensure "Enable RAG" checkbox is checked
2. Type a question: `What is RAG?`
3. Click "Send"

**What happens:**
- Backend retrieves relevant document chunks
- Sources appear at the top
- Answer streams in real-time
- Citations displayed inline

**Example Response:**
```
📚 Sources (3)
• rag_guide.pdf (score: 0.92)
• ml_concepts.txt (score: 0.85)
• documentation.md (score: 0.78)

RAG (Retrieval-Augmented Generation) is an AI framework that combines
information retrieval with text generation...
```

---

## 8. Upload Documents

**Navigate to Upload Tab:**
1. Click "Upload" in top navigation
2. Enter document path: `/path/to/your/document.pdf`
3. Click "Index Document"

**Progress:**
- Progress bar shows indexing status
- Success message displays metadata
- Document now available for RAG queries

**Supported Formats:**
- PDF (.pdf)
- Microsoft Word (.docx)
- Plain Text (.txt)
- Markdown (.md)
- CSV (.csv)

---

## 9. View System Metrics

**Navigate to Metrics Tab:**
1. Click "Metrics" in top navigation
2. View real-time system stats

**Metrics Displayed:**
- CPU Usage
- Memory Usage
- GPU Usage (if available)
- Documents Indexed
- Total Chunks
- Queries Processed
- Average Retrieval Time

**Refresh:** Automatically every 5 seconds

---

## 10. Enable NAS Auto-Indexing (Optional)

**Backend Configuration:**

Edit `backend/.env` or `docker-compose.yml`:
```bash
NAS_WATCH_ENABLED=true
NAS_MOUNT_POINT=/mnt/nas/LLM_docs
NAS_SCAN_INTERVAL=3600  # Full scan every hour
```

**Mount NAS (if not already):**
```bash
# Linux/macOS
sudo mount -t nfs nas.local:/LLM_docs /mnt/nas/LLM_docs

# Or add to /etc/fstab for persistent mounting
nas.local:/LLM_docs /mnt/nas/LLM_docs nfs defaults,ro 0 0
```

**Restart Backend:**
```bash
docker-compose restart backend
```

**Verify NAS Watcher:**
```bash
curl http://localhost:8000/ready | jq '.checks.nas_watcher'

# Should show: "enabled"
```

**Test Auto-Indexing:**
1. Copy a document to NAS: `cp test.pdf /mnt/nas/LLM_docs/`
2. Wait 5 seconds (debounce delay)
3. Check stats: `curl http://localhost:8000/rag/stats`
4. Document count should increase

---

## Common Tasks

### Create New Conversation

1. Click "New Chat" in sidebar
2. New conversation ID generated
3. Start chatting

### View Conversation History

1. Check sidebar for previous conversations
2. Click on any conversation to resume
3. Scroll through message history

### Delete Conversation

1. Hover over conversation in sidebar
2. Click trash icon
3. Confirm deletion

### Toggle RAG On/Off

1. In chat panel, check/uncheck "Enable RAG"
2. When disabled: Standard LLM responses (no document retrieval)
3. When enabled: RAG-enhanced responses with citations

### Check Service Health

```bash
# Overall health
curl http://localhost:8000/health

# Detailed readiness
curl http://localhost:8000/ready

# RAG service
curl http://localhost:8000/rag/health

# Conversation service
curl http://localhost:8000/conversation/health

# Metrics service
curl http://localhost:8000/metrics/health
```

---

## Troubleshooting

### Frontend Won't Start

**Problem:** `npm run dev` fails

**Solutions:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 18+

# Try different port
VITE_PORT=5174 npm run dev
```

### Cannot Connect to Backend

**Problem:** "Disconnected" status in UI

**Solutions:**
```bash
# 1. Check backend is running
curl http://localhost:8000/health

# 2. Check WebSocket is accessible
wscat -c ws://localhost:8000/ws/chat?token=test

# 3. Verify CORS settings in backend/.env
CORS_ORIGINS=http://localhost:5173

# 4. Restart backend
docker-compose restart backend
```

### RAG Not Working

**Problem:** No sources returned, or "No relevant documents found"

**Solutions:**
```bash
# 1. Check documents are indexed
curl http://localhost:8000/rag/stats

# 2. Verify ChromaDB connection
curl http://localhost:8000/rag/health

# 3. Check embedding service
curl http://localhost:8000/ready | jq '.checks.embedding_model'

# 4. Try re-indexing
curl -X POST http://localhost:8000/rag/index \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"file_path": "/path/to/doc.pdf"}'
```

### NAS Watcher Not Working

**Problem:** Files added to NAS are not indexed

**Solutions:**
```bash
# 1. Check NAS is mounted
mount | grep /mnt/nas

# 2. Verify watcher is enabled
curl http://localhost:8000/ready | jq '.checks.nas_watcher'

# 3. Check watcher stats
curl http://localhost:8000/rag/stats

# 4. Check logs
docker-compose logs backend | grep NAS

# 5. Trigger manual scan
# (Add endpoint in Phase 5)
```

### Metrics Not Loading

**Problem:** Metrics dashboard shows errors

**Solutions:**
```bash
# 1. Check metrics service
curl http://localhost:8000/metrics/health

# 2. Test metrics endpoint
curl http://localhost:8000/metrics/system

# 3. Check for GPU (optional)
nvidia-smi  # Should show GPU info

# 4. Restart backend
docker-compose restart backend
```

---

## Production Deployment

### Build Frontend for Production

```bash
cd ui
npm run build

# Output: ui/dist/
# Contains optimized static files
```

### Serve with Nginx

```nginx
server {
    listen 80;
    server_name tars.yourdomain.com;

    root /path/to/VDS_TARS/ui/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /ws/ {
        proxy_pass http://localhost:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }
}
```

### Enable HTTPS

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d tars.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Docker Production Build

```dockerfile
# ui/Dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Build and run:
```bash
docker build -t tars-ui:latest ui/
docker run -d -p 80:80 tars-ui:latest
```

---

## Next Steps

### Phase 5 Preview

After completing Phase 4, you'll be ready for:

1. **Advanced RAG** - Cross-encoder reranking, query expansion
2. **Semantic Chunking** - Better document splitting with LangChain
3. **Hybrid Search** - Combine keyword and vector search
4. **Multi-Document Reasoning** - Answer questions across multiple sources
5. **Analytics Dashboard** - Query patterns, document popularity

### Learn More

- [Phase 4 Implementation Report](PHASE4_IMPLEMENTATION_REPORT.md)
- [Phase 3 Documentation](PHASE3_IMPLEMENTATION_REPORT.md)
- [API Documentation](http://localhost:8000/docs)
- [Architecture Overview](README.md)

---

## Support

### Getting Help

- Check logs: `docker-compose logs -f backend`
- View browser console: Press F12 in browser
- Test API: `curl http://localhost:8000/docs`
- Health checks: See "Common Tasks" section above

### Reporting Issues

Include in your report:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Browser and OS
- Backend logs
- Frontend console errors

---

**Quick Start Complete!** 🎉

You now have a fully functional T.A.R.S. system with:
- ✅ Modern React UI
- ✅ Real-time RAG chat
- ✅ Document upload and indexing
- ✅ Conversation history
- ✅ System metrics dashboard
- ✅ Optional NAS auto-indexing

Ready for Phase 5: Advanced RAG & Semantic Chunking

---

**Last Updated:** November 7, 2025
**Version:** v0.2.0-alpha
**Phase:** 4 of 7
