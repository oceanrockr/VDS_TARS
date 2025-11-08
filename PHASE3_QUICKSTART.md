# T.A.R.S. Phase 3 Quick Start Guide
## Document Indexing & RAG in 10 Minutes

**Version:** v0.2.0-alpha
**Last Updated:** November 7, 2025

---

## Prerequisites

✅ Phase 2 complete and running
✅ Docker & Docker Compose installed
✅ NVIDIA GPU with CUDA support (optional, but recommended)
✅ 8GB+ RAM available
✅ 10GB+ disk space

---

## 1. Start T.A.R.S. with RAG

```bash
# Navigate to project directory
cd VDS_TARS

# Start all services (Ollama, ChromaDB, Backend)
docker-compose up -d

# Wait for services to initialize (~60 seconds)
docker-compose logs -f backend

# Look for these log messages:
# ✅ "Ollama service is healthy"
# ✅ "RAG service initialized successfully"
# ✅ "Uvicorn running on http://0.0.0.0:8000"
```

---

## 2. Verify All Components

```bash
# Check overall health
curl http://localhost:8000/health

# Check RAG service
curl http://localhost:8000/rag/health

# Expected output:
{
  "status": "healthy",
  "chromadb_status": "healthy",
  "embedding_model_status": "healthy",
  "collections": {
    "tars_documents": {
      "total_documents": 0,
      "total_chunks": 0
    }
  }
}
```

---

## 3. Authenticate

```bash
# Generate an access token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "quickstart-user",
    "device_name": "Terminal",
    "device_type": "linux"
  }' | jq -r '.access_token' > /tmp/token.txt

# Store token as environment variable
export TOKEN=$(cat /tmp/token.txt)

# Verify token
curl -X POST http://localhost:8000/auth/validate \
  -H "Authorization: Bearer $TOKEN"
```

---

## 4. Index Your First Document

### Option A: Index a Test Document

```bash
# Create a test markdown document
cat > /tmp/test_document.md << 'EOF'
# T.A.R.S. Overview

T.A.R.S. (Temporal Augmented Retrieval System) is a local LLM platform
with RAG capabilities. It uses Mistral 7B for generation and MiniLM-L6-v2
for embeddings.

## Key Features

- Multi-format document ingestion
- Real-time token streaming
- Citation-backed responses
- GPU-accelerated embeddings

## Architecture

T.A.R.S. consists of three main components:
1. Ollama (LLM inference)
2. ChromaDB (vector storage)
3. FastAPI Backend (orchestration)
EOF

# Index the document
curl -X POST http://localhost:8000/rag/index \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/tmp/test_document.md",
    "force_reindex": false
  }'

# Expected output:
{
  "document_id": "doc_abc123...",
  "file_name": "test_document.md",
  "status": "success",
  "chunks_created": 2,
  "processing_time_ms": 450.5
}
```

### Option B: Index an Existing PDF

```bash
# Index a PDF from your filesystem
curl -X POST http://localhost:8000/rag/index \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/your/document.pdf",
    "force_reindex": false
  }'
```

---

## 5. Your First RAG Query

### REST API (Non-Streaming)

```bash
# Query the indexed documents
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is T.A.R.S.?",
    "top_k": 3,
    "relevance_threshold": 0.7,
    "include_sources": true,
    "rerank": true
  }' | jq

# Expected output:
{
  "query": "What is T.A.R.S.?",
  "answer": "T.A.R.S. (Temporal Augmented Retrieval System) is a local LLM platform...",
  "sources": [
    {
      "file_name": "test_document.md",
      "similarity_score": 0.92,
      "excerpt": "T.A.R.S. (Temporal Augmented Retrieval System) is a local LLM..."
    }
  ],
  "total_tokens": 45,
  "retrieval_time_ms": 87.3,
  "generation_time_ms": 892.5,
  "total_time_ms": 979.8
}
```

### REST API (Streaming)

```bash
# Streaming query (returns NDJSON)
curl -X POST http://localhost:8000/rag/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features?",
    "top_k": 5
  }'

# Output (NDJSON):
{"type":"rag_sources","sources":[...],"retrieval_time_ms":95.2}
{"type":"rag_token","token":"The","has_sources":true}
{"type":"rag_token","token":" key","has_sources":true}
...
{"type":"rag_complete","total_tokens":120,"total_time_ms":2246.0}
```

---

## 6. WebSocket RAG Chat

### Install wscat (WebSocket client)

```bash
npm install -g wscat
```

### Connect and Chat

```bash
# Connect to WebSocket endpoint
wscat -c "ws://localhost:8000/ws/chat?token=$TOKEN"

# Wait for connection acknowledgment
< {"type":"connection_ack","client_id":"quickstart-user","session_id":"..."}

# Send a regular chat message (no RAG)
> {"type":"chat","content":"Hello!"}

# Receive streaming tokens
< {"type":"token","token":"Hello"}
< {"type":"token","token":"!"}
< {"type":"complete","total_tokens":5}

# Send a RAG-enabled message
> {"type":"chat","content":"What is T.A.R.S.?","use_rag":true,"rag_top_k":3}

# Receive RAG response
< {"type":"rag_sources","sources":[...],"retrieval_time_ms":95.2}
< {"type":"rag_token","token":"T","has_sources":true}
< {"type":"rag_token","token":".A.R.S.","has_sources":true}
< {"type":"rag_token","token":" is","has_sources":true}
...
< {"type":"rag_complete","total_tokens":120,"sources_count":2}
```

---

## 7. Batch Indexing

```bash
# Create multiple test documents
echo "Document 1 content..." > /tmp/doc1.txt
echo "Document 2 content..." > /tmp/doc2.txt
echo "Document 3 content..." > /tmp/doc3.txt

# Batch index
curl -X POST http://localhost:8000/rag/index/batch \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": [
      "/tmp/doc1.txt",
      "/tmp/doc2.txt",
      "/tmp/doc3.txt"
    ],
    "force_reindex": false,
    "max_concurrent": 3
  }' | jq

# Expected output:
{
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "already_indexed": 0,
  "total_chunks": 6,
  "processing_time_ms": 1250.5,
  "results": [...]
}
```

---

## 8. Monitor Your Collection

```bash
# Get collection statistics
curl http://localhost:8000/rag/stats | jq

# Output:
{
  "collection_name": "tars_documents",
  "total_documents": 4,
  "total_chunks": 8,
  "total_size_mb": 0.15,
  "last_updated": "2025-11-07T12:00:00",
  "embedding_dimension": 384
}

# Get active WebSocket sessions
curl http://localhost:8000/ws/sessions | jq
```

---

## 9. Search Documents (No Generation)

```bash
# Search for relevant chunks without generating an answer
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "architecture",
    "top_k": 5
  }' | jq

# Output: List of matching chunks with similarity scores
{
  "query": "architecture",
  "results": [
    {
      "chunk_id": "doc_abc_chunk_2",
      "content": "T.A.R.S. consists of three main components...",
      "similarity_score": 0.89,
      "metadata": {...}
    }
  ],
  "search_time_ms": 87.3
}
```

---

## 10. Clean Up & Maintenance

### Delete a Document

```bash
curl -X DELETE http://localhost:8000/rag/document \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_abc123..."
  }' | jq
```

### View Logs

```bash
# Backend logs
docker-compose logs -f backend

# ChromaDB logs
docker-compose logs -f chromadb

# Ollama logs
docker-compose logs -f ollama
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart only backend (preserves data)
docker-compose restart backend
```

---

## Common Issues & Solutions

### Issue: "Embedding model not loaded"

**Solution:**
```bash
# Check backend logs
docker-compose logs backend | grep "Embedding"

# Manually initialize
curl -X POST http://localhost:8000/rag/initialize \
  -H "Authorization: Bearer $TOKEN"
```

### Issue: "ChromaDB connection failed"

**Solution:**
```bash
# Check ChromaDB is running
docker-compose ps chromadb

# Check ChromaDB health
curl http://localhost:8001/api/v1/heartbeat

# Restart ChromaDB
docker-compose restart chromadb
```

### Issue: "File not found" when indexing

**Solution:**
```bash
# Documents must be accessible from Docker container
# Use absolute paths or mount volumes

# Check file exists in container
docker exec tars-backend ls /path/to/file.pdf

# Or mount directory in docker-compose.yml:
# volumes:
#   - /host/path:/container/path:ro
```

### Issue: Slow indexing/queries on CPU

**Solution:**
```bash
# Edit .env to use CPU explicitly
EMBED_DEVICE=cpu

# Reduce batch size for lower memory
EMBED_BATCH_SIZE=8

# Or enable GPU (recommended)
# Ensure NVIDIA Container Toolkit is installed
docker exec tars-backend nvidia-smi
```

---

## Configuration Quick Reference

### Key Environment Variables

```bash
# ChromaDB
CHROMA_HOST=http://chromadb:8000
CHROMA_COLLECTION_NAME=tars_documents

# Embedding
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_DEVICE=cuda  # or 'cpu'
EMBED_BATCH_SIZE=32

# Document Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=.pdf,.docx,.txt,.md,.csv

# RAG
RAG_TOP_K=5
RAG_RELEVANCE_THRESHOLD=0.7
RAG_RERANK_ENABLED=true
```

### Supported Document Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | .pdf | Text-based PDFs only (OCR planned) |
| Word | .docx | Full paragraph extraction |
| Text | .txt | Auto-encoding detection |
| Markdown | .md | Full formatting preserved |
| CSV | .csv | Table-to-text conversion |

---

## Next Steps

### Explore the API

Open the interactive API documentation:
```bash
open http://localhost:8000/docs
```

### Index Real Documents

1. Mount your document directory in `docker-compose.yml`:
```yaml
services:
  backend:
    volumes:
      - /path/to/your/docs:/mnt/docs:ro
```

2. Batch index all documents:
```bash
# Generate file list
find /mnt/docs -type f \( -name "*.pdf" -o -name "*.docx" \) > /tmp/files.txt

# Index (Python script)
python <<EOF
import requests, json

with open('/tmp/files.txt') as f:
    files = [line.strip() for line in f]

response = requests.post(
    'http://localhost:8000/rag/index/batch',
    headers={'Authorization': f'Bearer {TOKEN}'},
    json={'file_paths': files, 'max_concurrent': 5}
)
print(json.dumps(response.json(), indent=2))
EOF
```

### Build a Client

See [docs/examples/websocket_client_example.py](docs/examples/websocket_client_example.py) for a complete Python client implementation.

### Monitor Performance

```bash
# Real-time metrics
watch -n 1 'curl -s http://localhost:8000/rag/stats | jq'

# Query latency
time curl -X POST http://localhost:8000/rag/query \
  -d '{"query":"test"}' > /dev/null
```

---

## Advanced Usage

### Custom Relevance Threshold

```bash
# Higher threshold = more strict (fewer, more relevant results)
curl -X POST http://localhost:8000/rag/query \
  -d '{"query":"test","relevance_threshold":0.9}'

# Lower threshold = more permissive (more results, less relevant)
curl -X POST http://localhost:8000/rag/query \
  -d '{"query":"test","relevance_threshold":0.5}'
```

### Disable Reranking

```bash
# Faster queries, potentially less accurate
curl -X POST http://localhost:8000/rag/query \
  -d '{"query":"test","rerank":false}'
```

### Metadata Filters

```bash
# Filter by file type
curl -X POST http://localhost:8000/rag/query \
  -d '{
    "query":"test",
    "filters":{"file_type":"pdf"}
  }'
```

---

## Troubleshooting

### Check All Services

```bash
#!/bin/bash
echo "=== T.A.R.S. Health Check ==="

echo -n "Backend: "
curl -s http://localhost:8000/health | jq -r '.status'

echo -n "Ollama: "
curl -s http://localhost:11434/api/tags > /dev/null && echo "healthy" || echo "unhealthy"

echo -n "ChromaDB: "
curl -s http://localhost:8001/api/v1/heartbeat > /dev/null && echo "healthy" || echo "unhealthy"

echo -n "RAG Service: "
curl -s http://localhost:8000/rag/health | jq -r '.status'

echo ""
echo "=== Collection Stats ==="
curl -s http://localhost:8000/rag/stats | jq '{documents, chunks, size_mb}'
```

### Enable Debug Logging

```bash
# Edit .env
LOG_LEVEL=DEBUG

# Restart backend
docker-compose restart backend

# View detailed logs
docker-compose logs -f backend | grep -E "(RAG|Embedding|ChromaDB)"
```

---

## Resources

- **API Documentation:** http://localhost:8000/docs
- **Phase 3 Report:** [PHASE3_IMPLEMENTATION_REPORT.md](PHASE3_IMPLEMENTATION_REPORT.md)
- **Phase 2 Report:** [PHASE2_IMPLEMENTATION_REPORT.md](PHASE2_IMPLEMENTATION_REPORT.md)
- **Configuration:** [.env.example](.env.example)
- **GitHub Issues:** [Report bugs](https://github.com/your-org/tars/issues)

---

## Summary

You've now successfully:

✅ Started T.A.R.S. with RAG support
✅ Indexed documents
✅ Executed RAG queries
✅ Used both REST and WebSocket interfaces
✅ Monitored your vector collection

**Next:** Proceed to Phase 4 (Client UI & NAS Monitoring) or customize your RAG pipeline with advanced settings.

---

**Document Version:** 1.0
**Last Updated:** November 7, 2025
**Tested On:** Docker 24.0.7, Python 3.11, Ubuntu 22.04 / Windows 11
