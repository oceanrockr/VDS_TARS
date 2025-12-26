# T.A.R.S. Phase 5 Quick Start Guide
## Advanced RAG & Semantic Chunking

**Version:** v0.3.0-alpha
**Last Updated:** November 7, 2025

---

## Overview

This guide will help you quickly set up and test T.A.R.S. Phase 5's advanced RAG features:
- ✅ Cross-encoder reranking
- ✅ Semantic chunking
- ✅ Hybrid search (BM25 + vector)
- ✅ Query expansion
- ✅ Analytics tracking

**Estimated Setup Time:** 10-15 minutes

---

## Prerequisites

Ensure you have completed Phases 1-4:
- ✅ Docker and Docker Compose installed
- ✅ Ollama running with Mistral model
- ✅ ChromaDB container running
- ✅ Backend service operational
- ✅ At least one document indexed

---

## Quick Setup

### 1. Update Dependencies

```bash
cd backend

# Install new Phase 5 dependencies
pip install transformers==4.36.0 rank-bm25==0.2.2 langchain-text-splitters==0.0.1

# Or use requirements.txt
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` or create from example:

```bash
cp .env.example .env
nano .env
```

Add Phase 5 settings:

```bash
# Phase 5: Advanced RAG Configuration

# Semantic Chunking
USE_SEMANTIC_CHUNKING=true
SEMANTIC_CHUNK_MIN=400
SEMANTIC_CHUNK_MAX=800

# Advanced Reranking
USE_ADVANCED_RERANKING=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_K=10
RERANK_WEIGHT=0.35

# Hybrid Search
USE_HYBRID_SEARCH=true
HYBRID_ALPHA=0.3

# Query Expansion (disabled by default - adds latency)
USE_QUERY_EXPANSION=false

# Analytics
ANALYTICS_ENABLED=true
ANALYTICS_LOG_PATH=./logs/analytics.log
```

### 3. Create Logs Directory

```bash
mkdir -p logs
chmod 755 logs
```

### 4. Restart Backend

```bash
# If using Docker Compose
docker-compose restart backend

# If running directly
python -m app.main
```

**Watch for startup messages:**
```
INFO - Starting T.A.R.S. Backend v0.3.0-alpha
INFO - Phase 5: Advanced RAG & Semantic Chunking
INFO - Loading cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
INFO - Cross-encoder model loaded successfully (device: cuda, time: 850.3ms)
INFO - RAG components initialized successfully (with Phase 5 enhancements)
```

---

## Quick Test

### Test 1: Verify Phase 5 Services

```bash
# Check analytics health
curl http://localhost:8000/analytics/health

# Expected response:
{
  "status": "healthy",
  "enable_logging": true,
  "total_queries": 0,
  "total_errors": 0,
  "tracked_documents": 0
}
```

### Test 2: Index a Document with Semantic Chunking

```bash
# Get authentication token
TOKEN=$(curl -X POST http://localhost:8000/auth/authenticate \
  -H "Content-Type: application/json" \
  -d '{"client_id": "test_client"}' | jq -r '.access_token')

# Index a document
curl -X POST http://localhost:8000/rag/upload \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "force_reindex": false
  }'

# Expected response:
{
  "document_id": "doc_abc123",
  "file_name": "document.pdf",
  "status": "success",
  "chunks_created": 12,
  "processing_time_ms": 1250.5,
  "message": null
}
```

**Note:** Check backend logs for semantic chunking confirmation:
```
DEBUG - Using semantic chunking: 12 chunks created
DEBUG - Added 12 chunks to BM25 index
```

### Test 3: Query with Advanced RAG

```bash
# Perform RAG query
curl -X POST http://localhost:8000/rag/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is semantic chunking?",
    "top_k": 5,
    "rerank": true,
    "include_sources": true
  }'

# Expected response includes:
{
  "query": "What is semantic chunking?",
  "answer": "Semantic chunking is...",
  "sources": [
    {
      "chunk_id": "doc_abc123_chunk_5",
      "file_name": "document.pdf",
      "similarity_score": 0.9201,
      "excerpt": "...",
      "metadata": {
        "hybrid_search": true,
        "vector_score": 0.8542,
        "bm25_score": 0.7823,
        "reranked": true,
        "cross_encoder_score": 0.9534
      }
    }
  ],
  "retrieval_time_ms": 163.5,
  "generation_time_ms": 1850.2,
  "total_time_ms": 2013.7
}
```

**Check logs for advanced features:**
```
DEBUG - Expanded query to 1 variants
DEBUG - Applied hybrid search: 10 results
DEBUG - Applied advanced reranking: 5 results
INFO - Retrieved 5 relevant chunks (expansion: false, hybrid: true, rerank: true, time: 163.5ms)
```

### Test 4: View Analytics

```bash
# Get query statistics
curl http://localhost:8000/analytics/query-stats \
  -H "Authorization: Bearer $TOKEN"

# Expected response:
{
  "total_queries": 1,
  "successful_queries": 1,
  "failed_queries": 0,
  "success_rate": 1.0,
  "avg_retrieval_time_ms": 163.5,
  "avg_generation_time_ms": 1850.2,
  "avg_total_time_ms": 2013.7,
  "avg_sources_count": 5.0,
  "avg_relevance_score": 0.8542,
  "reranking_usage_count": 1,
  "reranking_usage_rate": 1.0,
  "hybrid_search_usage_count": 1,
  "hybrid_search_usage_rate": 1.0,
  "query_expansion_usage_count": 0,
  "query_expansion_usage_rate": 0.0
}
```

---

## Feature Demos

### Demo 1: Compare Standard vs Semantic Chunking

**Standard Chunking (Fixed 512 tokens):**
```bash
# Temporarily disable semantic chunking
export USE_SEMANTIC_CHUNKING=false

# Index document
curl -X POST http://localhost:8000/rag/upload \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"file_path": "/path/to/doc.pdf", "force_reindex": true}'

# Note chunk count and processing time
```

**Semantic Chunking (Dynamic 400-800 tokens):**
```bash
# Re-enable semantic chunking
export USE_SEMANTIC_CHUNKING=true

# Re-index same document
curl -X POST http://localhost:8000/rag/upload \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"file_path": "/path/to/doc.pdf", "force_reindex": true}'

# Compare chunk count and coherence
```

**Expected Differences:**
- Semantic: Fewer chunks (better boundaries)
- Semantic: Metadata includes `section_title`, `chunk_type`
- Semantic: ~2.7x slower indexing (acceptable trade-off)

### Demo 2: Test Hybrid Search Impact

**Vector Only:**
```bash
# Temporarily disable hybrid search
export USE_HYBRID_SEARCH=false

# Query
curl -X POST http://localhost:8000/rag/query \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "machine learning algorithms", "top_k": 5}'

# Note relevance scores and results
```

**Hybrid (Vector + BM25):**
```bash
# Re-enable hybrid search
export USE_HYBRID_SEARCH=true

# Same query
curl -X POST http://localhost:8000/rag/query \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "machine learning algorithms", "top_k": 5}'

# Compare results - should see improved keyword matching
```

**Expected Improvement:**
- Better matching for exact phrases
- Improved recall for keyword-heavy queries
- Metadata shows `hybrid_search: true` and component scores

### Demo 3: Advanced Reranking

**Baseline (Vector similarity only):**
```bash
export USE_ADVANCED_RERANKING=false

curl -X POST http://localhost:8000/rag/query \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "How does cross-encoder reranking work?", "top_k": 5}'
```

**With Cross-Encoder:**
```bash
export USE_ADVANCED_RERANKING=true

curl -X POST http://localhost:8000/rag/query \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "How does cross-encoder reranking work?", "top_k": 5}'

# Results should have better relevance ordering
# Metadata includes cross_encoder_score
```

**Expected Improvement:**
- Better semantic matching (query-document pairs)
- More relevant top results
- Slight latency increase (~45ms GPU, ~180ms CPU)

### Demo 4: Query Expansion (Advanced)

**Warning:** Query expansion adds significant latency (~1.2s per expansion).

```bash
# Enable query expansion
export USE_QUERY_EXPANSION=true
export QUERY_EXPANSION_MAX=3

# Restart backend
docker-compose restart backend

# Test query
curl -X POST http://localhost:8000/rag/query \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "ML", "top_k": 5}'

# Check logs for expanded variants
```

**Expected Log Output:**
```
DEBUG - Expanded query to 3 variants
INFO - Generated 3 expansions (strategy: general, time: 1200ms)
DEBUG - Queries: ["ML", "machine learning", "ML algorithms", "artificial intelligence models"]
```

**When to Use:**
- Complex technical queries
- Abbreviations (ML, RAG, NLP)
- When recall is critical and latency acceptable

---

## Configuration Tuning

### For Maximum Accuracy (Latency Tolerant)

```bash
# Enable all features
USE_SEMANTIC_CHUNKING=true
USE_ADVANCED_RERANKING=true
USE_HYBRID_SEARCH=true
USE_QUERY_EXPANSION=true
QUERY_EXPANSION_MAX=5

# Aggressive reranking
RERANK_TOP_K=15
RERANK_WEIGHT=0.5

# Balanced hybrid fusion
HYBRID_ALPHA=0.4
```

**Expected Latency:** ~4-5 seconds per query

### For Maximum Speed (Accuracy Tolerant)

```bash
# Disable expensive features
USE_SEMANTIC_CHUNKING=false
USE_ADVANCED_RERANKING=false
USE_HYBRID_SEARCH=false
USE_QUERY_EXPANSION=false

# Use standard RAG
RAG_TOP_K=5
RAG_RERANK_ENABLED=false
```

**Expected Latency:** ~2 seconds per query

### Balanced (Recommended)

```bash
# Use semantic chunking + hybrid + reranking
USE_SEMANTIC_CHUNKING=true
USE_ADVANCED_RERANKING=true
USE_HYBRID_SEARCH=true
USE_QUERY_EXPANSION=false  # Disable expensive expansion

# Conservative settings
RERANK_TOP_K=10
RERANK_WEIGHT=0.35
HYBRID_ALPHA=0.3
```

**Expected Latency:** ~2.2 seconds per query
**Expected Improvement:** +20-25% MRR over baseline

---

## Analytics Examples

### Export Analytics to CSV

```bash
curl -X POST http://localhost:8000/analytics/export \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"output_path": "/tmp/tars_analytics.csv"}'

# View exported data
head -n 20 /tmp/tars_analytics.csv
```

### Get Top Documents

```bash
curl http://localhost:8000/analytics/document-popularity?top_n=5 \
  -H "Authorization: Bearer $TOKEN"
```

### Query Patterns

```bash
curl http://localhost:8000/analytics/query-patterns \
  -H "Authorization: Bearer $TOKEN"

# Shows:
# - Common query words
# - Avg query length
# - Queries by hour of day
```

### Comprehensive Summary

```bash
curl http://localhost:8000/analytics/summary \
  -H "Authorization: Bearer $TOKEN" | jq '.'

# Returns all analytics in one response:
# - Query stats
# - Top documents
# - Query patterns
```

---

## Troubleshooting

### Issue: Cross-Encoder Model Not Loading

**Symptoms:**
```
WARNING - Advanced reranker not available, using fallback
```

**Solutions:**

1. **Check GPU availability:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. **Install transformers correctly:**
```bash
pip install --upgrade transformers==4.36.0 torch
```

3. **Download model manually:**
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

4. **Use CPU mode:**
```bash
export EMBED_DEVICE=cpu
```

### Issue: BM25 Index Errors

**Symptoms:**
```
ERROR - No BM25 results, returning vector results only
```

**Solution:**

Rebuild BM25 index:
```python
from app.services.hybrid_search_service import hybrid_search_service
from app.services.chromadb_service import chromadb_service

# Get all chunks from ChromaDB
results = await chromadb_service.query(
    query_embedding=[0.0]*384,
    top_k=10000
)

# Build BM25 index
hybrid_search_service.build_bm25_index(chunks)
```

### Issue: Analytics Log File Permission Denied

**Symptoms:**
```
ERROR - Failed to write analytics log: Permission denied
```

**Solution:**

```bash
# Create logs directory with correct permissions
mkdir -p logs
chmod 755 logs
touch logs/analytics.log
chmod 644 logs/analytics.log
```

### Issue: Slow Query Expansion

**Symptoms:**
Queries taking 5-10 seconds

**Solution:**

Query expansion is expensive. Disable or use caching:

```bash
# Disable query expansion
USE_QUERY_EXPANSION=false

# Or increase cache size
# In query_expansion.py, set cache_size=5000
```

---

## Performance Tips

### 1. GPU Acceleration

Ensure CUDA is available for:
- Cross-encoder reranking (5x faster)
- Embedding generation (3x faster)

```bash
# Check CUDA
nvidia-smi

# Set GPU device
export EMBED_DEVICE=cuda
```

### 2. Batch Document Uploads

Instead of indexing one document at a time:

```bash
# Use batch endpoint
curl -X POST http://localhost:8000/rag/batch-upload \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "file_paths": ["/path/doc1.pdf", "/path/doc2.pdf"],
    "force_reindex": false
  }'
```

### 3. Pre-warm Caches

On startup, run a few dummy queries to warm up:
- Embedding model cache
- Cross-encoder model cache
- BM25 index

### 4. Tune Top-K Values

```bash
# For reranking, retrieve more initially
RAG_TOP_K=5
RERANK_TOP_K=15  # Rerank top 15, return top 5

# This improves reranking effectiveness
```

---

## Next Steps

1. **Explore Analytics Dashboard** (if UI implemented)
   - View real-time query metrics
   - Analyze document popularity
   - Monitor system performance

2. **Fine-Tune Configuration**
   - Adjust RERANK_WEIGHT based on your use case
   - Optimize HYBRID_ALPHA for your document types
   - Test QUERY_EXPANSION on complex queries

3. **Scale to Production** (Phase 6)
   - Set up Kubernetes deployment
   - Configure Prometheus metrics
   - Implement Redis caching
   - Add PostgreSQL for analytics persistence

4. **Advanced Usage**
   - Custom reranking models
   - Domain-specific chunking strategies
   - Multi-lingual support
   - Document-specific retrieval strategies

---

## Resources

- **Full Documentation:** [PHASE5_IMPLEMENTATION_REPORT.md](PHASE5_IMPLEMENTATION_REPORT.md)
- **API Reference:** `http://localhost:8000/docs` (Swagger UI)
- **Configuration:** [backend/app/core/config.py](backend/app/core/config.py)
- **Environment:** [.env.example](.env.example)

---

## Support

For issues or questions:
1. Check backend logs: `docker-compose logs backend`
2. Review analytics: `curl http://localhost:8000/analytics/summary`
3. Verify health: `curl http://localhost:8000/health`
4. Consult implementation report for detailed architecture

---

**Quick Start Guide** | T.A.R.S. Phase 5 | v0.3.0-alpha
