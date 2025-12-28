# T.A.R.S. RAG Validation Guide

**Version:** v1.0.10 (GA)
**Phase:** 22 - RAG Validation
**Target:** Document Ingestion & Query Verification

---

## 1. Prerequisites

Before validating RAG, ensure:

```bash
# All services healthy
curl -s http://localhost:8000/ready | jq '.status'
# Expected: "ready"

# NAS mounted with documents
ls /mnt/llm_docs | head -5

# At least one LLM model available
docker exec tars-home-ollama ollama list
```

---

## 2. RAG Service Health

### Check RAG Component Status

```bash
curl -s http://localhost:8000/rag/health | jq .
```

**Expected:**
```json
{
  "status": "healthy",
  "chromadb_status": "healthy",
  "embedding_model_status": "healthy",
  "total_documents": 0,
  "total_chunks": 0
}
```

### Check Collection Stats

```bash
curl -s http://localhost:8000/rag/stats | jq .
```

---

## 3. Create Known-Answer Test Document

Create a test document with predictable content:

```bash
cat > /tmp/tars_test_document.txt << 'EOF'
# T.A.R.S. Validation Test Document

## Section 1: Project Metadata

Project Name: T.A.R.S. (Temporal Augmented Retrieval System)
Version: 1.0.10
Release Date: 2024-12-27
Deployment Target: Home Network LAN

## Section 2: Technical Specifications

- Primary LLM: Mistral 7B Instruct
- Vector Database: ChromaDB
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Embedding Dimensions: 384
- Chunk Size: 512 tokens
- Chunk Overlap: 50 tokens

## Section 3: Validation Answers

Q1: What is the project name?
A1: The project name is T.A.R.S., which stands for Temporal Augmented Retrieval System.

Q2: What LLM model is used?
A2: The primary LLM model is Mistral 7B Instruct.

Q3: What is the embedding dimension?
A3: The embedding dimension is 384.

## Section 4: Unique Identifiers

VALIDATION_TOKEN: TARS-VAL-7X9K2M
TEST_CHECKSUM: 0xDEADBEEF
VALIDATION_DATE: 2024-12-27
EOF
```

Copy to NAS mount (if writable) or use local path:

```bash
# If NAS is read-only, use tmp
export TEST_DOC_PATH="/tmp/tars_test_document.txt"

# Verify file exists
cat $TEST_DOC_PATH | head -10
```

---

## 4. Document Ingestion Flow

### 4.1 Get Authentication Token

```bash
# Get JWT token (adjust credentials as needed)
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin" | jq -r '.access_token')

echo "Token: ${TOKEN:0:50}..."
```

### 4.2 Index Single Document

```bash
curl -s -X POST http://localhost:8000/rag/index \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"file_path\": \"$TEST_DOC_PATH\"}" | jq .
```

**Expected:**
```json
{
  "document_id": "...",
  "file_name": "tars_test_document.txt",
  "status": "success",
  "chunks_created": 1,
  "processing_time_ms": ...
}
```

### 4.3 Verify Document Indexed

```bash
curl -s http://localhost:8000/rag/stats | jq '.total_chunks, .total_documents'
```

**Expected:** At least 1 document and 1+ chunks

### 4.4 Batch Indexing (Multiple Documents)

```bash
# Create list of files from NAS
NAS_DOCS=$(find /mnt/llm_docs -type f \( -name "*.txt" -o -name "*.md" \) | head -5 | jq -R . | jq -s .)

# Index batch
curl -s -X POST http://localhost:8000/rag/index/batch \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"file_paths\": $NAS_DOCS, \"max_concurrent\": 3}" | jq .
```

---

## 5. Chunking Strategy Validation

### Verify Chunk Size

```bash
# Query for test document chunks
curl -s -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "VALIDATION_TOKEN", "top_k": 5}' | jq '.results[] | {chunk_id, content_length: (.content | length)}'
```

**Expected:** Content length should be ~500-600 characters per chunk

### Verify Chunk Overlap

The test document should have sections that span chunk boundaries. Query for content from Section 2:

```bash
curl -s -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "embedding dimensions 384", "top_k": 3}' | jq '.results[0].content'
```

---

## 6. Embedding Verification

### Test Embedding Generation

```bash
# Search for unique token (should return high similarity)
curl -s -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "TARS-VAL-7X9K2M", "top_k": 1}' | jq '.results[0].similarity_score'
```

**Expected:** Similarity score > 0.7 for exact match

### Test Semantic Search

```bash
# Search for semantic concept (not exact words)
curl -s -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What vector database stores the embeddings?", "top_k": 3}' | jq '.results[] | {file_name, similarity_score}'
```

**Expected:** Test document should appear with similarity > 0.5

---

## 7. ChromaDB Collection Integrity

### List Collections

```bash
curl -s http://localhost:8001/api/v1/collections | jq '.[].name'
```

### Get Collection Count

```bash
COLLECTION="tars_home_documents"
curl -s http://localhost:8001/api/v1/collections/$COLLECTION | jq '.count'
```

### Verify Persistence After Restart

```bash
# Get current count
COUNT_BEFORE=$(curl -s http://localhost:8001/api/v1/collections/tars_home_documents | jq '.count')
echo "Before restart: $COUNT_BEFORE"

# Restart ChromaDB
docker restart tars-home-chromadb
sleep 15

# Get count after restart
COUNT_AFTER=$(curl -s http://localhost:8001/api/v1/collections/tars_home_documents | jq '.count')
echo "After restart: $COUNT_AFTER"

# Verify
if [ "$COUNT_BEFORE" = "$COUNT_AFTER" ]; then
  echo "[PASS] Persistence verified"
else
  echo "[FAIL] Data loss detected"
fi
```

---

## 8. Known-Answer Query Validation

### Test Query 1: Project Name

```bash
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the project name?", "top_k": 3, "include_sources": true}' | jq '{answer: .answer, sources: [.sources[].file_name]}'
```

**Expected Answer:** Should mention "T.A.R.S." and "Temporal Augmented Retrieval System"

### Test Query 2: LLM Model

```bash
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What LLM model does T.A.R.S. use?", "top_k": 3}' | jq '.answer'
```

**Expected Answer:** Should mention "Mistral 7B Instruct"

### Test Query 3: Embedding Dimension

```bash
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the embedding dimension?", "top_k": 3}' | jq '.answer'
```

**Expected Answer:** Should mention "384"

### Test Query 4: No Context (Negative Test)

```bash
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Paris today?", "top_k": 3}' | jq '{answer: .answer, sources_count: (.sources | length)}'
```

**Expected:** Should indicate no relevant information found, sources_count = 0 or low relevance

---

## 9. Cold-Start vs Warm-Query Behavior

### Cold Start (First Query After Restart)

```bash
# Restart backend
docker restart tars-home-backend
sleep 30

# Time first query
START=$(date +%s%3N)
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is T.A.R.S.?", "top_k": 3}' > /dev/null
END=$(date +%s%3N)
echo "Cold query time: $((END - START))ms"
```

### Warm Query (Subsequent Queries)

```bash
# Time second query
START=$(date +%s%3N)
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is T.A.R.S.?", "top_k": 3}' > /dev/null
END=$(date +%s%3N)
echo "Warm query time: $((END - START))ms"
```

**Expected:** Warm queries should be faster (Redis caching of embeddings)

---

## 10. Streaming Query Test

```bash
curl -s -X POST http://localhost:8000/rag/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Describe the T.A.R.S. project", "top_k": 3, "include_sources": true}'
```

**Expected:** Multiple JSON lines:
1. `rag_sources` with source documents
2. Multiple `rag_token` with incremental answer
3. `rag_complete` with timing stats

---

## 11. RAG Validation Script

Save as `deploy/validate-rag.sh`:

```bash
#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "=== T.A.R.S. RAG Validation ==="
echo ""

# 1. RAG Health
echo "--- RAG Service Health ---"
rag_status=$(curl -s http://localhost:8000/rag/health | jq -r '.status')
if [ "$rag_status" = "healthy" ]; then
  pass "RAG service healthy"
else
  fail "RAG service: $rag_status"
fi

# 2. Collection Stats
echo ""
echo "--- Collection Stats ---"
chunks=$(curl -s http://localhost:8000/rag/stats | jq '.total_chunks')
if [ "$chunks" -gt 0 ]; then
  pass "Collection has $chunks chunks indexed"
else
  warn "No chunks indexed yet"
fi

# 3. Create test document
echo ""
echo "--- Test Document ---"
TEST_DOC="/tmp/tars_rag_test.txt"
cat > $TEST_DOC << 'EOF'
RAG Validation Test
UNIQUE_ID: RAG-TEST-$(date +%s)
Answer: The capital of France is Paris.
EOF

# 4. Get token (using default credentials)
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin" 2>/dev/null | jq -r '.access_token' 2>/dev/null)

if [ -n "$TOKEN" ] && [ "$TOKEN" != "null" ]; then
  pass "Authentication successful"

  # 5. Index test document
  echo ""
  echo "--- Indexing Test ---"
  index_result=$(curl -s -X POST http://localhost:8000/rag/index \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"file_path\": \"$TEST_DOC\", \"force_reindex\": true}")

  index_status=$(echo "$index_result" | jq -r '.status')
  if [ "$index_status" = "success" ]; then
    pass "Document indexed successfully"
  else
    warn "Indexing: $index_status"
  fi
else
  warn "Authentication failed - skipping index test"
fi

# 6. Search test
echo ""
echo "--- Search Test ---"
search_results=$(curl -s -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "capital of France", "top_k": 3}')
search_count=$(echo "$search_results" | jq '.total_results')
if [ "$search_count" -gt 0 ]; then
  pass "Search returned $search_count results"
else
  warn "Search returned no results"
fi

# 7. Query test
echo ""
echo "--- RAG Query Test ---"
query_result=$(curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "top_k": 3}' --max-time 60)
answer=$(echo "$query_result" | jq -r '.answer')
if [ -n "$answer" ] && [ "$answer" != "null" ]; then
  pass "RAG query completed"
  echo "     Answer: ${answer:0:100}..."
else
  fail "RAG query failed"
fi

# Cleanup
rm -f $TEST_DOC

echo ""
echo "=== RAG Validation Complete ==="
```

Make executable:
```bash
chmod +x deploy/validate-rag.sh
```

---

## 12. Validation Criteria Summary

| Check | Status | Required |
|-------|--------|----------|
| RAG service healthy | ✅ | Yes |
| ChromaDB connected | ✅ | Yes |
| Embedding model loaded | ✅ | Yes |
| Document indexing works | ✅ | Yes |
| Chunks created correctly | ✅ | Yes |
| Search returns results | ✅ | Yes |
| Known-answer query correct | ✅ | Yes |
| Streaming works | ✅ | Yes |
| Persistence survives restart | ✅ | Yes |
| Cold/warm query difference | ⚠️ | Informational |

---

## 13. Troubleshooting

### No Results from Search

```bash
# Check if any documents indexed
curl -s http://localhost:8000/rag/stats | jq '.'

# Check ChromaDB directly
curl -s http://localhost:8001/api/v1/collections | jq '.'
```

### Slow Query Performance

```bash
# Check embedding model loaded
curl -s http://localhost:8000/rag/health | jq '.embedding_model_status'

# Check Redis cache working
docker exec tars-home-redis redis-cli info | grep used_memory_human
```

### Incorrect Answers

- Verify document was indexed with correct content
- Check similarity scores in search results
- Increase `top_k` to retrieve more context
- Lower `relevance_threshold` temporarily
