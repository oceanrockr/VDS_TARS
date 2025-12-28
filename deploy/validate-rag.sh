#!/bin/bash
# ==============================================================================
# T.A.R.S. RAG Validation Script
# Version: v1.0.10 (GA) - Phase 22 Validation
# Target: Document Ingestion & Query Verification
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; ((PASS_COUNT++)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; ((WARN_COUNT++)); }
header() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

# Test document
TEST_DOC="/tmp/tars_rag_validation_$(date +%s).txt"
TOKEN=""

# ==============================================================================
# Setup
# ==============================================================================
setup_test_document() {
    cat > "$TEST_DOC" << 'EOF'
# RAG Validation Test Document

## Unique Identifier
VALIDATION_TOKEN: TARS-RAG-VALIDATION-2024

## Known Answers

Question: What is the capital of France?
Answer: The capital of France is Paris.

Question: What color is the sky?
Answer: The sky is blue.

Question: What is 2 + 2?
Answer: 2 + 2 equals 4.

## Technical Details
This document was created for automated RAG validation testing.
It contains specific answers that can be verified programmatically.
EOF
    echo "Test document created: $TEST_DOC"
}

cleanup() {
    rm -f "$TEST_DOC" 2>/dev/null || true
}
trap cleanup EXIT

# ==============================================================================
# RAG Service Health
# ==============================================================================
validate_rag_health() {
    header "RAG Service Health"

    local rag_health
    rag_health=$(curl -s --max-time 10 http://localhost:8000/rag/health 2>/dev/null)

    local rag_status
    rag_status=$(echo "$rag_health" | jq -r '.status' 2>/dev/null)

    if [ "$rag_status" = "healthy" ]; then
        pass "RAG service healthy"
    elif [ "$rag_status" = "degraded" ]; then
        warn "RAG service degraded"
    else
        fail "RAG service: ${rag_status:-no response}"
        return 1
    fi

    local chroma_status
    chroma_status=$(echo "$rag_health" | jq -r '.chromadb_status' 2>/dev/null)
    if [ "$chroma_status" = "healthy" ]; then
        pass "ChromaDB: $chroma_status"
    else
        fail "ChromaDB: ${chroma_status:-unknown}"
    fi

    local embed_status
    embed_status=$(echo "$rag_health" | jq -r '.embedding_model_status' 2>/dev/null)
    if [ "$embed_status" = "healthy" ]; then
        pass "Embedding model: $embed_status"
    else
        fail "Embedding model: ${embed_status:-unknown}"
    fi
}

# ==============================================================================
# Collection Stats
# ==============================================================================
validate_collection() {
    header "Collection Statistics"

    local stats
    stats=$(curl -s --max-time 10 http://localhost:8000/rag/stats 2>/dev/null)

    local total_chunks
    total_chunks=$(echo "$stats" | jq -r '.total_chunks' 2>/dev/null)

    if [ "$total_chunks" != "null" ] && [ "$total_chunks" -gt 0 ] 2>/dev/null; then
        pass "Indexed chunks: $total_chunks"
    else
        warn "No chunks indexed (will test indexing)"
    fi

    local total_docs
    total_docs=$(echo "$stats" | jq -r '.total_documents' 2>/dev/null)
    echo "     Total documents: ${total_docs:-0}"
}

# ==============================================================================
# Authentication
# ==============================================================================
get_auth_token() {
    header "Authentication"

    TOKEN=$(curl -s --max-time 10 -X POST http://localhost:8000/auth/token \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=admin&password=admin" 2>/dev/null | jq -r '.access_token' 2>/dev/null)

    if [ -n "$TOKEN" ] && [ "$TOKEN" != "null" ]; then
        pass "Authentication successful"
        echo "     Token: ${TOKEN:0:30}..."
        return 0
    else
        warn "Authentication failed (will skip index tests)"
        return 1
    fi
}

# ==============================================================================
# Document Indexing
# ==============================================================================
validate_indexing() {
    header "Document Indexing"

    if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
        warn "Skipping indexing (no auth token)"
        return 0
    fi

    setup_test_document

    local index_result
    index_result=$(curl -s --max-time 30 -X POST http://localhost:8000/rag/index \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"file_path\": \"$TEST_DOC\", \"force_reindex\": true}" 2>/dev/null)

    local index_status
    index_status=$(echo "$index_result" | jq -r '.status' 2>/dev/null)

    if [ "$index_status" = "success" ]; then
        local chunks_created
        chunks_created=$(echo "$index_result" | jq -r '.chunks_created' 2>/dev/null)
        pass "Document indexed: $chunks_created chunks"
    else
        fail "Indexing failed: ${index_status:-no response}"
    fi

    local proc_time
    proc_time=$(echo "$index_result" | jq -r '.processing_time_ms' 2>/dev/null)
    echo "     Processing time: ${proc_time:-?}ms"
}

# ==============================================================================
# Search Validation
# ==============================================================================
validate_search() {
    header "Search Functionality"

    local search_result
    search_result=$(curl -s --max-time 30 -X POST http://localhost:8000/rag/search \
        -H "Content-Type: application/json" \
        -d '{"query": "capital of France", "top_k": 5}' 2>/dev/null)

    local total_results
    total_results=$(echo "$search_result" | jq -r '.total_results' 2>/dev/null)

    if [ "$total_results" != "null" ] && [ "$total_results" -gt 0 ] 2>/dev/null; then
        pass "Search returned $total_results results"

        local top_score
        top_score=$(echo "$search_result" | jq -r '.results[0].similarity_score' 2>/dev/null)
        echo "     Top similarity score: $top_score"
    else
        warn "Search returned no results"
    fi

    local search_time
    search_time=$(echo "$search_result" | jq -r '.search_time_ms' 2>/dev/null)
    echo "     Search time: ${search_time:-?}ms"
}

# ==============================================================================
# RAG Query Validation
# ==============================================================================
validate_query() {
    header "RAG Query (Known-Answer Test)"

    echo "Testing query: 'What is the capital of France?'"

    local query_start
    query_start=$(date +%s%3N)

    local query_result
    query_result=$(curl -s --max-time 120 -X POST http://localhost:8000/rag/query \
        -H "Content-Type: application/json" \
        -d '{"query": "What is the capital of France?", "top_k": 3, "include_sources": true}' 2>/dev/null)

    local query_end
    query_end=$(date +%s%3N)
    local query_time=$((query_end - query_start))

    local answer
    answer=$(echo "$query_result" | jq -r '.answer' 2>/dev/null)

    if [ -n "$answer" ] && [ "$answer" != "null" ]; then
        pass "RAG query completed (${query_time}ms)"

        # Check if answer contains expected content
        if echo "$answer" | grep -iq "paris"; then
            pass "Answer contains expected content (Paris)"
        else
            warn "Answer may not be correct"
        fi

        echo "     Answer: ${answer:0:150}..."
    else
        fail "RAG query failed or timed out"
    fi

    local sources_count
    sources_count=$(echo "$query_result" | jq '.sources | length' 2>/dev/null)
    echo "     Sources used: ${sources_count:-0}"

    local retrieval_time
    retrieval_time=$(echo "$query_result" | jq -r '.retrieval_time_ms' 2>/dev/null)
    echo "     Retrieval time: ${retrieval_time:-?}ms"

    local generation_time
    generation_time=$(echo "$query_result" | jq -r '.generation_time_ms' 2>/dev/null)
    echo "     Generation time: ${generation_time:-?}ms"
}

# ==============================================================================
# Streaming Query
# ==============================================================================
validate_streaming() {
    header "Streaming Query"

    local stream_output
    stream_output=$(curl -s --max-time 60 -X POST http://localhost:8000/rag/query/stream \
        -H "Content-Type: application/json" \
        -d '{"query": "What is 2+2?", "top_k": 2}' 2>/dev/null | head -20)

    if echo "$stream_output" | grep -q "rag_token\|token"; then
        pass "Streaming response received"
    else
        warn "Streaming response format unclear"
    fi

    local line_count
    line_count=$(echo "$stream_output" | wc -l)
    echo "     Response lines: $line_count"
}

# ==============================================================================
# ChromaDB Persistence
# ==============================================================================
validate_persistence() {
    header "ChromaDB Persistence"

    # Get current count
    local count_before
    count_before=$(curl -s http://localhost:8001/api/v1/collections/tars_home_documents 2>/dev/null | jq -r '.count' 2>/dev/null)

    if [ "$count_before" != "null" ] && [ -n "$count_before" ]; then
        pass "Collection accessible: $count_before items"
    else
        warn "Could not access collection count"
    fi

    # Check volume exists
    local volume_exists
    volume_exists=$(docker volume ls --format '{{.Name}}' | grep -c "chroma_data" || echo "0")
    if [ "$volume_exists" -gt 0 ]; then
        pass "ChromaDB volume exists"
    else
        warn "ChromaDB volume not found"
    fi
}

# ==============================================================================
# Summary
# ==============================================================================
print_summary() {
    header "RAG Validation Summary"

    echo -e "${GREEN}Passed:${NC}  $PASS_COUNT"
    echo -e "${YELLOW}Warnings:${NC} $WARN_COUNT"
    echo -e "${RED}Failed:${NC}  $FAIL_COUNT"
    echo ""

    if [ "$FAIL_COUNT" -eq 0 ]; then
        echo -e "${GREEN}RAG validation PASSED${NC}"
        exit 0
    else
        echo -e "${RED}RAG validation FAILED${NC}"
        exit 1
    fi
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    echo -e "${CYAN}=================================================================${NC}"
    echo -e "${CYAN} T.A.R.S. RAG Validation - v1.0.10 (Phase 22)${NC}"
    echo -e "${CYAN}=================================================================${NC}"

    validate_rag_health
    validate_collection
    get_auth_token || true
    validate_indexing
    validate_search
    validate_query
    validate_streaming
    validate_persistence
    print_summary
}

main "$@"
