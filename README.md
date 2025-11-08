# T.A.R.S. - Temporal Augmented Retrieval System

**Version:** v0.3.0-alpha
**Status:** Active Development - Phase 5 Complete
**Date:** November 7, 2025

## Overview

T.A.R.S. is a distributed, on-premises language model platform that functions as a private LAN-based inference and retrieval hub. Hosted on local hardware with GPU acceleration, it delivers real-time conversational AI and document reasoning to every device on the network while remaining completely offline and self-contained.

## Features

### Core Capabilities
- **Local LLM Inference:** GPU-accelerated inference using Ollama with Mistral 7B, Llama 3.1, and Phi-3 Mini models
- **Advanced RAG Pipeline:** State-of-the-art retrieval with cross-encoder reranking, semantic chunking, and hybrid search
- **WebSocket Streaming:** Real-time bidirectional communication with token-by-token streaming
- **Multi-Format Documents:** Support for PDF, DOCX, TXT, MD, CSV with automatic indexing
- **Privacy-First:** Complete data sovereignty with no cloud dependencies or telemetry
- **React UI:** Modern web interface with real-time chat, document upload, and system metrics

### Advanced Retrieval Features (Phase 5)
- **Cross-Encoder Reranking:** MS MARCO MiniLM model for improved relevance scoring
- **Semantic Chunking:** Dynamic chunk sizing (400-800 tokens) based on content boundaries
- **Hybrid Search:** BM25 keyword + vector similarity fusion for better recall
- **Query Expansion:** LLM-based query reformulation for enhanced retrieval
- **Analytics Tracking:** Comprehensive query and document usage metrics

### System Features
- **Conversation History:** Persistent conversation storage with context recall
- **NAS Integration:** Automatic document monitoring and indexing from network storage
- **System Metrics:** Real-time CPU, GPU, memory, and document statistics
- **Multi-Device Support:** RESTful API accessible from any network device

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Dell XPS 8950 (Central Node)                       │
│  ┌────────────┐  ┌─────────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Ollama    │  │  FastAPI        │  │ ChromaDB │  │  React UI    │  │
│  │  Mistral   │◄─┤  Backend        │◄─┤  Vector  │◄─┤  (Vite)      │  │
│  │  + GPU     │  │  - WebSocket    │  │  Store   │  │  Port 3000   │  │
│  │  Port 11434│  │  - RAG Pipeline │  │  + BM25  │  └──────────────┘  │
│  └────────────┘  │  - Auth (JWT)   │  │  Index   │                     │
│                  │  - Analytics    │  └──────────┘                     │
│                  │  Port 8000      │                                    │
│                  └─────────────────┘                                    │
│                           │                                             │
│                  ┌────────▼────────┐                                    │
│                  │  NAS Watcher    │                                    │
│                  │  Auto-Indexing  │                                    │
│                  └─────────────────┘                                    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ LAN Network (192.168.0.x)
          ┌────────────────┼─────────────────┐
          │                │                 │
    ┌─────▼─────┐    ┌─────▼──────┐   ┌─────▼──────┐
    │  Windows  │    │  Surface   │   │   Android  │
    │  Browser  │    │  Browser   │   │   Browser  │
    │  Client   │    │  Client    │   │   Client   │
    └───────────┘    └────────────┘   └────────────┘
```

## Quick Start

### Prerequisites

- Docker Desktop 4.25+ with GPU support
- Windows 11 or Ubuntu 22.04+
- NVIDIA Driver 535+ (for GPU inference)
- 16 GB RAM minimum (32 GB recommended)
- NVIDIA GTX 1660 or better (RTX 3060+ recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/oceanrockr/VDS_TARS.git
cd VDS_TARS

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env  # Linux/Mac
# notepad .env  # Windows

# Start the Docker stack (backend + Ollama + ChromaDB)
docker-compose up -d

# Verify GPU access
docker exec tars-ollama nvidia-smi

# Pull the Mistral model (first time only)
docker exec tars-ollama ollama pull mistral:7b-instruct

# Start the React UI
cd ui
npm install
npm run dev

# Access the application
# UI: http://localhost:3000
# API Docs: http://localhost:8000/docs
# Health: http://localhost:8000/health
```

### First Steps

1. **Authenticate:** Navigate to `http://localhost:3000` and the app will automatically authenticate
2. **Upload Documents:** Use the document upload panel to index PDF, DOCX, TXT, MD, or CSV files
3. **Start Chatting:** Ask questions about your documents with RAG-enabled responses
4. **Monitor System:** View real-time metrics including GPU usage, document count, and query analytics
5. **Explore API:** Check `http://localhost:8000/docs` for complete API documentation

For detailed setup instructions, see:
- [QUICKSTART.md](QUICKSTART.md) - General quick start guide
- [PHASE3_QUICKSTART.md](PHASE3_QUICKSTART.md) - RAG pipeline setup
- [PHASE4_QUICKSTART.md](PHASE4_QUICKSTART.md) - UI and NAS watcher setup
- [PHASE5_QUICKSTART.md](PHASE5_QUICKSTART.md) - Advanced RAG features

## Project Structure

```
VDS_TARS/
├── backend/                        # FastAPI backend application
│   ├── app/
│   │   ├── api/                   # API routes and endpoints
│   │   │   ├── auth.py            # JWT authentication
│   │   │   ├── websocket.py       # WebSocket chat endpoint
│   │   │   ├── rag.py             # Document indexing and RAG queries
│   │   │   ├── conversation.py    # Conversation history management
│   │   │   ├── metrics.py         # System metrics API
│   │   │   └── analytics.py       # Query analytics API
│   │   ├── core/                  # Core configuration and utilities
│   │   │   ├── config.py          # Environment settings
│   │   │   ├── security.py        # JWT utilities
│   │   │   └── middleware.py      # CORS and error handling
│   │   ├── models/                # Pydantic data models
│   │   │   ├── auth.py            # Authentication models
│   │   │   ├── websocket.py       # WebSocket message models
│   │   │   ├── rag.py             # RAG request/response models
│   │   │   └── conversation.py    # Conversation models
│   │   ├── services/              # Business logic services
│   │   │   ├── ollama_service.py          # LLM inference
│   │   │   ├── connection_manager.py      # WebSocket connections
│   │   │   ├── document_loader.py         # Multi-format parsing
│   │   │   ├── embedding_service.py       # Sentence transformers
│   │   │   ├── chromadb_service.py        # Vector storage
│   │   │   ├── rag_service.py             # RAG orchestration
│   │   │   ├── advanced_reranker.py       # Cross-encoder reranking
│   │   │   ├── semantic_chunker.py        # Dynamic chunking
│   │   │   ├── hybrid_search_service.py   # BM25 + vector fusion
│   │   │   ├── query_expansion.py         # LLM query expansion
│   │   │   ├── analytics_service.py       # Usage tracking
│   │   │   ├── conversation_service.py    # History management
│   │   │   └── nas_watcher.py             # Auto document indexing
│   │   └── main.py                # Application entry point
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # Backend container config
│
├── ui/                            # React web application
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPanel.tsx      # Main chat interface
│   │   │   ├── Sidebar.tsx        # Conversation history
│   │   │   ├── DocumentUpload.tsx # Document indexing UI
│   │   │   └── MetricsDashboard.tsx # System metrics display
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts    # WebSocket state management
│   │   │   └── useAuth.ts         # Authentication hook
│   │   ├── lib/
│   │   │   ├── api.ts             # REST API client
│   │   │   └── websocket.ts       # WebSocket client
│   │   ├── types/
│   │   │   └── index.ts           # TypeScript definitions
│   │   ├── App.tsx                # Main application component
│   │   └── main.tsx               # Entry point
│   ├── package.json               # Node.js dependencies
│   ├── vite.config.ts             # Vite build configuration
│   └── tailwind.config.js         # TailwindCSS configuration
│
├── docker/                        # Container configurations
│   ├── backend/                   # Backend Dockerfile
│   └── ollama/                    # Ollama setup scripts
│
├── docs/                          # Documentation
│   ├── PHASE1_IMPLEMENTATION_REPORT.md  # Infrastructure setup
│   ├── PHASE2_IMPLEMENTATION_REPORT.md  # WebSocket & auth
│   ├── PHASE3_IMPLEMENTATION_REPORT.md  # RAG pipeline
│   ├── PHASE4_IMPLEMENTATION_REPORT.md  # UI & NAS watcher
│   ├── PHASE5_IMPLEMENTATION_REPORT.md  # Advanced RAG
│   └── examples/                  # Code examples
│
├── Reference Docs/                # Development guidelines
│   ├── prd-localllm.md           # Product requirements
│   ├── planning-localllm.md      # Project planning
│   └── rules-localllm.md         # Coding conventions
│
├── docker-compose.yml             # Container orchestration
├── .env.example                   # Environment template
├── QUICKSTART.md                  # Quick start guide
├── PHASE3_QUICKSTART.md           # RAG setup guide
├── PHASE4_QUICKSTART.md           # UI setup guide
└── PHASE5_QUICKSTART.md           # Advanced RAG guide
```

## Development Status

### Phase 1: Foundation & Infrastructure (Weeks 1-2) - COMPLETE ✅
- [x] Repository scaffolding
- [x] Docker Compose configuration (Ollama, ChromaDB, Backend)
- [x] GPU passthrough validation
- [x] Network configuration
- [x] Health check endpoints
- [x] Development environment setup

### Phase 2: WebSocket Gateway & Authentication (Weeks 3-4) - COMPLETE ✅
- [x] JWT authentication system (24h expiry, refresh tokens)
- [x] WebSocket endpoint at `/ws/chat`
- [x] Real-time token streaming from Ollama
- [x] Connection manager (10+ concurrent sessions)
- [x] Heartbeat ping/pong (30s intervals)
- [x] Reconnection & retry logic (exponential backoff)
- [x] Comprehensive testing (107 tests, 88% coverage)
- [x] Performance validation (<100ms inter-token latency)

### Phase 3: Document Indexing & RAG (Weeks 5-6) - COMPLETE ✅
- [x] Multi-format document loading (PDF, DOCX, TXT, MD, CSV)
- [x] Sentence-transformers embedding pipeline (all-MiniLM-L6-v2)
- [x] ChromaDB vector storage with metadata
- [x] RAG service with context retrieval and reranking
- [x] REST API endpoints for document upload and queries
- [x] WebSocket integration for RAG-enabled streaming
- [x] Citation tracking and source attribution

### Phase 4: Client UI & NAS Monitoring (Weeks 7-8) - COMPLETE ✅
- [x] React web application with Vite + TypeScript + TailwindCSS
- [x] Real-time chat interface with WebSocket streaming
- [x] Document upload UI with drag-and-drop support
- [x] Conversation history management with ChromaDB persistence
- [x] NAS watcher service for automatic document indexing
- [x] System metrics dashboard (CPU, GPU, memory, documents)
- [x] REST API for conversations and metrics

### Phase 5: Advanced RAG & Semantic Chunking (Weeks 9-10) - COMPLETE ✅
- [x] Cross-encoder reranking (MS MARCO MiniLM-L-6-v2)
- [x] Semantic chunking with dynamic sizing (400-800 tokens)
- [x] Hybrid search (BM25 + vector similarity fusion)
- [x] Query expansion with LLM-based reformulation
- [x] Analytics service for query and document tracking
- [x] Analytics REST API with comprehensive metrics
- [x] Enhanced RAG pipeline integration
- [x] Performance benchmarking (+20-25% MRR improvement)

### Phase 6: Testing, Optimization & Production (Weeks 11-12) - PLANNED
- [ ] Comprehensive test suite for all components
- [ ] Performance optimization and benchmarking
- [ ] Production deployment guides
- [ ] Security hardening and audit
- [ ] Documentation completion
- [ ] User acceptance testing

## Documentation

### Implementation Reports
- [Phase 1 Implementation Report](PHASE1_IMPLEMENTATION_REPORT.md) - Infrastructure & Docker setup ✅
- [Phase 2 Implementation Report](PHASE2_IMPLEMENTATION_REPORT.md) - WebSocket & authentication ✅
- [Phase 3 Implementation Report](PHASE3_IMPLEMENTATION_REPORT.md) - RAG pipeline & document indexing ✅
- [Phase 4 Implementation Report](PHASE4_IMPLEMENTATION_REPORT.md) - React UI & NAS monitoring ✅
- [Phase 5 Implementation Report](PHASE5_IMPLEMENTATION_REPORT.md) - Advanced RAG features ✅

### Quick Start Guides
- [General Quick Start](QUICKSTART.md) - Getting started with T.A.R.S.
- [Phase 3 Quick Start](PHASE3_QUICKSTART.md) - RAG pipeline setup (10-15 min)
- [Phase 4 Quick Start](PHASE4_QUICKSTART.md) - UI and NAS watcher setup (15-20 min)
- [Phase 5 Quick Start](PHASE5_QUICKSTART.md) - Advanced RAG configuration (10-15 min)

### API & Examples
- [API Documentation](http://localhost:8000/docs) - Interactive Swagger UI
- [WebSocket Client Example](docs/examples/websocket_client_example.py) - Python WebSocket client
- [Reference Documentation](Reference%20Docs/) - Development guidelines and conventions

## Technology Stack

### Backend
- **Framework:** FastAPI 0.104.1 (Python 3.11+)
- **LLM Engine:** Ollama with Mistral 7B Instruct
- **Vector Database:** ChromaDB 0.4.18
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Reranking:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **Search:** BM25 (rank-bm25) + vector similarity
- **Authentication:** JWT (python-jose)
- **WebSocket:** FastAPI native WebSocket support

### Frontend
- **Framework:** React 18.2.0 with TypeScript 5.2.2
- **Build Tool:** Vite 5.0.8
- **Styling:** TailwindCSS 3.3.6
- **HTTP Client:** Axios 1.6.2
- **Charts:** Recharts 2.10.3
- **Markdown:** react-markdown 9.0.1

### Infrastructure
- **Containerization:** Docker & Docker Compose
- **GPU:** NVIDIA CUDA support (RTX 3060 recommended)
- **Storage:** Persistent volumes for ChromaDB and logs

## Performance Metrics

Based on Phase 5 implementation testing:

| Metric | Value |
|--------|-------|
| **Query Latency** | ~2.2s (balanced config) |
| **Retrieval Time** | ~160ms (hybrid + reranking) |
| **Generation Time** | ~1.8s (Mistral 7B on GPU) |
| **MRR Improvement** | +20-25% (vs. baseline RAG) |
| **Cross-Encoder Latency** | ~45ms (GPU) / ~180ms (CPU) |
| **Semantic Chunking** | 2.7x slower indexing, better coherence |
| **Concurrent Connections** | 10+ WebSocket sessions |
| **Token Throughput** | 20+ tokens/sec (GPU accelerated) |

## Contributing

This project follows the VDS RiPIT Agent Coding Workflow v2.9 conventions. Please see the [Reference Docs](Reference%20Docs/) for development guidelines.

### Development Setup
1. Follow the installation instructions above
2. Install development dependencies: `pip install -r backend/requirements.txt`
3. Run tests: `pytest backend/tests/`
4. Format code: `black backend/` and `prettier ui/src/`

## Security

T.A.R.S. is designed with security and privacy as core principles:
- **Data Sovereignty:** All data remains local - no cloud dependencies
- **Encryption:** TLS 1.3 for network traffic (production deployment)
- **Authentication:** JWT-based authentication with 24h token expiry
- **Privacy:** No telemetry or usage tracking - analytics stored locally only
- **Isolation:** Docker containerization for service isolation
- **Dependencies:** Regular security audits and dependency scanning

## License

[License information to be added]

## Acknowledgments

Built with:
- [Ollama](https://ollama.ai) - Local LLM inference
- [FastAPI](https://fastapi.tiangolo.com) - Modern Python web framework
- [ChromaDB](https://www.trychroma.com) - Vector database
- [LangChain](https://langchain.com) - RAG orchestration
- [React](https://react.dev) - UI framework
- [Electron](https://electronjs.org) - Desktop application framework

## Contact

Project maintained by Veleron Dev Studios

---

## Project Statistics

- **Total Lines of Code:** ~9,920 lines
- **Backend Code:** ~7,000 lines (Python)
- **Frontend Code:** ~2,900 lines (TypeScript/React)
- **Services Implemented:** 13 core services
- **API Endpoints:** 25+ REST endpoints + WebSocket
- **Development Time:** 10 weeks (5 phases complete)
- **Test Coverage:** 88% (Phase 2 comprehensive testing)

---

**Last Updated:** November 7, 2025
**Documentation Version:** v0.3.0-alpha
**Repository:** [https://github.com/oceanrockr/VDS_TARS](https://github.com/oceanrockr/VDS_TARS)
