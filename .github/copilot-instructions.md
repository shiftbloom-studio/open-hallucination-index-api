# Open Hallucination Index - Copilot Instructions

OHI is a middleware platform for detecting LLM hallucinations via claim decomposition and multi-source verification.

---

## 🧭 Monorepo Structure

| Path | Role | Tech |
|------|------|------|
| `src/api/` | Verification API | Python 3.14+, FastAPI, Hexagonal |
| `src/frontend/` | UI | Next.js 16, React 19, Tailwind v4 |
| `src/ohi-mcp-server/` | External knowledge | Node.js, MCP protocol |
| `gui_ingestion_app/` | Wikipedia ingestion | Python, GPU embeddings |
| `gui_benchmark_app/` | Evaluation suite | Python, statistical tests |
| `docker/compose/` | Full stack | Neo4j, Qdrant, Redis, vLLM |

---

## 🏗️ API Hexagonal Architecture (STRICT)

Under `src/api/`:
```
interfaces/   → ABCs (ports): stores.py, verification.py, decomposition.py, etc.
adapters/     → Concrete impls: neo4j.py, qdrant.py, redis_*.py, openai.py
pipeline/     → Domain services: oracle.py, decomposer.py, collector.py, scorer.py
services/     → Use cases: verify.py (VerifyTextUseCase), track.py
config/       → DI wiring: dependencies.py (lifespan_manager), settings.py
models/       → Domain entities: entities.py, results.py, track.py
```

**Rules:**
- `pipeline/` and `models/` must NOT import from `adapters/` or `config/`
- Adapters implement interfaces from `interfaces/`
- DI is wired in `config/dependencies.py` via FastAPI lifespan
- Use `typing.Annotated` for FastAPI dependency injection

---

## 🔄 Verification Pipeline Flow

`VerifyTextUseCase` in `services/verify.py` orchestrates:
1. **Cache check** → `RedisCacheAdapter` (SHA-256 hash key)
2. **Claim decomposition** → `LLMClaimDecomposer` (LLM extraction with fallback)
3. **Claim routing** → `ClaimRouter` (domain classification + source priorities)
4. **Evidence collection** → `AdaptiveEvidenceCollector` (local tier → MCP tier)
5. **Verification** → `HybridVerificationOracle` (strategy-based decision)
6. **Scoring** → `WeightedScorer` (trust score with confidence)
7. **Trace storage** → `RedisTraceAdapter` (12h TTL for knowledge-track)

**Strategies** (`VerificationStrategy` enum in `interfaces/verification.py`):
- `adaptive` (default): tiered local-first with early-exit
- `hybrid`: graph + vector parallel
- `cascading`: graph first, vector fallback
- `mcp_enhanced`: MCP first, local fallback

---

## 🔧 Development Commands

```bash
# API (from src/api/)
pip install -e ".[dev]"   # Install with dev deps
ohi-server                # Run API server

# Frontend (from src/frontend/)
npm install && npm run dev

# Docker full stack (from repo root)
docker compose -f docker/compose/docker-compose.yml up -d

# Rebuild after code changes to ohi-api or ohi-mcp-server
docker compose -f docker/compose/docker-compose.yml up -d --build ohi-api ohi-mcp-server

# Ingestion (from gui_ingestion_app/ingestion/)
python -m ingestion --limit 1000

# Benchmark
python -m benchmark  # or use gui_benchmark_app/launch_gui.bat
```

**⚠️ After changing `src/api/` or `src/ohi-mcp-server/`**: Always rebuild and restart with `docker compose up -d --build`

---

## 🌐 Nginx Configuration (CRITICAL)

The nginx config at `docker/nginx/nginx.conf` handles all routing and is **complex**:

- **Upstreams**: `ohi_frontend` (port 3000), `ohi_api` (port 8080)
- **Rate limiting**: 60 req/min per IP (`limit_req_zone`)
- **Cloudflare integration**: mTLS origin pull, `CF-Connecting-IP` header, origin lock
- **Timeouts**: 240s connect, 300s send/read (required for slow LLM responses)
- **Local dev**: HTTP on port 80 for `localhost`/`127.0.0.1`
- **Tunnel ingress**: Port 8080 for Cloudflare tunnel (Docker-internal)

**When modifying network behavior**, check:
1. Upstream definitions and health checks
2. Proxy timeouts (API can take minutes)
3. Cloudflare mTLS requirements (`ssl_verify_client on`)
4. Rate limit zones

---

## ⚙️ Key Environment Variables

| Prefix | Purpose | Example |
|--------|---------|---------|
| `LLM_*` | vLLM/OpenAI-compatible | `LLM_BASE_URL=http://localhost:8000/v1` |
| `NEO4J_*` | Graph DB | `NEO4J_URI=bolt://localhost:7687` |
| `QDRANT_*` | Vector DB | `QDRANT_HOST=localhost`, `QDRANT_COLLECTION_NAME=wikipedia_hybrid` |
| `REDIS_*` | Cache | `REDIS_ENABLED=true`, `REDIS_HOST=localhost` |
| `VERIFY_*` | Pipeline tuning | `VERIFY_DEFAULT_STRATEGY=adaptive` |
| `API_*` | Server config | `API_PORT=8080`, `API_API_KEY=` (empty = no auth) |

See `src/api/config/settings.py` for all options with defaults.

---

## 📊 Data Store Patterns

**Neo4j** (`adapters/neo4j.py`):
- 27 relationship types (ingestion-aligned): `LINKS_TO`, `IN_CATEGORY`, `MENTIONS`, person/org/geo relations
- Specialized queries: `query_person_facts()`, `query_organization_facts()`, `query_geographic_facts()`

**Qdrant** (`adapters/qdrant.py`):
- Collection: `wikipedia_hybrid` (384-dim, all-MiniLM-L12-v2)
- Hybrid search: dense + sparse (BM25) vectors

**Redis**:
- Result cache: SHA-256 hash of input text (16 chars)
- Claim cache: SHA-256 hash of claim text (24 chars), long TTL
- Trace storage: knowledge-track with 12h TTL

---

## ⚛️ Frontend Patterns

- **App Router first**: Server Components by default
- **Client islands**: Use `"use client"` only for interactivity (forms, charts)
- **Data fetching**: React Query + Server Actions
- **API proxy**: `/api/ohi/*` routes to backend
- **UI**: shadcn/ui components, Tailwind v4, Zod validation

---

## ⚠️ Critical Constraints

1. **LLM is local** (vLLM) - do not assume OpenAI API availability
2. **Extreme latencies**: ohi-api and vLLM can take **multiple minutes** per request - never set short timeouts
3. **Slow container startup**: vLLM takes 30-60s to load models; ohi-api waits for vLLM health - don't assume services are ready immediately after `docker compose up`
4. **MCP evidence persists** to Neo4j + Qdrant when `VERIFY_PERSIST_MCP_EVIDENCE=true`
5. **Redis optional** but disables caching and knowledge-track when unavailable
6. **API key auth** disabled when `API_API_KEY` is empty string
7. **GPU required** for vLLM and fast embeddings in ingestion
8. **Always rebuild Docker** after changing `src/api/` or `src/ohi-mcp-server/`
9. **For architectural changes**: also review `docker/compose/docker-compose.yml` and relevant `Dockerfile`s

---

## ✅ Coding Standards

- Python 3.14+ features allowed (match statements, type hints)
- Strict typing everywhere; `mypy --strict` compatible
- Async for all I/O-bound operations
- Linting: `ruff check` + `ruff format`
- Docstrings: module-level explanation + Args/Returns for public APIs
- **No summary/update markdown files** — do not create new `.md` files to document changes; only update existing READMEs for major feature additions

---

## 🎯 Efficiency Guidelines

- **Stay focused** — complete tasks with minimal tangents
- **Batch related edits** — use multi-file edits when possible
- **Don't over-verify** — services take time to start; trust healthchecks
- **Read before writing** — understand existing patterns before adding code