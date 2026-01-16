# GitHub Issues to Create

This document contains all the issues that need to be created based on the task list. Each issue is formatted and ready to be created.

---

## Issue 1: Optimize Ingestion Upsert Operations

**Title:** `[Enhancement]: Optimize Ingestion Upsert Operations`

**Labels:** `enhancement`, `performance`, `ingestion`

**Category:** Performance

**Description:**

### Problem Statement
The current ingestion pipeline may perform redundant upsert operations that could be optimized for better performance when processing large Wikipedia dumps.

### Proposed Solution
Optimize the upsert operations in the ingestion pipeline to:
- Batch upsert operations more efficiently
- Reduce redundant database calls
- Implement intelligent conflict resolution
- Add performance metrics to track improvements

### Implementation Areas
- `src/ingestion/` - Core ingestion logic
- Neo4j adapter upsert operations
- Qdrant vector store upsert operations

### Success Criteria
- Measurable improvement in ingestion throughput
- Reduced database load
- Better handling of duplicate entries

---

## Issue 2: Implement Ingestion LLM Gardener

**Title:** `[Feature]: Implement Ingestion LLM Gardener for Data Quality`

**Labels:** `enhancement`, `ingestion`, `llm`

**Category:** Verification Logic

**Description:**

### Problem Statement
Ingested data may contain inconsistencies, formatting issues, or quality problems that should be detected and corrected automatically.

### Proposed Solution
Create an LLM-based "gardener" service that:
- Reviews ingested content for quality issues
- Identifies and corrects formatting problems
- Flags low-quality or suspicious content
- Standardizes entity representations
- Improves claim extraction quality

### Implementation Areas
- New module: `src/ingestion/gardener/`
- Integration with existing ingestion pipeline
- Quality metrics and reporting

### Success Criteria
- Automated quality checks on ingested data
- Measurable improvement in data quality scores
- Reduced false positives in verification

---

## Issue 3: Support Multiple Knowledge Bases

**Title:** `[Feature]: Multiple Knowledge Base Support`

**Labels:** `enhancement`, `architecture`, `knowledge-sources`

**Category:** API / Endpoints

**Description:**

### Problem Statement
Currently, OHI supports a single knowledge base configuration. Users may want to configure multiple knowledge bases for different domains or use cases.

### Proposed Solution
Enable multi-tenancy for knowledge bases:
- Allow configuration of multiple Neo4j/Qdrant instances
- Support knowledge base selection per verification request
- Implement knowledge base routing logic
- Add admin dashboard controls for knowledge base management

### Implementation Areas
- `src/api/src/open_hallucination_index/adapters/` - Database adapters
- Configuration management
- API endpoints to select/list knowledge bases
- Frontend UI for knowledge base selection

### Success Criteria
- Users can configure multiple knowledge bases
- Verification requests can target specific knowledge bases
- Admin dashboard shows all configured knowledge bases

---

## Issue 4: Ensure Target Evidence Count Per Claim

**Title:** `[Enhancement]: Ensure Target Evidence Count Per Claim`

**Labels:** `enhancement`, `verification`

**Category:** Verification Logic

**Description:**

### Problem Statement
When the target evidence count is set (e.g., 8 pieces of evidence), but vector search only returns 3, the system should attempt to find the remaining 5 using MCP or other sources to meet the target.

### Proposed Solution
Implement adaptive evidence collection strategy:
- Track evidence count per claim
- If local sources (Neo4j + Qdrant) don't meet target, query MCP sources
- Ensure minimum evidence threshold per claim
- Add configuration for min/max evidence per claim

### Implementation Areas
- `src/api/src/open_hallucination_index/adapters/evidence_collector.py`
- `AdaptiveEvidenceCollector` class
- Evidence collection strategy logic

### Success Criteria
- Each claim receives target evidence count
- MCP sources are queried when local sources insufficient
- Configurable evidence targets per claim

---

## Issue 5: Remove Wikipedia MCP for Redundancy

**Title:** `[Cleanup]: Remove Wikipedia MCP Source (Redundant)`

**Labels:** `cleanup`, `mcp`, `optimization`

**Category:** Knowledge Sources (MCP)

**Description:**

### Problem Statement
Wikipedia data is already ingested into local Neo4j and Qdrant stores, making the Wikipedia MCP source redundant and potentially causing duplicate evidence.

### Proposed Solution
Remove Wikipedia MCP integration:
- Remove Wikipedia from MCP server (`src/ohi-mcp-server/`)
- Update documentation to reflect local Wikipedia data
- Ensure local Wikipedia data is prioritized
- Clean up any Wikipedia-specific MCP code

### Implementation Areas
- `src/ohi-mcp-server/` - MCP server
- Configuration files
- Documentation updates

### Success Criteria
- Wikipedia MCP source removed
- No duplicate evidence from Wikipedia
- Documentation updated

---

## Issue 6: Extend Ingestion to Check for New Wikipedia Dumps

**Title:** `[Feature]: Auto-detect New Wikipedia Dumps`

**Labels:** `enhancement`, `ingestion`, `automation`

**Category:** Developer Experience

**Description:**

### Problem Statement
Currently, the ingestion process requires manual checking for new Wikipedia dumps. This should be automated to ensure the knowledge base stays up-to-date.

### Proposed Solution
Implement automated dump detection:
- Periodically check Wikimedia dump repository
- Compare available dumps with ingested versions
- Notify or auto-trigger ingestion of new dumps
- Add admin dashboard notification for new dumps

### Implementation Areas
- New module: `src/ingestion/dump_monitor.py`
- Scheduled task or cron job
- Admin dashboard notifications

### Success Criteria
- Automated detection of new Wikipedia dumps
- Notification system for administrators
- Optional auto-ingestion of new dumps

---

## Issue 7: Extend Ingestion to Use All Dump Data

**Title:** `[Feature]: Ingest All Wikipedia Dump Data (Beyond Multistream)`

**Labels:** `enhancement`, `ingestion`, `data`

**Category:** Verification Logic

**Description:**

### Problem Statement
Current ingestion only processes multistream dumps. Other valuable data files in Wikipedia dumps (redirects, categories, page properties, etc.) are not being utilized.

### Proposed Solution
Extend ingestion pipeline to process:
- Article text (multistream) ✓ (current)
- Redirects
- Categories
- Page links
- External links
- Page properties
- Abstracts

### Implementation Areas
- `src/ingestion/` - Pipeline extensions
- New parsers for different dump file types
- Data model updates for additional metadata

### Success Criteria
- All relevant Wikipedia dump files processed
- Richer knowledge graph with metadata
- Better verification accuracy from additional context

---

## Issue 8: Rework Neo4j Knowledge Retrieval

**Title:** `[Enhancement]: Rework Neo4j Knowledge Retrieval Logic`

**Labels:** `enhancement`, `neo4j`, `performance`

**Category:** Performance

**Description:**

### Problem Statement
The current Neo4j knowledge retrieval may not be optimized for the specific graph patterns and queries used in OHI. Performance and accuracy could be improved.

### Proposed Solution
Redesign Neo4j retrieval:
- Analyze current query patterns
- Optimize Cypher queries for common access patterns
- Add graph-specific indexes
- Implement more sophisticated graph traversal strategies
- Leverage Neo4j's native graph algorithms

### Implementation Areas
- `src/api/src/open_hallucination_index/adapters/neo4j_adapter.py`
- Query optimization
- Index management

### Success Criteria
- Faster graph queries
- More relevant evidence retrieval
- Better utilization of graph relationships

---

## Issue 9: Prioritize Neo4j Over Qdrant

**Title:** `[Enhancement]: Prioritize Neo4j Over Qdrant in Evidence Collection`

**Labels:** `enhancement`, `architecture`, `prioritization`

**Category:** Verification Logic

**Description:**

### Problem Statement
Current evidence collection treats Neo4j and Qdrant equally. Graph-based evidence (Neo4j) may provide more structured and verifiable information and should be prioritized.

### Proposed Solution
Implement tiered evidence collection:
1. First, query Neo4j for graph-based evidence
2. If insufficient, supplement with Qdrant semantic search
3. If still insufficient, query MCP sources
4. Adjust scoring weights to favor graph evidence

### Implementation Areas
- `src/api/src/open_hallucination_index/adapters/evidence_collector.py`
- Evidence prioritization logic
- Scoring adjustments

### Success Criteria
- Neo4j queried before Qdrant
- Graph evidence weighted higher in scoring
- Configurable prioritization strategy

---

## Issue 10: Validate Need for Both Neo4j and Qdrant

**Title:** `[Research]: Find Valid Reason for Having Both Neo4j and Qdrant`

**Labels:** `research`, `architecture`, `documentation`

**Category:** Documentation

**Description:**

### Problem Statement
The architecture uses both Neo4j (graph) and Qdrant (vector) for knowledge storage. We should clearly document why both are necessary and what unique value each provides.

### Proposed Solution
Research and document:
- Unique capabilities of Neo4j vs Qdrant
- Use cases where each excels
- Query patterns that benefit from each
- Performance characteristics comparison
- Cost-benefit analysis
- Recommendation for different deployment scenarios

### Deliverables
- Architecture decision record (ADR)
- Documentation update explaining the dual-store approach
- Benchmarks comparing single-store vs dual-store accuracy

### Success Criteria
- Clear documentation of why both stores are used
- Evidence-based justification
- Guidance for users on when to use which

---

## Issue 11: Fix GitHub Actions CI/CD Pipeline

**Title:** `[Bug]: Fix GitHub Actions CI/CD Pipeline`

**Labels:** `bug`, `ci-cd`, `github-actions`

**Category:** Developer Experience

**Description:**

### Problem Statement
GitHub Actions workflows are failing or not functioning correctly, blocking automated testing and deployment.

### Proposed Solution
Investigate and fix GitHub Actions issues:
- Review workflow files in `.github/workflows/`
- Fix any syntax or configuration errors
- Update deprecated actions
- Ensure all jobs pass
- Add proper error handling
- Improve workflow documentation

### Implementation Areas
- `.github/workflows/` - All workflow files
- Repository settings (secrets, environments)

### Success Criteria
- All GitHub Actions workflows passing
- CI runs successfully on pull requests
- CD deploys correctly (if applicable)

---

## Issue 12: Fix Loading Bar Progress Color

**Title:** `[Bug]: Fix Loading Bar Progress Color`

**Labels:** `bug`, `frontend`, `ui`

**Category:** API / Endpoints

**Description:**

### Problem Statement
The loading bar progress indicator has incorrect or inconsistent colors that don't match the design system or provide proper visual feedback.

### Proposed Solution
Fix the loading bar styling:
- Update color scheme to match design system
- Ensure proper contrast and accessibility
- Add color states (loading, success, error)
- Test across different themes/modes

### Implementation Areas
- `src/frontend/` - Loading bar component
- CSS/Tailwind styling

### Success Criteria
- Loading bar has correct, consistent colors
- Accessible contrast ratios (WCAG AA)
- Proper visual feedback for different states

---

## Issue 13: Create Admin Property for User Model

**Title:** `[Feature]: Add Admin Property to User Model`

**Labels:** `enhancement`, `backend`, `authentication`

**Category:** API / Endpoints

**Description:**

### Problem Statement
Currently, there's no way to designate users as administrators with elevated privileges. An admin property is needed to enable role-based access control.

### Proposed Solution
Extend user model with admin capabilities:
- Add `is_admin` boolean field to user model
- Add database migration
- Implement admin-only route protection
- Add admin check middleware
- Update authentication logic

### Implementation Areas
- `src/api/src/open_hallucination_index/domain/` - User model
- Database schema and migrations
- Authentication middleware

### Success Criteria
- Users can be marked as admin
- Admin-only routes are protected
- Middleware enforces admin permissions

---

## Issue 14: Create Admin Dashboard with Live Logs and Controls

**Title:** `[Feature]: Admin Dashboard with Live Anonymous Logs and Controls`

**Labels:** `enhancement`, `frontend`, `admin`

**Category:** API / Endpoints

**Description:**

### Problem Statement
Administrators need a centralized dashboard to monitor system activity, view logs, and manage users/resources without accessing backend systems directly.

### Proposed Solution
Build comprehensive admin dashboard:
- Live log streaming (anonymized)
- User management (list, edit, delete, token management)
- System statistics and metrics
- API key management
- Knowledge base status
- Real-time verification activity
- Admin controls for system settings

### Implementation Areas
- `src/frontend/app/admin/` - New admin pages
- Backend API endpoints for admin data
- WebSocket or SSE for live updates
- RBAC enforcement

### Success Criteria
- Admins can view live system logs
- User management capabilities
- Real-time metrics dashboard
- All actions properly secured

---

## Issue 15: Improve API for External Developer Reusability

**Title:** `[Enhancement]: Improve API Design for External Developer Reusability`

**Labels:** `enhancement`, `api`, `documentation`, `developer-experience`

**Category:** API / Endpoints

**Description:**

### Problem Statement
The API should be more developer-friendly, with better documentation, consistent patterns, and easier integration for external developers.

### Proposed Solution
Enhance API for external developers:
- Improve OpenAPI/Swagger documentation
- Add more examples and code samples
- Provide client SDK (Python, JavaScript)
- Better error messages and error codes
- Rate limiting headers
- Versioning strategy
- Developer quickstart guide

### Implementation Areas
- API documentation
- OpenAPI spec enhancements
- Client SDK development
- Developer portal

### Success Criteria
- Comprehensive API documentation
- Working client SDKs
- Positive feedback from external developers
- Reduced support requests

---

## Issue 16: Extend Benchmark with More HuggingFace Datasets

**Title:** `[Feature]: Extend Benchmark with More HuggingFace Datasets`

**Labels:** `enhancement`, `benchmark`, `evaluation`

**Category:** Verification Logic

**Description:**

### Problem Statement
The current benchmark suite uses limited datasets. Adding more HuggingFace datasets would provide better evaluation coverage and more robust performance metrics.

### Proposed Solution
Integrate additional HuggingFace datasets:
- Survey relevant datasets (FEVER, HoVer, VitaminC, etc.)
- Add dataset loaders to benchmark suite
- Standardize evaluation metrics across datasets
- Generate comparative reports
- Add dataset-specific configurations

### Implementation Areas
- `src/benchmark/` - Benchmark suite
- Dataset adapters
- Evaluation metrics

### Success Criteria
- At least 5+ additional HuggingFace datasets integrated
- Comparative benchmark reports
- Automated evaluation pipeline

---

## Issue 17: Better Reasoning Display for Declined Text Verification

**Title:** `[Enhancement]: Better Reasoning Display for Declined Precheck`

**Labels:** `enhancement`, `frontend`, `ux`

**Category:** API / Endpoints

**Description:**

### Problem Statement
When text verification is declined during precheck, users don't get clear feedback on why it was rejected. Better reasoning and display would improve user experience.

### Proposed Solution
Enhance precheck rejection feedback:
- Detailed rejection reasons
- Clear explanation of what went wrong
- Suggestions for fixing the input
- Better UI display for rejection messages
- Categorize rejection types

### Implementation Areas
- `src/api/` - Precheck logic and response
- `src/frontend/` - Rejection display UI
- Error message templates

### Success Criteria
- Clear, actionable rejection messages
- Users understand why text was declined
- Reduced confusion and support requests

---

## Issue 18: Token Refund for Aborted/Failed/Declined Verifications

**Title:** `[Feature]: Token Refund for Aborted/Failed/Declined Verifications`

**Labels:** `enhancement`, `billing`, `tokens`

**Category:** API / Endpoints

**Description:**

### Problem Statement
Users are charged tokens even when verifications are aborted, fail, or are declined during precheck. This is unfair as no meaningful work was performed.

### Proposed Solution
Implement token refund logic:
- Identify verification failure/abort scenarios
- Automatically refund tokens for:
  - Precheck rejections
  - System errors/failures
  - User-initiated aborts
- Add refund audit log
- Display refunds in user dashboard

### Implementation Areas
- Token management service
- Verification pipeline error handling
- User dashboard token history

### Success Criteria
- Tokens automatically refunded on failure
- Refunds visible in user dashboard
- Audit trail for all refunds

---

## Issue 19: Extend Admin Dashboard with API Key Management

**Title:** `[Feature]: Comprehensive API Key Management in Admin Dashboard`

**Labels:** `enhancement`, `admin`, `api-keys`

**Category:** API / Endpoints

**Description:**

### Problem Statement
Administrators need the ability to create and manage API keys for users, including user-specific keys, master keys, and guest keys with different token allocations.

### Proposed Solution
Implement multi-tier API key system:
- **User API Keys**: Assigned to user email, consumes user's tokens
- **Master Keys**: Unlimited tokens, admin-only, multiple allowed (0..n)
- **Guest Keys**: Anonymous keys with fixed token allocation
- Admin UI for creating/revoking/listing all key types
- Key metadata (created date, last used, usage stats)

### Implementation Areas
- `src/api/` - API key models and authentication
- `src/frontend/app/admin/` - API key management UI
- Database schema for different key types
- Token consumption logic per key type

### Success Criteria
- Admins can create all three key types
- User keys consume user tokens
- Master keys have unlimited tokens
- Guest keys have configurable token limits

---

## Issue 20: Implement Public API with API Key Authentication

**Title:** `[Feature]: Public API with API Key Authentication`

**Labels:** `enhancement`, `api`, `authentication`

**Category:** API / Endpoints

**Description:**

### Problem Statement
The API should support public access via API keys, allowing users to create their own keys that deduct from their token balance.

### Proposed Solution
Implement API key authentication:
- API key generation for users
- API key validation middleware
- Token deduction per request
- Rate limiting per key
- Key rotation support
- Usage analytics per key

### Implementation Areas
- Authentication middleware
- User dashboard for key management
- Token consumption logic
- Rate limiting

### Success Criteria
- Users can create/manage their own API keys
- API key authentication works for all endpoints
- Tokens properly deducted per request
- Rate limiting enforced

---

## Issue 21: Implement MCP Server with API Key Authentication

**Title:** `[Feature]: MCP Server with API Key Authentication`

**Labels:** `enhancement`, `mcp`, `authentication`

**Category:** Knowledge Sources (MCP)

**Description:**

### Problem Statement
The MCP server should support API key authentication, allowing it to be used as a public service with token-based access control.

### Proposed Solution
Add API key auth to MCP server:
- Integrate with API key authentication system
- Token consumption for MCP requests
- Support same key types (user, master, guest)
- Rate limiting per key
- Usage tracking

### Implementation Areas
- `src/ohi-mcp-server/` - MCP server
- Authentication middleware
- Token integration with main API

### Success Criteria
- MCP server requires valid API key
- Tokens deducted for MCP usage
- Same keys work for both API and MCP

---

## Issue 22: Implement Token Amount Check Tool for API Keys

**Title:** `[Feature]: API & MCP Token Amount Check Tool`

**Labels:** `enhancement`, `api`, `tokens`

**Category:** API / Endpoints

**Description:**

### Problem Statement
Users need a way to check their remaining token balance for a given API key before making requests.

### Proposed Solution
Create token check endpoint:
- `GET /api/v1/tokens/balance` endpoint
- Accept API key in header or parameter
- Return remaining token count
- Return token usage statistics
- Add MCP tool for token checking
- Add frontend widget for token display

### Implementation Areas
- New API endpoint
- MCP tool integration
- Frontend token display component

### Success Criteria
- Endpoint returns accurate token balance
- Works with all key types (user, master, guest)
- MCP tool available for token checking
- Frontend displays current balance

---

## Summary

**Total Issues:** 22

**Categories:**
- Performance: 3 issues
- Verification Logic: 5 issues
- Knowledge Sources (MCP): 3 issues
- API / Endpoints: 9 issues
- Developer Experience: 3 issues
- Frontend: 2 issues
- Ingestion: 4 issues
- Documentation: 1 issue
- Architecture: 3 issues

**Priority Recommendations:**
1. **High Priority**: Issues 11, 13, 18, 20, 22 (Foundation for public API)
2. **Medium Priority**: Issues 4, 8, 9, 14, 19, 21 (Core improvements)
3. **Low Priority**: Issues 1, 2, 5, 6, 7, 10, 12, 15, 16, 17 (Nice-to-have enhancements)

---

## How to Create These Issues

### Option 1: Manual Creation (GitHub UI)
1. Go to https://github.com/shiftbloom-studio/open-hallucination-index/issues
2. Click "New Issue"
3. Choose "✨ Feature Request" or "🐛 Bug Report" template
4. Copy the content from each issue above
5. Add the specified labels

### Option 2: Using GitHub CLI (with authentication)
```bash
gh auth login
gh issue create --title "TITLE" --body "DESCRIPTION" --label "label1,label2"
```

### Option 3: Using GitHub API Script
Create a script that uses GitHub API to bulk-create these issues (requires personal access token).
