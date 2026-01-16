#!/usr/bin/env python3
"""
Script to create GitHub issues from the task list using GitHub API.
Requires: pip install requests
Usage: python scripts/create_github_issues.py

Set GITHUB_TOKEN environment variable with a personal access token that has 'repo' scope.
"""

import os
import sys
import json
import requests
from typing import Dict, List

REPO_OWNER = "shiftbloom-studio"
REPO_NAME = "open-hallucination-index"
GITHUB_API_URL = "https://api.github.com"


def get_github_token() -> str:
    """Get GitHub token from environment."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set.")
        print("Create a personal access token at: https://github.com/settings/tokens")
        print("Token needs 'repo' scope.")
        print("Then run: export GITHUB_TOKEN='your_token_here'")
        sys.exit(1)
    return token


def create_issue(token: str, title: str, body: str, labels: List[str]) -> Dict:
    """Create a GitHub issue."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    
    data = {
        "title": title,
        "body": body,
        "labels": labels,
    }
    
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/issues"
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 201:
        return response.json()
    else:
        print(f"Error creating issue: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def main():
    """Main function to create all issues."""
    print(f"Creating GitHub issues for {REPO_OWNER}/{REPO_NAME}...")
    print()
    
    token = get_github_token()
    
    issues = [
        {
            "title": "[Enhancement]: Optimize Ingestion Upsert Operations",
            "labels": ["enhancement", "performance", "ingestion"],
            "body": """### Problem Statement
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
- Better handling of duplicate entries"""
        },
        {
            "title": "[Feature]: Implement Ingestion LLM Gardener for Data Quality",
            "labels": ["enhancement", "ingestion", "llm"],
            "body": """### Problem Statement
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
- Reduced false positives in verification"""
        },
        {
            "title": "[Feature]: Multiple Knowledge Base Support",
            "labels": ["enhancement", "architecture", "knowledge-sources"],
            "body": """### Problem Statement
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
- Admin dashboard shows all configured knowledge bases"""
        },
        {
            "title": "[Enhancement]: Ensure Target Evidence Count Per Claim",
            "labels": ["enhancement", "verification"],
            "body": """### Problem Statement
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
- Configurable evidence targets per claim"""
        },
        {
            "title": "[Cleanup]: Remove Wikipedia MCP Source (Redundant)",
            "labels": ["cleanup", "mcp", "optimization"],
            "body": """### Problem Statement
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
- Documentation updated"""
        },
        {
            "title": "[Feature]: Auto-detect New Wikipedia Dumps",
            "labels": ["enhancement", "ingestion", "automation"],
            "body": """### Problem Statement
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
- Optional auto-ingestion of new dumps"""
        },
        {
            "title": "[Feature]: Ingest All Wikipedia Dump Data (Beyond Multistream)",
            "labels": ["enhancement", "ingestion", "data"],
            "body": """### Problem Statement
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
- Better verification accuracy from additional context"""
        },
        {
            "title": "[Enhancement]: Rework Neo4j Knowledge Retrieval Logic",
            "labels": ["enhancement", "neo4j", "performance"],
            "body": """### Problem Statement
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
- Better utilization of graph relationships"""
        },
        {
            "title": "[Enhancement]: Prioritize Neo4j Over Qdrant in Evidence Collection",
            "labels": ["enhancement", "architecture", "prioritization"],
            "body": """### Problem Statement
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
- Configurable prioritization strategy"""
        },
        {
            "title": "[Research]: Find Valid Reason for Having Both Neo4j and Qdrant",
            "labels": ["research", "architecture", "documentation"],
            "body": """### Problem Statement
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
- Guidance for users on when to use which"""
        },
        {
            "title": "[Bug]: Fix GitHub Actions CI/CD Pipeline",
            "labels": ["bug", "ci-cd", "github-actions"],
            "body": """### Problem Statement
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
- CD deploys correctly (if applicable)"""
        },
        {
            "title": "[Bug]: Fix Loading Bar Progress Color",
            "labels": ["bug", "frontend", "ui"],
            "body": """### Problem Statement
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
- Proper visual feedback for different states"""
        },
        {
            "title": "[Feature]: Add Admin Property to User Model",
            "labels": ["enhancement", "backend", "authentication"],
            "body": """### Problem Statement
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
- Middleware enforces admin permissions"""
        },
        {
            "title": "[Feature]: Admin Dashboard with Live Anonymous Logs and Controls",
            "labels": ["enhancement", "frontend", "admin"],
            "body": """### Problem Statement
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
- All actions properly secured"""
        },
        {
            "title": "[Enhancement]: Improve API Design for External Developer Reusability",
            "labels": ["enhancement", "api", "documentation", "developer-experience"],
            "body": """### Problem Statement
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
- Reduced support requests"""
        },
        {
            "title": "[Feature]: Extend Benchmark with More HuggingFace Datasets",
            "labels": ["enhancement", "benchmark", "evaluation"],
            "body": """### Problem Statement
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
- Automated evaluation pipeline"""
        },
        {
            "title": "[Enhancement]: Better Reasoning Display for Declined Precheck",
            "labels": ["enhancement", "frontend", "ux"],
            "body": """### Problem Statement
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
- Reduced confusion and support requests"""
        },
        {
            "title": "[Feature]: Token Refund for Aborted/Failed/Declined Verifications",
            "labels": ["enhancement", "billing", "tokens"],
            "body": """### Problem Statement
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
- Audit trail for all refunds"""
        },
        {
            "title": "[Feature]: Comprehensive API Key Management in Admin Dashboard",
            "labels": ["enhancement", "admin", "api-keys"],
            "body": """### Problem Statement
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
- Guest keys have configurable token limits"""
        },
        {
            "title": "[Feature]: Public API with API Key Authentication",
            "labels": ["enhancement", "api", "authentication"],
            "body": """### Problem Statement
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
- Rate limiting enforced"""
        },
        {
            "title": "[Feature]: MCP Server with API Key Authentication",
            "labels": ["enhancement", "mcp", "authentication"],
            "body": """### Problem Statement
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
- Same keys work for both API and MCP"""
        },
        {
            "title": "[Feature]: API & MCP Token Amount Check Tool",
            "labels": ["enhancement", "api", "tokens"],
            "body": """### Problem Statement
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
- Frontend displays current balance"""
        },
    ]
    
    created_count = 0
    failed_count = 0
    
    for i, issue in enumerate(issues, 1):
        print(f"Creating Issue {i}/{len(issues)}: {issue['title']}")
        result = create_issue(token, issue["title"], issue["body"], issue["labels"])
        
        if result:
            created_count += 1
            print(f"  ✓ Created: {result['html_url']}")
        else:
            failed_count += 1
            print(f"  ✗ Failed")
        print()
    
    print()
    print(f"Summary:")
    print(f"  ✓ Created: {created_count}")
    print(f"  ✗ Failed: {failed_count}")
    print()
    print(f"View all issues at: https://github.com/{REPO_OWNER}/{REPO_NAME}/issues")


if __name__ == "__main__":
    main()
