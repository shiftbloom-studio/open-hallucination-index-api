# GitHub Issues Creation Scripts

This directory contains scripts to create GitHub issues for the Open Hallucination Index project based on the task list from the problem statement.

## Available Scripts

### 1. Bash Script (`create_github_issues.sh`)

Uses GitHub CLI (`gh`) to create issues.

**Prerequisites:**
- Install GitHub CLI: https://cli.github.com/
- Authenticate: `gh auth login`

**Usage:**
```bash
./scripts/create_github_issues.sh
```

### 2. Python Script (`create_github_issues.py`)

Uses GitHub REST API via Python requests library.

**Prerequisites:**
```bash
pip install requests
```

**Setup:**
1. Create a Personal Access Token at: https://github.com/settings/tokens
2. Select `repo` scope
3. Set environment variable:
   ```bash
   export GITHUB_TOKEN='your_token_here'
   ```

**Usage:**
```bash
python scripts/create_github_issues.py
```

Or make it executable and run directly:
```bash
chmod +x scripts/create_github_issues.py
./scripts/create_github_issues.py
```

## Issues to be Created

The scripts will create **22 GitHub issues** covering:

1. **Ingestion Pipeline Improvements** (7 issues)
   - Optimize upsert operations
   - LLM-based data quality gardener
   - Auto-detect new Wikipedia dumps
   - Process all dump data types
   - Remove redundant Wikipedia MCP

2. **Verification & Evidence Collection** (4 issues)
   - Ensure target evidence count per claim
   - Rework Neo4j knowledge retrieval
   - Prioritize Neo4j over Qdrant
   - Research justification for dual-store architecture

3. **Admin Features** (4 issues)
   - Add admin property to user model
   - Build comprehensive admin dashboard
   - API key management system
   - Token refund mechanism

4. **Public API & Authentication** (4 issues)
   - Public API with API key auth
   - MCP server with API key auth
   - Token balance check endpoint
   - Improve API for external developers

5. **Frontend & UX** (2 issues)
   - Fix loading bar colors
   - Better reasoning display for declined verifications

6. **Infrastructure & Testing** (2 issues)
   - Fix GitHub Actions CI/CD
   - Extend benchmark with HuggingFace datasets

Each issue includes:
- Clear problem statement
- Proposed solution
- Implementation areas
- Success criteria
- Appropriate labels

## Troubleshooting

### Bash Script Issues

**Error: `gh not found`**
- Install GitHub CLI: https://cli.github.com/

**Error: `You are not logged into any GitHub hosts`**
- Run: `gh auth login`

### Python Script Issues

**Error: `ModuleNotFoundError: No module named 'requests'`**
- Run: `pip install requests`

**Error: `GITHUB_TOKEN environment variable not set`**
- Create token: https://github.com/settings/tokens
- Export it: `export GITHUB_TOKEN='your_token'`

**Error: `401 Unauthorized`**
- Token expired or invalid
- Create new token with `repo` scope

**Error: `403 Forbidden`**
- Token lacks `repo` scope
- Recreate token with proper permissions

## Manual Creation

If you prefer to create issues manually, refer to `GITHUB_ISSUES_TO_CREATE.md` in the repository root. It contains formatted content for all 22 issues that can be copy-pasted into the GitHub web interface.

## Issue Labels

The following labels will be applied to issues (create them if they don't exist):

- `enhancement` - Feature requests and improvements
- `bug` - Bug reports
- `performance` - Performance optimizations
- `ingestion` - Ingestion pipeline related
- `verification` - Verification logic related
- `frontend` - Frontend/UI related
- `admin` - Admin features
- `api` - API related
- `authentication` - Auth related
- `mcp` - MCP server related
- `tokens` - Token system related
- `documentation` - Documentation updates
- `ci-cd` - CI/CD pipeline related
- `github-actions` - GitHub Actions specific
- `ux` - User experience
- `research` - Research tasks
- `architecture` - Architecture decisions
- `neo4j` - Neo4j specific
- `cleanup` - Code cleanup
- `optimization` - Optimizations
- `automation` - Automation tasks
- `benchmark` - Benchmark suite
- `evaluation` - Evaluation metrics
- `billing` - Billing/token charging
- `api-keys` - API key management
- `knowledge-sources` - Knowledge source related
- `prioritization` - Priority changes
- `llm` - LLM related
- `data` - Data processing
- `backend` - Backend specific
- `developer-experience` - Developer tools/DX

## Post-Creation Steps

After creating the issues:

1. **Triage**: Review and adjust labels/priorities as needed
2. **Assignment**: Assign issues to team members
3. **Milestones**: Group related issues into milestones
4. **Projects**: Add issues to GitHub Projects for tracking
5. **Dependencies**: Link issues that depend on each other

## Priority Recommendations

**High Priority** (Foundation for public API):
- Issue 11: Fix GitHub Actions CI/CD
- Issue 13: Add Admin Property to User Model
- Issue 18: Token Refund System
- Issue 20: Public API with API Key Auth
- Issue 22: Token Balance Check Tool

**Medium Priority** (Core improvements):
- Issue 4: Ensure Target Evidence Count
- Issue 8: Rework Neo4j Retrieval
- Issue 9: Prioritize Neo4j Over Qdrant
- Issue 14: Admin Dashboard
- Issue 19: API Key Management
- Issue 21: MCP Server Auth

**Low Priority** (Nice-to-have):
- All remaining issues

## Support

For issues with the scripts, please:
1. Check the troubleshooting section
2. Verify your authentication setup
3. Check GitHub API status: https://www.githubstatus.com/
4. Open an issue in the repository
