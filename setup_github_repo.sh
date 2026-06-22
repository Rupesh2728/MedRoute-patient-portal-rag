#!/bin/bash
# One-shot script to initialize git, commit everything, and push to GitHub.
# Run from the project root: bash setup_github_repo.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

REPO_NAME="patient-portal-rag"

# ---- Step 0: clear any stale git locks from earlier attempts -----------------
rm -f .git/index.lock .git/HEAD.lock 2>/dev/null || true

# ---- Step 1: init git (idempotent — safe to re-run) --------------------------
if [ ! -d .git ]; then
    git init -q
fi
git config user.email "maqsoodhuman@gmail.com"
git config user.name "Maqsood"
git branch -M main 2>/dev/null || true

# ---- Step 2: stage and commit -----------------------------------------------
echo "Staging files..."
git add -A

NUM_FILES=$(git status --short | wc -l | tr -d ' ')
echo "  $NUM_FILES items to commit"

if git rev-parse HEAD >/dev/null 2>&1; then
    # Already has commits — make a new one
    git commit -m "Update: $(date +%Y-%m-%d)" || echo "Nothing to commit"
else
    # First commit
    git commit -q -m "Initial commit: MedRoute patient portal

- Streamlit app with live mock + 3-mode pipeline
- 400 synthetic patients, 2000 labelled questions
- MedlinePlus corpus (2127 articles), PrimeKG integration
- 27-feature MLP router with strict medically-aware labels
- MedQA + patient-portal notebooks, training scripts, results

CSE 676A · Spring 2026 · University at Buffalo"
    echo "Initial commit created"
fi

# ---- Step 3: create GitHub repo and push ------------------------------------
echo ""
echo "Creating GitHub repo: $REPO_NAME (public)..."

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: GitHub CLI (gh) not installed."
    echo ""
    echo "Install it: brew install gh"
    echo "Then authenticate: gh auth login"
    echo "Then re-run this script."
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "GitHub CLI not authenticated. Run: gh auth login"
    echo "Then re-run this script."
    exit 1
fi

# Check if repo already exists
if gh repo view "$REPO_NAME" >/dev/null 2>&1; then
    echo "Repo '$REPO_NAME' already exists on GitHub."
    REPO_URL=$(gh repo view "$REPO_NAME" --json url -q .url)
    echo "URL: $REPO_URL"

    # Make sure origin is set
    if ! git remote get-url origin >/dev/null 2>&1; then
        gh repo set-default "$REPO_NAME" 2>/dev/null || true
        git remote add origin "$REPO_URL.git"
    fi
else
    # Create the public repo and push
    gh repo create "$REPO_NAME" \
        --public \
        --source=. \
        --remote=origin \
        --description "MedRoute — context-aware clinical decision support with learned routing between LLM, RAG, and KG (CSE 676A Spring 2026)" \
        --push
fi

echo ""
echo "Done. Repo URL:"
gh repo view "$REPO_NAME" --json url -q .url
echo ""
echo "View at: $(gh repo view "$REPO_NAME" --json url -q .url)"
