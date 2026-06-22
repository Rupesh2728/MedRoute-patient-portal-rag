#!/bin/bash
# Removes large files from the git history and force-pushes a clean repo.
# Run from the project root: bash fix_repo_size.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=== Step 1: identify largest tracked files (top 20) ==="
git ls-files | while read f; do
    [ -f "$f" ] && echo "$(wc -c <"$f")  $f"
done | sort -rn | head -20

echo ""
echo "=== Step 2: append known-big paths to .gitignore ==="

cat >> .gitignore <<'EOF'

# --- Added by fix_repo_size.sh — large local artefacts ---
models/
patient_index/
medlineplus_index/
streamlit_app/.venv/
**/.venv/
**/__pycache__/
*.bin
*.pt
*.pkl
*.faiss
*.npz
*.npy
*.safetensors
*.gguf
*.ggml
*.h5
*.onnx
EOF

# Special-case: keep the trained router weights (small) but exclude everything
# else matching *.pt / *.pkl. Use force-include:
cat >> .gitignore <<'EOF'

# --- Force-include trained router artefacts (small) ---
!patient_router/
!patient_router/*.pt
!patient_router/*.pkl
EOF

echo ""
echo "=== Step 3: untrack large files from git index ==="

# Remove from index everything matching the new ignore rules (preserves files
# on disk, just removes from git tracking)
git rm -rf --cached models 2>/dev/null || true
git rm -rf --cached patient_index 2>/dev/null || true
git rm -rf --cached medlineplus_index 2>/dev/null || true
git rm -rf --cached streamlit_app/.venv 2>/dev/null || true
find . -name "__pycache__" -type d 2>/dev/null | while read d; do
    git rm -rf --cached "$d" 2>/dev/null || true
done

# Sweep any leftover *.bin / *.npz / *.safetensors that slipped in
git ls-files | grep -E '\.(bin|safetensors|gguf|ggml|h5|onnx|faiss|npz|npy)$' | while read f; do
    git rm -f --cached "$f" 2>/dev/null || true
done

echo ""
echo "=== Step 4: rewrite git history (squash everything into one clean commit) ==="
# Easiest way to drop big files from history: remove .git, re-init.
# We keep your current working directory untouched.

REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
echo "Saved remote URL: $REMOTE_URL"

rm -rf .git
git init -q
git config user.email "maqsoodhuman@gmail.com"
git config user.name "Maqsood"
git branch -M main

git add -A
git commit -q -m "MedRoute patient portal — cleaned commit

CSE 676A · Spring 2026 · University at Buffalo

Excludes pre-trained model weights, FAISS indices, and venvs
(all regeneratable via scripts/setup_local_pipeline.py)."

echo ""
echo "=== Step 5: total size of cleaned commit ==="
du -sh .git
git ls-files | wc -l
echo "files tracked"
git ls-files | xargs -I{} sh -c 'wc -c <"{}" 2>/dev/null' | awk '{s+=$1} END {printf "Total tracked bytes: %.1f MB\n", s/1024/1024}'

echo ""
echo "=== Step 6: re-attach origin and force-push ==="
if [ -n "$REMOTE_URL" ]; then
    git remote add origin "$REMOTE_URL"
    echo "Force-pushing to $REMOTE_URL..."
    git push --force --set-upstream origin main
    echo ""
    echo "Done. Repo URL: $REMOTE_URL"
else
    echo "No remote URL saved. Run manually:"
    echo "  git remote add origin https://github.com/Maqsoodhuman/patient-portal-rag.git"
    echo "  git push --force --set-upstream origin main"
fi
