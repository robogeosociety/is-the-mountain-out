#!/usr/bin/env bash
# BREAK-GLASS deploy path — NOT the normal one.
#
# The normal deploy is CI: .github/workflows/deploy-worker.yml runs the worker
# tests + typecheck and then `wrangler deploy` on every push to main touching
# worker/**, recording the same GitHub Deployment this script does. Prefer a
# push to main, or `gh workflow run deploy-worker.yml`, over running this.
#
# Use this script only when CI cannot: GitHub Actions is down, the repo's
# CLOUDFLARE_API_TOKEN is missing/expired, or you need to ship an uncommitted
# working tree in an emergency. Because it deploys whatever is on YOUR disk, a
# run from a dirty tree puts code in production that exists in no commit — the
# GitHub Deployment it records will point at a ref that does not match the live
# Worker. Commit first if at all possible.
#
# Deploys the mountain-inference Worker + container via wrangler and records a
# GitHub Deployment, so the repo's Environments page and the deployed commit
# show deploy history plus the live URL.
#
# Auth is `wrangler login` (interactive OAuth), unlike CI, which uses the
# scoped CLOUDFLARE_API_TOKEN secret. It does NOT use Terraform. The container
# image must already be in the Cloudflare managed registry and referenced by
# worker/wrangler.toml — build/push it with `wrangler containers push` first
# (see worker/wrangler.toml and the README).
#
# The GitHub Deployment is best-effort: a gh/API hiccup logs a warning but never
# blocks the actual deploy. A failed `wrangler deploy` marks the deployment
# `failure` and exits non-zero.
#
# Requires: wrangler (authenticated), gh (authenticated), git.
#
# Env overrides:
#   DEPLOY_ENV   GitHub Deployment environment      (default: production)
#   WORKER_URL   live URL recorded on the status    (default: the workers.dev URL)
#   GH_REPO      owner/repo                          (default: robogeosociety/is-the-mountain-out)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_DIR="$REPO_ROOT/worker"
GH_REPO="${GH_REPO:-robogeosociety/is-the-mountain-out}"
DEPLOY_ENV="${DEPLOY_ENV:-production}"
WORKER_URL="${WORKER_URL:-https://mountain-inference.tommy-b-doerr.workers.dev}"

log()  { printf '\033[1;34m▸\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*" >&2; }

command -v gh  >/dev/null 2>&1 || { echo "gh not found in PATH" >&2; exit 1; }
command -v npx >/dev/null 2>&1 || { echo "npx (Node.js) not found in PATH" >&2; exit 1; }

REF="$(git -C "$REPO_ROOT" rev-parse HEAD)"
DEPLOY_ID=""

warn "break-glass path — the normal deploy is CI (.github/workflows/deploy-worker.yml)."
warn "Prefer: push to main (worker/** changes) or \`gh workflow run deploy-worker.yml\`."
if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
  warn "working tree is DIRTY: this deploys code that exists in no commit, and the"
  warn "GitHub Deployment recorded for ${REF:0:7} will not match what goes live."
fi

create_deployment() {
  local body
  body="$(printf '{"ref":"%s","environment":"%s","production_environment":true,"auto_merge":false,"required_contexts":[],"description":"wrangler deploy of mountain-inference"}' "$REF" "$DEPLOY_ENV")"
  DEPLOY_ID="$(printf '%s' "$body" | gh api "repos/$GH_REPO/deployments" --input - --jq '.id' 2>/dev/null || true)"
}

set_status() { # $1=state  $2=description
  [ -n "$DEPLOY_ID" ] || return 0
  gh api "repos/$GH_REPO/deployments/$DEPLOY_ID/statuses" \
    -f state="$1" \
    -f environment="$DEPLOY_ENV" \
    -f environment_url="$WORKER_URL" \
    -f description="$2" >/dev/null 2>&1 || warn "could not set deployment status=$1"
}

log "Creating GitHub Deployment (env=$DEPLOY_ENV, ref=${REF:0:7})…"
create_deployment
if [ -n "$DEPLOY_ID" ]; then
  log "GitHub Deployment #$DEPLOY_ID created"
  set_status in_progress "Deploying via wrangler"
else
  warn "GitHub Deployment not created (continuing with deploy anyway)"
fi

log "Running wrangler deploy…"
if ( cd "$WORKER_DIR" && npx wrangler deploy ); then
  set_status success "Deployed to $WORKER_URL"
  log "Deploy succeeded → $WORKER_URL"
  [ -n "$DEPLOY_ID" ] && log "GitHub Deployment #$DEPLOY_ID marked success"
else
  rc=$?
  set_status failure "wrangler deploy failed (exit $rc)"
  warn "Deploy failed (exit $rc); GitHub Deployment marked failure"
  exit "$rc"
fi
