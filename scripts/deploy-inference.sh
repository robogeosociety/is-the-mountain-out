#!/usr/bin/env bash
# Deploy the Cloudflare Worker + Container that runs */15 mountain inference.
#
# Wraps the manual cutover described in
# .github/workflows/build-inference-image.yml, terraform/cloudflare_worker.tf,
# and worker/wrangler.toml into one command.
#
# Required env vars:
#   CLOUDFLARE_API_TOKEN    Cloudflare API token (Workers + R2 + Containers scopes)
#   CLOUDFLARE_ACCOUNT_ID   Cloudflare account ID
#   R2_ACCESS_KEY_ID        R2 S3-API access key — pushed as Worker secret so the
#                           container can pull its checkpoint from R2 at startup
#   R2_SECRET_ACCESS_KEY    R2 S3-API secret access key (same purpose as above)
#
# Optional:
#   INFERENCE_IMAGE_TAG    Image tag to deploy (default: HEAD SHA on origin/main)
#   GH_TOKEN               PAT for polling GHCR for the image; defaults to gh CLI auth
#   SKIP_IMAGE_WAIT=1      Don't poll GHCR — assume the image already exists
#   AUTO_APPROVE=1         Pass -auto-approve to terraform apply
#   PLAN_ONLY=1            Run `terraform plan` instead of apply
#   TAIL=1                 After apply, run `wrangler tail mountain-inference`

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TERRAFORM_DIR="$REPO_ROOT/terraform"
WORKER_DIR="$REPO_ROOT/worker"

GH_OWNER="robogeosociety"
GH_REPO="is-the-mountain-out"
IMAGE_REPO="ghcr.io/$GH_OWNER/$GH_REPO/inference"

log()   { printf '\033[1;34m▸\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m!\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    die "$name is required (export it or set it in your shell)"
  fi
}

resolve_image_tag() {
  if [[ -n "${INFERENCE_IMAGE_TAG:-}" ]]; then
    printf '%s\n' "$INFERENCE_IMAGE_TAG"
    return
  fi
  if ! git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    die "INFERENCE_IMAGE_TAG not set and not in a git checkout"
  fi
  git -C "$REPO_ROOT" fetch --quiet origin main 2>/dev/null || true
  git -C "$REPO_ROOT" rev-parse origin/main 2>/dev/null \
    || git -C "$REPO_ROOT" rev-parse HEAD
}

wait_for_image() {
  local tag="$1"
  local manifest_url="https://ghcr.io/v2/$GH_OWNER/$GH_REPO/inference/manifests/$tag"
  local token=""

  if [[ -n "${GH_TOKEN:-}" ]]; then
    token="$GH_TOKEN"
  elif command -v gh >/dev/null 2>&1; then
    token="$(gh auth token 2>/dev/null || true)"
  fi

  local auth_header=()
  if [[ -n "$token" ]]; then
    auth_header=(-H "Authorization: Bearer $(printf '%s' "$token" | base64 -w0 2>/dev/null || printf '%s' "$token" | base64)")
  fi

  log "Waiting for $IMAGE_REPO:$tag to appear on GHCR (up to 30 min)..."
  local deadline=$(( $(date +%s) + 1800 ))
  while (( $(date +%s) < deadline )); do
    local status
    status=$(curl -sS -o /dev/null -w '%{http_code}' \
      "${auth_header[@]}" \
      -H "Accept: application/vnd.oci.image.manifest.v1+json" \
      -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
      "$manifest_url" || echo "000")
    case "$status" in
      200) log "Image $tag is live."; return 0 ;;
      404) printf '.' ;;
      401|403)
        warn "GHCR returned $status — image may be private and \$GH_TOKEN missing. Treating as not-yet-published and retrying."
        printf '.' ;;
      *)
        warn "GHCR HTTP $status; will retry."
        printf '.' ;;
    esac
    sleep 15
  done
  die "Image $IMAGE_REPO:$tag did not appear within 30 minutes."
}

run_terraform() {
  local tag="$1"
  log "Running terraform from $TERRAFORM_DIR..."
  (
    cd "$TERRAFORM_DIR"
    terraform init -upgrade -input=false
    local action=apply
    [[ "${PLAN_ONLY:-}" == "1" ]] && action=plan
    local extra=()
    if [[ "$action" == "apply" && "${AUTO_APPROVE:-}" == "1" ]]; then
      extra+=(-auto-approve)
    fi
    TF_VAR_cloudflare_api_token="$CLOUDFLARE_API_TOKEN" \
    TF_VAR_cloudflare_account_id="$CLOUDFLARE_ACCOUNT_ID" \
    TF_VAR_r2_access_key_id="$R2_ACCESS_KEY_ID" \
    TF_VAR_r2_secret_access_key="$R2_SECRET_ACCESS_KEY" \
    TF_VAR_inference_image_tag="$tag" \
      terraform "$action" -input=false "${extra[@]}"
  )
}

maybe_tail() {
  [[ "${TAIL:-}" == "1" ]] || return 0
  log "Tailing worker logs (Ctrl-C to stop)..."
  (
    cd "$WORKER_DIR"
    CLOUDFLARE_API_TOKEN="$CLOUDFLARE_API_TOKEN" \
    CLOUDFLARE_ACCOUNT_ID="$CLOUDFLARE_ACCOUNT_ID" \
      npx --no-install wrangler tail mountain-inference
  )
}

main() {
  require_var CLOUDFLARE_API_TOKEN
  require_var CLOUDFLARE_ACCOUNT_ID
  require_var R2_ACCESS_KEY_ID
  require_var R2_SECRET_ACCESS_KEY

  command -v terraform >/dev/null 2>&1 || die "terraform not found in PATH"
  command -v npx >/dev/null 2>&1       || die "npx not found in PATH (install Node.js)"
  command -v curl >/dev/null 2>&1      || die "curl not found in PATH"

  local tag
  tag="$(resolve_image_tag)"
  log "Inference image tag: $tag"

  if [[ "${SKIP_IMAGE_WAIT:-}" != "1" ]]; then
    wait_for_image "$tag"
  fi

  run_terraform "$tag"

  log "Deploy complete."
  log "  Worker:    https://dash.cloudflare.com/?to=/:account/workers/services/view/mountain-inference"
  log "  Manual run: curl -X POST https://mountain-inference.<your-subdomain>.workers.dev/run"

  maybe_tail
}

main "$@"
