#!/usr/bin/env bash
# runtime/k8s/dedicated-tenant/render.sh — substitutes {{TENANT_ID}} and
# {{WORKER_IMAGE}} and prints the rendered manifests to stdout (or applies
# them directly with --apply). Same {{PLACEHOLDER}} sed-substitution
# convention as workflow-templates/ (see ai-tenant-init in install-ai-stack.sh).
#
# Usage:
#   ./render.sh <tenant-id> <worker-image> [--apply]

set -euo pipefail

TENANT_ID="${1:-}"
WORKER_IMAGE="${2:-}"
[ -n "$TENANT_ID" ] && [ -n "$WORKER_IMAGE" ] || {
  echo "❌ Usage: $0 <tenant-id> <worker-image> [--apply]"
  exit 1
}
# Both values land inside a sed replacement string with no escaping below —
# a tenant_id or image ref containing '/', '&', or other sed-significant
# characters would corrupt the substitution (or, for tenant_id, K8s namespace
# naming requires this pattern anyway) (Product_Archive.md 4.12).
if ! [[ "$TENANT_ID" =~ ^[a-z0-9-]+$ ]]; then
  echo "❌ <tenant-id> must match ^[a-z0-9-]+$ (Kubernetes namespace naming rules): got '$TENANT_ID'"
  exit 1
fi
if [[ "$WORKER_IMAGE" == *"|"* ]] || [[ "$WORKER_IMAGE" == *$'\n'* ]]; then
  echo "❌ <worker-image> must not contain '|' or newlines (sed delimiter used for this substitution): got '$WORKER_IMAGE'"
  exit 1
fi
APPLY="${3:-}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
RENDERED=$(mktemp)
trap 'rm -f "$RENDERED"' EXIT

for f in namespace.yaml configmap.yaml deployment.yaml resourcequota.yaml; do
  sed "s/{{TENANT_ID}}/${TENANT_ID}/g; s|{{WORKER_IMAGE}}|${WORKER_IMAGE}|g" "$DIR/$f" >> "$RENDERED"
  echo "---" >> "$RENDERED"
done

if [ "$APPLY" = "--apply" ]; then
  echo "🚀 Applying dedicated worker pool manifests for tenant '$TENANT_ID'..."
  kubectl apply -f "$RENDERED"
  echo ""
  echo "⚠️  secret.yaml.example was NOT applied — create agenticframework-secrets"
  echo "   in namespace tenant-${TENANT_ID} via your secrets manager."
else
  cat "$RENDERED"
fi
