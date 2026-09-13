# Dedicated Worker Pool (SPECS.md §23, §30)

Reference Kubernetes manifests for `tenant.isolation: dedicated` — a tenant
gets its own namespace, its own worker `Deployment`, and its own
budget-store credentials, instead of sharing the partitioned worker fleet
used by `isolation: shared` tenants.

## What "dedicated" actually isolates

| Concern | Mechanism here |
|---|---|
| Compute | Own `Namespace` + `Deployment`, own `ResourceQuota` |
| Budget / cost | Own `DATABASE_URL` Secret → own `llm_gateway_budget` table, never shared with other tenants |
| Traces | Own `AGENT_PHOENIX_ENDPOINT` (own Phoenix project), no cross-tenant query access |

This is a **pattern**, not a turnkey deploy. The framework provides the
manifest shape; tenant repos build their own worker image around
`runtime/worker.py` + their domain workflows/activities (§25 — framework
workflows are never deployed as tenant production code) and provision their
own secrets via their org's secrets manager.

## Usage

```bash
# Print rendered manifests without applying:
./render.sh acme my-registry/acme-worker:1.0.0

# Apply directly:
./render.sh acme my-registry/acme-worker:1.0.0 --apply

# secret.yaml.example is intentionally NOT applied by render.sh — create the
# real secret via your secrets manager:
kubectl create secret generic agenticframework-secrets -n tenant-acme \
  --from-literal=DATABASE_URL="postgresql://..." \
  --from-literal=ANTHROPIC_API_KEY="..." \
  --from-literal=OPENAI_API_KEY="..." \
  --from-literal=AGENT_OWNER_ID="..."
```

`ai-tenant-init <id> --isolation dedicated` (in `install-ai-stack.sh`) sets
`isolation: dedicated` in `.agenticframework/tenant.yaml` and prints this
`render.sh` command as the next step.

## Verified against a real cluster

These manifests were applied to a real `kind` cluster (not just YAML-linted)
during implementation:
- `kubectl apply` succeeds and creates the namespace, ConfigMap, Deployment,
  and ResourceQuota
- The Deployment correctly **refuses to start** (`CreateContainerConfigError:
  secret "agenticframework-secrets" not found`) until the per-tenant secret
  exists — it can't silently fall back to no config
- Once the secret is created, pods reach `Running` and `kubectl exec env`
  shows exactly this tenant's `TENANT_ID`, `AGENT_PHOENIX_ENDPOINT`, and
  `DATABASE_URL` — nothing from another tenant
- Two tenants applied side by side (`tenant-acme`, `tenant-globex`) are
  fully isolated: separate namespaces, separate ConfigMaps, separate
  Secrets, no shared state

## Files

| File | Purpose |
|---|---|
| `namespace.yaml` | Per-tenant `Namespace`, labelled `agenticframework.io/tenant-id` |
| `configmap.yaml` | Non-secret env: `TENANT_ID`, `AGENT_PHOENIX_ENDPOINT`, `BUDGET_BACKEND`, etc. |
| `secret.yaml.example` | Template only — apply via your secrets manager, never commit a filled-in copy |
| `deployment.yaml` | Worker `Deployment`, 2 replicas, resource requests/limits |
| `resourcequota.yaml` | Bounds this tenant's footprint if the cluster hosts multiple dedicated tenants |
| `render.sh` | `{{TENANT_ID}}` / `{{WORKER_IMAGE}}` substitution + optional `kubectl apply` |
