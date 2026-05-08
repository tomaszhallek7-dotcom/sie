# SIE Cluster Helm Chart

Deploy SIE (Search Inference Engine) to Kubernetes with autoscaling and observability.

## Quick Start

```bash
helm install sie-cluster oci://ghcr.io/superlinked/charts/sie-cluster \
   --namespace sie \
   --create-namespace
  --namespace sie \
  --create-namespace
```

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────┐     ┌──────────────┐
│   Client    │────▶│      Gateway (1 replica; 2+ for HA)  │◀───▶│  sie-config  │
└─────────────┘     └───────────────┬─────────────────────┘     │ (singleton)  │
                                    │                           └──────────────┘
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌─────────┐     ┌─────────┐     ┌─────────┐
              │ L4 Pool │     │A100 Pool│     │ CPU Pool│
              │ 0-N     │     │ 0-N     │     │ 0-N     │
              └─────────┘     └─────────┘     └─────────┘
```

- **Gateway**: Stateless request proxy that routes to workers based on GPU type and model affinity. Consumes config via GET/NATS from `sie-config`.
- **sie-config**: Authoritative control plane for model/bundle configuration. Serves `/v1/configs/*` writes and publishes NATS deltas to the gateway and workers. Deployed as a singleton (`replicas: 1`, `strategy: Recreate`).
- **Worker Pools**: StatefulSets per GPU type, each with KEDA autoscaling.

## Cold Start Expectations

When scaling from zero, expect the following latencies:

| Phase | Duration | Notes |
|-------|----------|-------|
| **Node provisioning** | 2-5 min | GKE/EKS spins up GPU node (spot may be slower) |
| **Container startup** | 20-40s | Pull image, start process, health checks |
| **Model loading** | 10-120s | Download weights (if not cached), load to GPU |
| **Total cold start** | 3-7 min | First request to a scaled-to-zero pool |

### Reducing Cold Start Time

1. **Use cluster cache**: Pre-populate S3/GCS with model weights (`--cluster-cache`)
2. **Set minReplicas=1**: Keep one warm replica per critical GPU type
3. **Use reserved capacity**: Avoid spot for latency-sensitive workloads
4. **Pre-warm models**: Call `/v1/encode/{model}` on startup to load weights

### Client Handling

When a pool is scaling from zero, the gateway returns:
- **202 Accepted** with `Retry-After: 120` header
- Client should retry after the indicated delay

The SDK handles this automatically with configurable retries.

## Cluster model cache (S3/GCS)

Pre-populate a shared bucket with model weights so worker pods don't re-download from HuggingFace on every cold start. The Python SDK pulls from the bucket first and falls back to HF on miss.

**AWS (Terraform-managed bucket):**

```bash
# 1. Provision the bucket via the AWS Terraform module (opt-in via create_model_cache=true)
cd deploy/terraform/aws/examples/dev-g6-spot
terraform apply

# 2. One-time populate from your laptop
sie-admin cache weights sync --bundle default \
  --dest $(terraform output -raw model_cache_bucket_url)/

# 3. Wire into Helm
helm upgrade --install sie-cluster . \
  --set workers.common.clusterCache.enabled=true \
  --set workers.common.clusterCache.url=$(terraform output -raw model_cache_bucket_url)
```

The Terraform output already includes the `/models` prefix, so the same URL is used for both `sie-admin --target` and `clusterCache.url`.

**Other clouds / BYO bucket:** point `workers.common.clusterCache.url` at any `s3://...` or `gs://...` URL the workload Service Account can read; populate it with the same `sie-admin cache weights sync --dest ...` command.

## Autoscaling

KEDA-based autoscaling with scale-to-zero support:

```yaml
autoscaling:
  enabled: true
  # Scale-to-zero after 10 min idle
  cooldownPeriod: 600
  # Check metrics every 15s
  pollingInterval: 15
```

### Scale-from-Zero Trigger

The gateway exposes `sie_gateway_pending_demand{gpu="..."}` metric when requests
arrive for GPU types with no available workers. KEDA uses this to trigger scale-up
even when there are 0 workers (and thus no worker metrics).

### Scaling Metrics

| Metric | Source | Purpose |
|--------|--------|---------|
| `sie_gateway_pending_demand` | Gateway | Trigger scale from 0 |
| `sie_request_queue_depth` | Workers | Scale up on load |
| `sie_active_requests` | Workers | Scale up on concurrent requests |

## Configuration

See `values.yaml` for all options. Key settings:

**Important**: All worker pools are disabled by default. You must explicitly enable
the pools you need in your values override.

```yaml
# Worker pool configuration (must explicitly enable pools)
# Pool naming: <machineProfile> (e.g. l4, a100-40gb, cpu)
workers:
  pools:
    l4:
      enabled: true     # Enable this pool (disabled by default)
      minReplicas: 0    # Scale to zero
      maxReplicas: 10

# Gateway configuration
gateway:
  replicaCount: 2  # HA by default

# Autoscaling
autoscaling:
  enabled: true
  cooldownPeriod: 600  # 10 min before scale-down
```

## TLS / HTTPS

The chart supports two TLS modes for the Ingress (set via `ingress.tls.mode`):

- `byo` — bring your own `kubernetes.io/tls` Secret (default, backward compatible).
- `cert-manager` — chart annotates the Ingress; [cert-manager](https://cert-manager.io/) provisions and renews the certificate via ACME (HTTP-01 challenge to Let's Encrypt by default).

Only HTTP-01 ACME challenges are supported by the chart. DNS-01 / wildcard certs (which require cloud-provider IRSA / Workload Identity for Route53 / Cloud DNS) are out of scope — set them up manually outside the chart and reference the resulting Secret via `mode: byo`.

### BYO certificate

Create the TLS Secret yourself (e.g. from a corporate CA, ACM cert exported to a Secret, or an existing wildcard cert), then point the chart at it:

```bash
kubectl -n sie create secret tls sie-tls --cert=path/to/tls.crt --key=path/to/tls.key
```

```yaml
ingress:
  enabled: true
  className: nginx
  host: sie.example.com
  tls:
    enabled: true
    mode: byo            # default
    secretName: sie-tls  # default
```

### cert-manager + Let's Encrypt

Prerequisite: install cert-manager once in the cluster (its CRDs are cluster-scoped and must exist exactly once — that's why it is **not** bundled as a subchart):

```bash
helm repo add jetstack https://charts.jetstack.io && helm repo update
helm install cert-manager jetstack/cert-manager \
  --set crds.enabled=true -n cert-manager --create-namespace
```

Then enable cert-manager mode in your SIE values:

```yaml
ingress:
  enabled: true
  className: nginx
  host: sie.example.com
  tls:
    enabled: true
    mode: cert-manager
    certManager:
      email: ops@example.com
      # Use Let's Encrypt staging while iterating to avoid the 50 new-cert/registered-domain/week prod limit (duplicate-cert limit is 5/week):
      # server: https://acme-staging-v02.api.letsencrypt.org/directory
      kind: ClusterIssuer  # cluster-scoped; share across namespaces. Use "Issuer" for namespace-scoped.
      create: true         # chart renders the Issuer/ClusterIssuer
```

The chart renders a `{kind}` named `{release-fullname}-letsencrypt-prod` (release-scoped to avoid collisions when multiple SIE releases share a cluster) and adds the appropriate `cert-manager.io/cluster-issuer` (or `/issuer`) annotation to the main Ingress. cert-manager populates `ingress.tls.secretName` (default `sie-tls`); the same Secret is referenced by the oauth2-proxy Ingress when auth is enabled.

Note: Helm's standard `fullname` collapses when the release name already contains the chart name, so `helm install sie-cluster …` produces `sie-cluster-letsencrypt-prod` (not `sie-cluster-sie-cluster-letsencrypt-prod`). If you override `certManager.name`, set the full intended name explicitly rather than expecting a particular default.

Issuer kind tradeoff:

- `ClusterIssuer` — single ACME account / private key shared across all namespaces. Best for shared clusters.
- `Issuer` — namespace-scoped. Use for hard tenant isolation, or when you don't have permission to create cluster-scoped resources.

### Reusing an existing ClusterIssuer/Issuer

In multi-tenant clusters where a platform team already manages a shared `ClusterIssuer`, set `create: false` and reference it by name:

```yaml
ingress:
  tls:
    enabled: true
    mode: cert-manager
    certManager:
      kind: ClusterIssuer
      create: false
      name: platform-letsencrypt-prod
```

The chart only adds the annotation — it does not render any Issuer resource.

## Gated Models

Some HuggingFace models require authentication to download (gated models). Examples:

- `google/embeddinggemma-300m` - Manual gating (requires approval)
- `naver/splade-v3` - Auto gating (requires license acceptance)

### Prerequisites

1. Create a HuggingFace account and generate an access token at <https://huggingface.co/settings/tokens>
2. For manually gated models, request access on the model page (e.g., <https://huggingface.co/google/embeddinggemma-300m>)
3. For auto-gated models, accept the license agreement on the model page

### Kubernetes Setup

Create a secret with your HuggingFace token:

```bash
kubectl create secret generic hf-token \
  --namespace sie \
  --from-literal=token=hf_your_token_here
```

Configure the Helm chart to use the secret:

```yaml
workers:
  common:
    hfCache:
      tokenSecret: hf-token      # Secret name
      tokenSecretKey: token      # Key within the secret
```

The token is mounted as the `HF_TOKEN` environment variable, which HuggingFace libraries automatically detect.

### Local Development

For local development, set the `HF_TOKEN` environment variable:

```bash
# Option 1: Direct export
export HF_TOKEN=hf_your_token_here
mise run serve

# Option 2: From file
export HF_TOKEN=$(cat ~/.secrets/hf_token)
mise run serve
```

### Docker

Pass the token as an environment variable:

```bash
docker run -e HF_TOKEN=hf_your_token_here \
  -p 8080:8080 \
  sie-server:cuda12-default
```

## Telemetry

SIE collects anonymous usage telemetry (version, OS, architecture, GPU type) to help maintainers understand adoption and hardware distribution. Telemetry is on by default and sends a lightweight heartbeat once per hour.

**No IP addresses, hostnames, cluster names, API keys, or request data are collected.**

Disable telemetry:

```yaml
telemetry:
  enabled: false
```

Enterprise customers can route heartbeats through their own collector:

```yaml
telemetry:
  url: "https://telemetry.internal.example.com/api/telemetry"
```

Tag non-production deployments to filter them out of dashboards:

```yaml
telemetry:
  deploymentEnv: staging  # production (default) | staging | development | ci
```

> **Internal Superlinked clusters:** any cluster owned by Superlinked that is
> not a customer-facing production install MUST set `telemetry.deploymentEnv`
> to one of `staging | development | ci`. The chart default is `production`
> so that customer Helm installs are correctly tagged out of the box; internal
> stacks must opt out explicitly to keep them out of the production telemetry
> dashboards. See `deploy/terraform/{aws,gcp}/internal-examples/` for the
> per-cluster mapping.

## Observability

Observability components (Prometheus, Grafana, Loki, DCGM Exporter, Alloy, Event Exporter) are included as optional sub-chart dependencies. Enable them in your values overlay (e.g. `kube-prometheus-stack.install: true`).

Pre-configured dashboards:

- Cluster overview (QPS, latency, GPU utilization)
- Per-model performance
- Worker health
