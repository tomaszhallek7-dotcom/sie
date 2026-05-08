use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use crate::server::AppState;
use crate::state::model_registry::ModelRegistry;
use crate::types::model::{ModelEntry, ModelInfoExtras};

#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "models",
    responses((status = 200, description = "Models visible to this gateway replica", body = crate::openapi::ModelsResponse))
)]
pub async fn get_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let model_workers =
        canonical_worker_models(state.registry.get_models().await, &state.model_registry);

    let model_names: BTreeSet<String> = state
        .model_registry
        .list_models()
        .into_iter()
        .chain(model_workers.keys().cloned())
        .collect();
    let models: Vec<serde_json::Value> = model_names
        .iter()
        .map(|name| {
            let worker_urls = model_workers.get(name).cloned().unwrap_or_default();
            let loaded = !worker_urls.is_empty();
            match state.model_registry.get_model_info(name) {
                Some(entry) => entry.to_model_info_value(loaded),
                None => worker_only_model_info(name, loaded),
            }
        })
        .collect();

    (StatusCode::OK, Json(json!({"models": models}))).into_response()
}

/// Detail counterpart to `get_models`.
#[utoipa::path(
    get,
    path = "/v1/models/{model}",
    tag = "models",
    params(("model" = String, Path, description = "Model id; percent-encode slashes when using OpenAPI-generated clients")),
    responses(
        (status = 200, description = "Model detail", body = crate::openapi::ModelInfoWire),
        (status = 404, description = "Model not found", body = crate::openapi::ModelNotFoundResponse)
    )
)]
pub async fn get_model(Path(model): Path<String>, State(state): State<Arc<AppState>>) -> Response {
    let model_entry = state.model_registry.get_model_info(&model);
    let known_in_registry = model_entry.is_some();
    let canonical_model = model_entry
        .as_ref()
        .map(|entry| entry.name.as_str())
        .unwrap_or(model.as_str());
    let bundles = state.model_registry.get_model_bundles(&model);
    let model_workers =
        canonical_worker_models(state.registry.get_models().await, &state.model_registry);
    let worker_urls = model_workers
        .get(canonical_model)
        .cloned()
        .unwrap_or_default();

    if !known_in_registry && bundles.is_empty() && worker_urls.is_empty() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "detail": {
                    "code": "MODEL_NOT_FOUND",
                    "message": format!("Model '{}' not found", model),
                }
            })),
        )
            .into_response();
    }

    let loaded = !worker_urls.is_empty();
    let body = match model_entry {
        Some(entry) => entry.to_model_info_value(loaded),
        None => worker_only_model_info(&model, loaded),
    };

    (StatusCode::OK, Json(body)).into_response()
}

fn canonical_worker_models(
    model_workers: HashMap<String, Vec<String>>,
    model_registry: &ModelRegistry,
) -> HashMap<String, Vec<String>> {
    let mut normalized: HashMap<String, Vec<String>> = HashMap::new();
    for (model, urls) in model_workers {
        let canonical = model_registry
            .get_model_info(&model)
            .map(|entry| entry.name)
            .unwrap_or(model);
        normalized.entry(canonical).or_default().extend(urls);
    }
    normalized
}

/// ``ModelInfo``-shaped JSON when workers advertise a model id that is not
/// present in the local registry snapshot (bootstrap / race window).
fn worker_only_model_info(name: &str, loaded: bool) -> Value {
    ModelEntry {
        name: name.to_string(),
        bundles: Vec::new(),
        adapter_modules: HashSet::new(),
        profile_names: HashSet::new(),
        profile_configs: HashMap::new(),
        info_extras: ModelInfoExtras {
            inputs: Vec::new(),
            outputs: Vec::new(),
            dims: HashMap::new(),
            max_sequence_length: None,
        },
    }
    .to_model_info_value(loaded)
}

pub fn extract_bearer_token(headers: &HeaderMap) -> Option<String> {
    let header = headers
        .get("authorization")?
        .to_str()
        .ok()?
        .trim()
        .to_string();
    if header.is_empty() {
        return None;
    }
    let token = if header.to_lowercase().starts_with("bearer ") {
        header[7..].trim().to_string()
    } else {
        header
    };
    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

pub fn mask_token(token: &str) -> String {
    if token.len() <= 4 {
        "****".to_string()
    } else {
        format!(
            "{}{}",
            "*".repeat(token.len() - 4),
            &token[token.len() - 4..]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_bearer_token ───────────────────────────────────────

    #[test]
    fn test_extract_bearer_token_with_prefix() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "Bearer my-token-123".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), Some("my-token-123".into()));
    }

    #[test]
    fn test_extract_bearer_token_case_insensitive_prefix() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "bearer my-token".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), Some("my-token".into()));
    }

    #[test]
    fn test_extract_bearer_token_without_prefix() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "raw-token-value".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), Some("raw-token-value".into()));
    }

    #[test]
    fn test_extract_bearer_token_missing_header() {
        let h = HeaderMap::new();
        assert_eq!(extract_bearer_token(&h), None);
    }

    #[test]
    fn test_extract_bearer_token_empty_value() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), None);
    }

    #[test]
    fn test_extract_bearer_token_bearer_only() {
        // "Bearer " trims to "Bearer", which doesn't start with "bearer " (missing trailing space),
        // so it's treated as a raw token value.
        let mut h = HeaderMap::new();
        h.insert("authorization", "Bearer ".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), Some("Bearer".into()));
    }

    #[test]
    fn test_extract_bearer_token_whitespace_trimmed() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "  Bearer  my-token  ".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), Some("my-token".into()));
    }

    // ── mask_token ─────────────────────────────────────────────────

    #[test]
    fn test_mask_token_long() {
        assert_eq!(mask_token("secret-token-123"), "************-123");
    }

    #[test]
    fn test_mask_token_short() {
        assert_eq!(mask_token("abc"), "****");
        assert_eq!(mask_token(""), "****");
    }

    #[test]
    fn test_mask_token_exactly_4() {
        assert_eq!(mask_token("abcd"), "****");
    }

    #[test]
    fn test_mask_token_5_chars() {
        assert_eq!(mask_token("12345"), "*2345");
    }
}

#[cfg(test)]
mod route_tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::response::Response;
    use axum::Router;
    use tower::ServiceExt;

    use crate::config::Config;
    use crate::server::{create_router, AppState};
    use crate::state::demand_tracker::DemandTracker;
    use crate::state::model_registry::ModelRegistry;
    use crate::state::pool_manager::PoolManager;
    use crate::state::worker_registry::WorkerRegistry;
    use crate::types::model::{ModelConfig, ProfileConfig};
    use crate::types::worker::WorkerStatusMessage;

    fn test_config(bundles_dir: &str, models_dir: &str) -> Config {
        Config {
            host: "127.0.0.1".to_string(),
            port: 0,
            worker_urls: Vec::new(),
            use_kubernetes: false,
            k8s_namespace: "default".to_string(),
            k8s_service: "sie-worker".to_string(),
            k8s_port: 8080,
            health_mode: "ws".to_string(),
            nats_url: String::new(),
            nats_config_trusted_producers: vec!["sie-config".to_string()],
            auth_mode: "none".to_string(),
            auth_tokens: Vec::new(),
            admin_token: String::new(),
            auth_exempt_operational: false,
            log_level: "info".to_string(),
            json_logs: false,
            enable_pools: false,
            hot_reload: false,
            watch_polling: false,
            multi_router: false,
            request_timeout: 30.0,
            max_stream_pending: 50_000,
            configured_gpus: Vec::new(),
            gpu_profile_map: HashMap::new(),
            bundles_dir: bundles_dir.to_string(),
            models_dir: models_dir.to_string(),
            payload_store_url: String::new(),
            config_service_url: None,
            config_service_token: None,
        }
    }

    // Returned tempdirs must outlive the router so they drop after the test.
    async fn build_router_with_state(
    ) -> (Router, Arc<AppState>, tempfile::TempDir, tempfile::TempDir) {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            bundles_dir.path().join("default.yaml"),
            "name: default\nadapters:\n  - module\ndefault: true\n",
        )
        .unwrap();

        let config = Arc::new(test_config(
            bundles_dir.path().to_str().unwrap(),
            models_dir.path().to_str().unwrap(),
        ));
        let model_registry = Arc::new(ModelRegistry::new(
            bundles_dir.path(),
            models_dir.path(),
            true,
        ));
        let state = Arc::new(AppState {
            registry: Arc::new(WorkerRegistry::new(Duration::from_secs(30), None)),
            config: Arc::clone(&config),
            model_registry,
            pool_manager: Arc::new(PoolManager::new(Vec::new())),
            work_publisher: None,
            demand_tracker: Arc::new(DemandTracker::new()),
            config_epoch: crate::state::config_epoch::ConfigEpoch::new(),
        });
        let router = create_router(Arc::clone(&state), config);
        (router, state, bundles_dir, models_dir)
    }

    fn seed_model(state: &AppState, model_id: &str) {
        let mut profiles = HashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        state
            .model_registry
            .add_model_config(ModelConfig {
                name: model_id.to_string(),
                adapter_module: None,
                default_bundle: None,
                profiles,
                inputs: None,
                tasks: None,
                max_sequence_length: None,
            })
            .unwrap();
    }

    fn worker_msg(name: &str, loaded_models: Vec<String>) -> WorkerStatusMessage {
        WorkerStatusMessage {
            name: name.into(),
            gpu_count: 1,
            machine_profile: "A100".into(),
            bundle: "default".into(),
            bundle_config_hash: String::new(),
            ready: true,
            loaded_models,
            queue_depth: Some(0),
            models: Vec::new(),
            memory_used_bytes: Some(0),
            memory_total_bytes: Some(0),
            gpus: Vec::new(),
            pool_name: String::new(),
        }
    }

    async fn body_json(response: Response) -> serde_json::Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn test_list_models_returns_seeded_model() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let models = body["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"], "BAAI/bge-m3");
        assert_eq!(models[0]["loaded"], false);
    }

    #[tokio::test]
    async fn test_list_models_includes_registry_and_worker_only_models() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");
        state
            .registry
            .update_worker(
                "http://worker-1:8080",
                worker_msg("worker-1", vec!["worker/only".to_string()]),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let models = body["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);

        let by_name: HashMap<_, _> = models
            .iter()
            .map(|model| (model["name"].as_str().unwrap(), model))
            .collect();
        assert_eq!(by_name["BAAI/bge-m3"]["loaded"], false);
        assert_eq!(by_name["worker/only"]["loaded"], true);
        assert!(by_name["worker/only"]["inputs"]
            .as_array()
            .unwrap()
            .is_empty());
        assert!(by_name["worker/only"]["outputs"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn test_list_models_canonicalizes_worker_model_names() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");
        state
            .registry
            .update_worker(
                "http://worker-1:8080",
                worker_msg("worker-1", vec!["baai/BGE-M3".to_string()]),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let models = body["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"], "BAAI/bge-m3");
        assert_eq!(models[0]["loaded"], true);
        assert_eq!(models[0]["state"], "loaded");
    }

    #[tokio::test]
    async fn test_get_model_detail_known_model_no_workers() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/BAAI/bge-m3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["name"], "BAAI/bge-m3");
        assert_eq!(body["loaded"], false);
        assert_eq!(body["state"], "available");
        assert!(!body["inputs"].as_array().unwrap().is_empty());
        assert!(body["profiles"].is_object());
    }

    #[tokio::test]
    async fn test_get_model_detail_with_live_worker_reports_loaded() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");
        state
            .registry
            .update_worker(
                "http://worker-1:8080",
                worker_msg("worker-1", vec!["BAAI/bge-m3".to_string()]),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/BAAI/bge-m3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["name"], "BAAI/bge-m3");
        assert_eq!(body["loaded"], true);
        assert_eq!(body["state"], "loaded");
    }

    #[tokio::test]
    async fn test_get_model_detail_canonicalizes_worker_model_names() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");
        state
            .registry
            .update_worker(
                "http://worker-1:8080",
                worker_msg("worker-1", vec!["baai/BGE-M3".to_string()]),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/baai/bge-m3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["name"], "BAAI/bge-m3");
        assert_eq!(body["loaded"], true);
        assert_eq!(body["state"], "loaded");
    }

    #[tokio::test]
    async fn test_get_model_detail_unknown_returns_404() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/does/not/exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = body_json(response).await;
        assert_eq!(body["detail"]["code"], "MODEL_NOT_FOUND");
        assert!(body["detail"]["message"]
            .as_str()
            .unwrap()
            .contains("does/not/exist"));
    }

    #[tokio::test]
    async fn test_get_model_detail_unknown_when_registry_empty_returns_404() {
        let (app, _state, _bundles_dir, _models_dir) = build_router_with_state().await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/anything")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = body_json(response).await;
        assert_eq!(body["detail"]["code"], "MODEL_NOT_FOUND");
        assert!(body["detail"]["message"]
            .as_str()
            .unwrap()
            .contains("anything"));
    }
}
