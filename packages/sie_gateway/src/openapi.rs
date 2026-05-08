use axum::http::header;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use std::path::Path;
use std::sync::LazyLock;
use utoipa::{OpenApi, ToSchema};

static OPENAPI_DOC: LazyLock<utoipa::openapi::OpenApi> = LazyLock::new(|| {
    let mut doc = ApiDoc::openapi();
    doc.info.version = env!("CARGO_PKG_VERSION").to_string();
    doc
});

static OPENAPI_JSON: LazyLock<String> = LazyLock::new(|| {
    let mut value = serde_json::to_value(&*OPENAPI_DOC).expect("OpenAPI document should serialize");
    apply_gateway_openapi_overrides(&mut value);
    serde_json::to_string_pretty(&value).expect("OpenAPI document should serialize") + "\n"
});

#[derive(OpenApi)]
#[openapi(
    info(
        title = "SIE Gateway",
        description = "Rust gateway API for SIE inference, pool coordination, and read-only runtime config.",
        version = "0.0.0"
    ),
    paths(
        crate::handlers::health::status_page,
        openapi_json,
        crate::handlers::health::healthz,
        crate::handlers::health::readyz,
        crate::handlers::health::health,
        crate::handlers::health::ws_cluster_status,
        crate::handlers::models::get_models,
        crate::handlers::models::get_model,
        crate::handlers::pools::create_pool,
        crate::handlers::pools::list_pools,
        crate::handlers::pools::get_pool,
        crate::handlers::pools::delete_pool,
        crate::handlers::pools::renew_pool,
        crate::handlers::config_api::get_model_configs,
        crate::handlers::config_api::get_model_config_or_status,
        get_model_config_status_doc,
        crate::handlers::config_api::get_bundle_configs,
        crate::handlers::config_api::get_bundle_config,
        crate::handlers::config_api::resolve_config,
        crate::handlers::proxy::proxy_encode,
        crate::handlers::proxy::proxy_openai_embeddings,
        crate::handlers::proxy::proxy_score,
        crate::handlers::proxy::proxy_extract,
        crate::server::metrics_handler
    ),
    components(schemas(
        BundleConfigDocument,
        BundleConfigSummary,
        BundleConfigsResponse,
        ClusterSummary,
        ConfigModelDocument,
        ConfigModelSummary,
        ConfigModelsResponse,
        crate::handlers::pools::CreatePoolRequest,
        ErrorDetailCore,
        StandardApiError,
        GpuNotConfiguredDetail,
        GpuNotConfiguredError,
        GatewayModelLoadFailedDetail,
        GatewayModelLoadFailedResponse,
        GatewayErrorResponse,
        InferenceInternalServerErrorResponse,
        InferenceServiceUnavailableResponse,
        AllItemsFailedResponse,
        BundleRoutingConflictDetail,
        BundleConflictResponse,
        DocumentInput,
        DenseVector,
        EncodeParams,
        EncodeRequest,
        EncodeResponse,
        EncodeResult,
        Entity,
        ExtractParams,
        ExtractRequest,
        ExtractResponse,
        ExtractResult,
        ImageInput,
        HealthResponse,
        ItemInput,
        InferenceErrorDetail,
        MessageResponse,
        ModelAckBundleStatus,
        ModelConfigStatusResponse,
        ModelInfoWire,
        ModelNotFoundDetail,
        ModelNotFoundResponse,
        ProfileInfoWire,
        crate::types::worker::ModelInfo,
        MultiVector,
        OpenAIEmbeddingDataEntry,
        OpenAIEmbeddingEncodingFormat,
        OpenAIEmbeddingInput,
        OpenAIEmbeddingRequest,
        OpenAIEmbeddingUsage,
        OpenAIEmbeddingVector,
        OpenAIEmbeddingsListResponse,
        PoolListResponse,
        crate::types::pool::PoolSpec,
        crate::types::pool::PoolStatus,
        crate::types::pool::PoolState,
        ProvisioningResponse,
        Relation,
        ResolveBundleConflictDetail,
        ResolveBundleConflictResponse,
        ResolveModelNotFoundDetail,
        ResolveModelNotFoundResponse,
        ResolveConfigResponse,
        ScoreEntry,
        crate::handlers::config_api::ResolveRequest,
        ScoreRequest,
        ScoreResponse,
        SparseVector,
        TimingInfo,
        crate::types::pool::AssignedWorker,
        crate::types::worker::WorkerInfo
    )),
    tags(
        (name = "health", description = "Gateway health, readiness, and status surfaces"),
        (name = "inference", description = "Queue-backed inference entrypoints"),
        (name = "models", description = "Models visible to this gateway replica"),
        (name = "pools", description = "Runtime pool coordination"),
        (name = "config", description = "Read-only gateway view of model and bundle config"),
        (name = "observability", description = "Metrics and streaming status"),
        (name = "docs", description = "API description")
    )
)]
pub struct ApiDoc;

#[utoipa::path(
    get,
    path = "/openapi.json",
    tag = "docs",
    responses((status = 200, description = "OpenAPI document"))
)]
pub async fn openapi_json() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/json")],
        OPENAPI_JSON.as_str(),
    )
}

#[cfg(test)]
fn openapi_document() -> utoipa::openapi::OpenApi {
    OPENAPI_DOC.clone()
}

fn apply_gateway_openapi_overrides(value: &mut Value) {
    value["info"]["license"] = json!({
        "name": "Apache-2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    });
    value["security"] = json!([{ "bearerAuth": [] }]);

    let components = value
        .as_object_mut()
        .expect("OpenAPI root should be an object")
        .entry("components")
        .or_insert_with(|| json!({}));
    components["securitySchemes"] = json!({
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "opaque",
            "description": "SIE bearer token. Mutating pool/config/admin routes require the configured admin token; other protected routes accept a normal gateway token."
        }
    });

    if let Some(create_pool) = value
        .get_mut("components")
        .and_then(|components| components.get_mut("schemas"))
        .and_then(|schemas| schemas.get_mut("CreatePoolRequest"))
    {
        create_pool["required"] = json!(["name", "gpus"]);
        if let Some(gpus) = create_pool
            .get_mut("properties")
            .and_then(|properties| properties.get_mut("gpus"))
        {
            gpus["minProperties"] = json!(1);
        }
    }

    let paths = value["paths"]
        .as_object_mut()
        .expect("OpenAPI paths should be an object");

    for path in ["/healthz", "/readyz", "/openapi.json"] {
        if let Some(operation) = paths
            .get_mut(path)
            .and_then(|path_item| path_item.get_mut("get"))
        {
            operation["security"] = json!([]);
        }
    }

    if let Some(response) = paths
        .get_mut("/openapi.json")
        .and_then(|path_item| path_item.get_mut("get"))
        .and_then(|operation| operation.get_mut("responses"))
        .and_then(|responses| responses.get_mut("200"))
    {
        response["content"] = json!({
            "application/json": {
                "schema": {
                    "type": "object"
                }
            }
        });
    }

    inject_inference_msgpack_content(paths);
    inject_inference_response_headers(paths);
    inject_bearer_auth_error_responses(paths);
    annotate_slash_bearing_path_parameters(paths);
}

/// Duplicate ``application/json`` schemas to the msgpack media types accepted by queue inference.
fn inject_inference_msgpack_content(paths: &mut serde_json::Map<String, Value>) {
    for path in [
        "/v1/encode/{model}",
        "/v1/score/{model}",
        "/v1/extract/{model}",
    ] {
        let Some(path_item) = paths.get_mut(path) else {
            continue;
        };
        let Some(post) = path_item.get_mut("post").and_then(|p| p.as_object_mut()) else {
            continue;
        };
        if let Some(rb) = post.get_mut("requestBody").and_then(|r| r.as_object_mut()) {
            if let Some(content) = rb.get_mut("content").and_then(|c| c.as_object_mut()) {
                if let Some(json_entry) = content.get("application/json").cloned() {
                    for media_type in [
                        "application/msgpack",
                        "application/x-msgpack",
                        "application/vnd.msgpack",
                    ] {
                        content.insert(media_type.to_string(), json_entry.clone());
                    }
                }
            }
        }
        let Some(responses) = post.get_mut("responses").and_then(|r| r.as_object_mut()) else {
            continue;
        };
        let Some(ok) = responses.get_mut("200").and_then(|x| x.as_object_mut()) else {
            continue;
        };
        let Some(content) = ok.get_mut("content").and_then(|c| c.as_object_mut()) else {
            continue;
        };
        if let Some(json_entry) = content.get("application/json").cloned() {
            for media_type in [
                "application/msgpack",
                "application/x-msgpack",
                "application/vnd.msgpack",
            ] {
                content.insert(media_type.to_string(), json_entry.clone());
            }
        }
    }
}

fn header(description: &str) -> Value {
    json!({
        "description": description,
        "schema": { "type": "string" }
    })
}

fn merge_headers(response: &mut Value, headers: &Value) {
    let Some(response) = response.as_object_mut() else {
        return;
    };
    let Some(header_map) = headers.as_object() else {
        return;
    };
    let headers = response
        .entry("headers")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("OpenAPI response headers should be an object");
    for (name, value) in header_map {
        headers.entry(name.clone()).or_insert(value.clone());
    }
}

fn inject_inference_response_headers(paths: &mut serde_json::Map<String, Value>) {
    let queue_success_headers = json!({
        "X-SIE-Version": header("Gateway package version that handled the request"),
        "X-SIE-Server-Version": header("Gateway-compatible server version advertised by this gateway"),
        "X-SIE-Request-Id": header("Gateway request id for queue-backed inference"),
        "X-SIE-Worker": header("Logical queue worker tag that produced the response"),
        "X-Queue-Publish-Time": header("Milliseconds spent publishing work to the queue"),
        "X-Queue-Wait-Time": header("Milliseconds spent waiting for worker results"),
        "X-Queue-Time": header("Worker-reported queue time in milliseconds"),
        "X-Inference-Time": header("Worker-reported inference time in milliseconds"),
        "X-Tokenization-Time": header("Worker-reported tokenization time in milliseconds, when available"),
        "X-Postprocessing-Time": header("Worker-reported postprocessing time in milliseconds, when available"),
        "X-Payload-Fetch-Time": header("Worker-reported offloaded payload fetch time in milliseconds, when available")
    });
    let provisioning_headers = json!({
        "Retry-After": header("Suggested retry delay in seconds"),
        "X-SIE-Version": header("Gateway package version that handled the request"),
        "X-SIE-Server-Version": header("Gateway-compatible server version advertised by this gateway")
    });
    let retryable_error_headers = json!({
        "Retry-After": header("Suggested retry delay in seconds, when retryable"),
        "X-SIE-Error-Code": header("SDK-stable gateway or worker error code"),
        "X-SIE-Version": header("Gateway package version that handled the request"),
        "X-SIE-Server-Version": header("Gateway-compatible server version advertised by this gateway")
    });
    let terminal_error_headers = json!({
        "X-SIE-Error-Code": header("SDK-stable gateway or worker error code"),
        "X-SIE-Error-Version": header("Gateway package version associated with the error envelope"),
        "X-SIE-Version": header("Gateway package version that handled the request"),
        "X-SIE-Server-Version": header("Gateway-compatible server version advertised by this gateway")
    });

    for path in [
        "/v1/encode/{model}",
        "/v1/score/{model}",
        "/v1/extract/{model}",
    ] {
        let Some(responses) = paths
            .get_mut(path)
            .and_then(|path_item| path_item.get_mut("post"))
            .and_then(|post| post.get_mut("responses"))
            .and_then(|responses| responses.as_object_mut())
        else {
            continue;
        };
        if let Some(response) = responses.get_mut("200") {
            merge_headers(response, &queue_success_headers);
        }
        if let Some(response) = responses.get_mut("202") {
            merge_headers(response, &provisioning_headers);
        }
        if let Some(response) = responses.get_mut("502") {
            merge_headers(response, &terminal_error_headers);
        }
        if let Some(response) = responses.get_mut("503") {
            merge_headers(response, &retryable_error_headers);
        }
    }

    if let Some(responses) = paths
        .get_mut("/v1/embeddings")
        .and_then(|path_item| path_item.get_mut("post"))
        .and_then(|post| post.get_mut("responses"))
        .and_then(|responses| responses.as_object_mut())
    {
        if let Some(response) = responses.get_mut("200") {
            merge_headers(response, &queue_success_headers);
        }
        if let Some(response) = responses.get_mut("202") {
            merge_headers(response, &provisioning_headers);
        }
        if let Some(response) = responses.get_mut("502") {
            merge_headers(response, &terminal_error_headers);
        }
        if let Some(response) = responses.get_mut("503") {
            merge_headers(response, &retryable_error_headers);
        }
    }
}

/// Add documented auth responses for bearer-protected operations (spec completeness).
fn inject_bearer_auth_error_responses(paths: &mut serde_json::Map<String, Value>) {
    let unauthorized = json!({
        "description": "Missing or invalid bearer token (inference token)",
        "content": {
            "application/json": {
                "schema": { "$ref": "#/components/schemas/StandardApiError" }
            }
        }
    });
    let forbidden = json!({
        "description": "Valid bearer token but admin token required for this mutation (or admin token not configured)",
        "content": {
            "application/json": {
                "schema": { "$ref": "#/components/schemas/StandardApiError" }
            }
        }
    });
    let auth_misconfigured = json!({
        "description": "Gateway auth enabled but no tokens configured",
        "content": {
            "application/json": {
                "schema": { "$ref": "#/components/schemas/StandardApiError" }
            }
        }
    });

    for (path, path_item) in paths.iter_mut() {
        let Some(path_item) = path_item.as_object_mut() else {
            continue;
        };
        for (method, op) in path_item.iter_mut() {
            if !matches!(method.as_str(), "get" | "post" | "put" | "delete" | "patch") {
                continue;
            }
            let Some(op_ob) = op.as_object_mut() else {
                continue;
            };
            let secured = op_ob
                .get("security")
                .and_then(|s| s.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(true);
            if !secured {
                continue;
            }
            let Some(responses) = op_ob.get_mut("responses").and_then(|r| r.as_object_mut()) else {
                continue;
            };
            responses
                .entry("401".to_string())
                .or_insert(unauthorized.clone());
            merge_auth_misconfigured_response(responses, &auth_misconfigured);

            if admin_mutation(method.as_str(), path) {
                responses
                    .entry("403".to_string())
                    .or_insert(forbidden.clone());
            }
        }
    }
}

fn standard_api_error_schema() -> Value {
    json!({ "$ref": "#/components/schemas/StandardApiError" })
}

fn schema_allows_standard_api_error(schema: &Value) -> bool {
    if schema == &standard_api_error_schema() {
        return true;
    }
    schema
        .get("oneOf")
        .and_then(|one_of| one_of.as_array())
        .map(|schemas| schemas.iter().any(schema_allows_standard_api_error))
        .unwrap_or(false)
}

fn merge_auth_misconfigured_response(
    responses: &mut serde_json::Map<String, Value>,
    auth_misconfigured: &Value,
) {
    let Some(existing) = responses.get_mut("500") else {
        responses.insert("500".to_string(), auth_misconfigured.clone());
        return;
    };

    let Some(existing) = existing.as_object_mut() else {
        return;
    };
    let description = existing
        .entry("description")
        .or_insert_with(|| json!("Internal server error"));
    if let Some(description_text) = description.as_str() {
        if !description_text.contains("auth enabled but no tokens configured") {
            *description = json!(format!(
                "{description_text}; gateway auth enabled but no tokens configured"
            ));
        }
    }

    let content = existing
        .entry("content")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("OpenAPI response content should be an object");
    let json_content = content
        .entry("application/json")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("OpenAPI application/json response should be an object");
    let schema = json_content.entry("schema").or_insert_with(|| json!(null));
    if schema.is_null() {
        *schema = standard_api_error_schema();
    } else if !schema_allows_standard_api_error(schema) {
        let old_schema = schema.clone();
        *schema = json!({
            "oneOf": [
                old_schema,
                standard_api_error_schema()
            ]
        });
    }
}

fn admin_mutation(method: &str, path: &str) -> bool {
    if !matches!(method, "post" | "put" | "delete") {
        return false;
    }
    path.starts_with("/v1/config") || path.starts_with("/v1/admin") || path.starts_with("/v1/pools")
}

fn annotate_slash_bearing_path_parameters(paths: &mut serde_json::Map<String, Value>) {
    for (path, param_name, runtime_path) in [
        ("/v1/models/{model}", "model", "/v1/models/{*model}"),
        ("/v1/encode/{model}", "model", "/v1/encode/{*model}"),
        ("/v1/score/{model}", "model", "/v1/score/{*model}"),
        ("/v1/extract/{model}", "model", "/v1/extract/{*model}"),
        ("/v1/configs/models/{id}", "id", "/v1/configs/models/{*id}"),
        (
            "/v1/configs/models/{id}/status",
            "id",
            "/v1/configs/models/{*id}",
        ),
    ] {
        let Some(path_item) = paths.get_mut(path).and_then(|item| item.as_object_mut()) else {
            continue;
        };
        for operation in path_item.values_mut() {
            let Some(operation) = operation.as_object_mut() else {
                continue;
            };
            operation.insert("x-sie-axum-catch-all".to_string(), json!(runtime_path));
            let Some(parameters) = operation
                .get_mut("parameters")
                .and_then(|parameters| parameters.as_array_mut())
            else {
                continue;
            };
            for parameter in parameters {
                if parameter.get("name").and_then(|name| name.as_str()) != Some(param_name) {
                    continue;
                }
                parameter["description"] = json!(format!(
                    "Model id. Runtime route is Axum catch-all `{runtime_path}`; clients using this OpenAPI path template should percent-encode slashes in model ids, for example `BAAI%2Fbge-m3`."
                ));
            }
        }
    }
}

pub fn write_openapi_json(path: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    match path {
        Some(path) => std::fs::write(path, OPENAPI_JSON.as_bytes())?,
        None => print!("{}", OPENAPI_JSON.as_str()),
    }
    Ok(())
}

#[utoipa::path(
    get,
    path = "/v1/configs/models/{id}/status",
    operation_id = "get_model_config_status",
    tag = "config",
    params(("id" = String, Path, description = "Model id; percent-encode slashes when using OpenAPI-generated clients")),
    responses(
        (status = 200, description = "Per-replica worker acknowledgement status", body = ModelConfigStatusResponse),
        (status = 404, description = "Model not found", body = StandardApiError)
    )
)]
#[allow(dead_code)]
pub fn get_model_config_status_doc() {}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ErrorDetailCore {
    pub code: String,
    pub message: String,
}

/// FastAPI-style error envelope for most gateway JSON errors.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct StandardApiError {
    pub detail: ErrorDetailCore,
}

/// ``detail`` when the caller's ``X-SIE-MACHINE-PROFILE`` GPU is not in the gateway allow-list.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GpuNotConfiguredDetail {
    pub code: String,
    pub message: String,
    pub gpu: String,
    pub configured_gpu_types: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GpuNotConfiguredError {
    pub detail: GpuNotConfiguredDetail,
}

/// SDK-style ``502`` body for ``MODEL_LOAD_FAILED`` (SDK short-circuit).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GatewayModelLoadFailedDetail {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_class: Option<String>,
    pub attempts: i32,
    pub permanent: bool,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GatewayModelLoadFailedResponse {
    pub error: GatewayModelLoadFailedDetail,
}

/// OpenAI-compatible ``POST /v1/embeddings`` request (subset supported on gateway).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAIEmbeddingRequest {
    pub model: String,
    pub input: OpenAIEmbeddingInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<OpenAIEmbeddingEncodingFormat>,
    /// Accepted but ignored; the gateway returns the model's native dimension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Accepted but ignored; kept for OpenAI SDK compatibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum OpenAIEmbeddingInput {
    String(String),
    Strings(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIEmbeddingEncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum OpenAIEmbeddingVector {
    Float(Vec<f64>),
    Base64(String),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAIEmbeddingDataEntry {
    pub object: String,
    pub embedding: OpenAIEmbeddingVector,
    pub index: usize,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAIEmbeddingUsage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAIEmbeddingsListResponse {
    pub object: String,
    pub data: Vec<OpenAIEmbeddingDataEntry>,
    pub model: String,
    pub usage: OpenAIEmbeddingUsage,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GatewayErrorResponse {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub details: Option<Vec<InferenceErrorDetail>>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub gpu: Option<String>,
    #[serde(default)]
    pub configured_gpu_types: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum InferenceInternalServerErrorResponse {
    AllItemsFailed(AllItemsFailedResponse),
    Standard(StandardApiError),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum InferenceServiceUnavailableResponse {
    Gateway(GatewayErrorResponse),
    Standard(StandardApiError),
    GpuNotConfigured(GpuNotConfiguredError),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BundleRoutingConflictDetail {
    pub code: String,
    pub message: String,
    pub compatible_bundles: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BundleConflictResponse {
    pub detail: BundleRoutingConflictDetail,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResolveModelNotFoundDetail {
    pub code: String,
    pub message: String,
    pub model: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResolveModelNotFoundResponse {
    pub detail: ResolveModelNotFoundDetail,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResolveBundleConflictDetail {
    pub code: String,
    pub message: String,
    pub model: String,
    pub bundle: String,
    pub compatible_bundles: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResolveBundleConflictResponse {
    pub detail: ResolveBundleConflictDetail,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AllItemsFailedResponse {
    pub error: String,
    pub details: Vec<InferenceErrorDetail>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MessageResponse {
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    #[serde(rename = "type")]
    #[schema(rename = "type")]
    pub service_type: String,
    pub configured_gpu_types: Vec<String>,
    pub live_gpu_types: Vec<String>,
    pub cluster: ClusterSummary,
    pub workers: Vec<crate::types::worker::WorkerInfo>,
    pub models: Vec<crate::types::worker::ModelInfo>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ClusterSummary {
    pub worker_count: i32,
    pub gpu_count: i32,
    pub models_loaded: i32,
    pub total_qps: f64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProfileInfoWire {
    #[serde(default)]
    pub is_default: bool,
}

/// Wire shape aligned with ``sie_server`` ``ModelInfo`` for ``GET /v1/models``.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelInfoWire {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub dims: std::collections::HashMap<String, i64>,
    pub loaded: bool,
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<serde_json::Value>,
    pub max_sequence_length: Option<u64>,
    pub profiles: std::collections::HashMap<String, ProfileInfoWire>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelsResponse {
    pub models: Vec<ModelInfoWire>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelNotFoundResponse {
    pub detail: ModelNotFoundDetail,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelNotFoundDetail {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct PoolListResponse {
    pub pools: Vec<crate::types::pool::Pool>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ConfigModelsResponse {
    pub models: Vec<ConfigModelSummary>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ConfigModelSummary {
    pub model_id: String,
    pub profiles: Vec<String>,
    pub source: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ConfigModelDocument {
    pub sie_id: String,
    pub source: String,
    pub bundles: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelConfigStatusResponse {
    pub model_id: String,
    pub config_epoch: u64,
    pub all_bundles_acked: bool,
    pub no_bundles: bool,
    pub bundles: Vec<ModelAckBundleStatus>,
    pub source: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelAckBundleStatus {
    pub bundle_id: String,
    pub expected_bundle_config_hash: String,
    pub total_eligible_workers: usize,
    pub acked_workers: Vec<String>,
    pub pending_workers: Vec<String>,
    pub acked: bool,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BundleConfigsResponse {
    pub bundles: Vec<BundleConfigSummary>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BundleConfigSummary {
    pub bundle_id: String,
    pub priority: i32,
    pub adapter_count: usize,
    pub source: String,
    pub connected_workers: usize,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BundleConfigDocument {
    pub name: String,
    pub priority: i32,
    pub source: String,
    pub adapters: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResolveConfigResponse {
    pub model: String,
    pub resolved_bundle: String,
    pub compatible_bundles: Vec<String>,
    pub profiles: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ImageInput {
    pub data: Vec<u8>,
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DocumentInput {
    pub data: Vec<u8>,
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ItemInput {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub images: Option<Vec<ImageInput>>,
    #[serde(default)]
    pub document: Option<DocumentInput>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct EncodeParams {
    #[serde(default)]
    pub output_types: Option<Vec<OutputType>>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub output_dtype: Option<OutputDtype>,
    #[serde(default)]
    pub options: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum OutputType {
    Dense,
    Sparse,
    Multivector,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum OutputDtype {
    Float32,
    Float16,
    Int8,
    Uint8,
    Binary,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct EncodeRequest {
    #[schema(min_items = 1)]
    pub items: Vec<ItemInput>,
    #[serde(default)]
    pub params: Option<EncodeParams>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExtractParams {
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    #[serde(default)]
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub options: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExtractRequest {
    #[schema(min_items = 1)]
    pub items: Vec<ItemInput>,
    #[serde(default)]
    pub params: Option<ExtractParams>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ScoreRequest {
    pub query: ItemInput,
    #[schema(min_items = 1)]
    pub items: Vec<ItemInput>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub options: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DenseVector {
    pub dims: usize,
    pub dtype: OutputDtype,
    pub values: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SparseVector {
    #[serde(default)]
    pub dims: Option<usize>,
    pub dtype: OutputDtype,
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MultiVector {
    pub token_dims: usize,
    pub num_tokens: usize,
    pub dtype: OutputDtype,
    pub values: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct EncodeResult {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub dense: Option<DenseVector>,
    #[serde(default)]
    pub sparse: Option<SparseVector>,
    #[serde(default)]
    pub multivector: Option<MultiVector>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct TimingInfo {
    pub total_ms: f64,
    pub queue_ms: f64,
    pub tokenization_ms: f64,
    pub inference_ms: f64,
    #[serde(default)]
    pub postprocessing_ms: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct EncodeResponse {
    pub model: String,
    pub items: Vec<EncodeResult>,
    #[serde(default)]
    pub timing: Option<TimingInfo>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct Entity {
    pub text: String,
    pub label: String,
    pub score: f64,
    #[serde(default)]
    pub start: Option<usize>,
    #[serde(default)]
    pub end: Option<usize>,
    #[serde(default)]
    pub bbox: Option<Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct Relation {
    pub head: String,
    pub tail: String,
    pub relation: String,
    pub score: f64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExtractResult {
    pub id: String,
    #[serde(default)]
    pub entities: Vec<Entity>,
    #[serde(default)]
    pub relations: Vec<Relation>,
    #[serde(default)]
    pub classifications: Vec<Value>,
    #[serde(default)]
    pub data: Value,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExtractResponse {
    pub model: String,
    pub items: Vec<ExtractResult>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ScoreEntry {
    pub item_id: String,
    pub score: f64,
    pub rank: usize,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ScoreResponse {
    pub model: String,
    #[serde(default)]
    pub query_id: Option<String>,
    pub scores: Vec<ScoreEntry>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct InferenceErrorDetail {
    pub item_index: u32,
    pub error: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProvisioningResponse {
    pub status: String,
    #[serde(default)]
    pub gpu: String,
    #[serde(default)]
    pub bundle: String,
    pub estimated_wait_s: u64,
    #[serde(default)]
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openapi_document_has_gateway_metadata() {
        let doc = ApiDoc::openapi();
        assert_eq!(doc.info.title, "SIE Gateway");
    }

    #[tokio::test]
    async fn openapi_handler_sets_gateway_version() {
        let doc: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        assert_eq!(doc["info"]["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(doc["info"]["license"]["name"], "Apache-2.0");
    }

    #[test]
    fn openapi_json_documents_auth_and_exempt_routes() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        assert_eq!(spec["security"], json!([{ "bearerAuth": [] }]));
        assert_eq!(spec["paths"]["/healthz"]["get"]["security"], json!([]));
        assert_eq!(spec["paths"]["/readyz"]["get"]["security"], json!([]));
        assert!(spec["paths"]["/readyz"]["get"]["description"]
            .as_str()
            .unwrap()
            .contains("never returns 503"));
        assert_eq!(spec["paths"]["/openapi.json"]["get"]["security"], json!([]));
        assert_eq!(
            spec["components"]["securitySchemes"]["bearerAuth"]["scheme"],
            "bearer"
        );
    }

    #[test]
    fn openapi_json_documents_its_response_content() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        assert_eq!(
            spec["paths"]["/openapi.json"]["get"]["responses"]["200"]["content"]
                ["application/json"]["schema"]["type"],
            "object"
        );
    }

    #[test]
    fn openapi_json_documents_create_pool_runtime_validation() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        let create_pool = &spec["components"]["schemas"]["CreatePoolRequest"];
        assert_eq!(create_pool["required"], json!(["name", "gpus"]));
        assert_eq!(create_pool["properties"]["gpus"]["minProperties"], 1);
    }

    #[test]
    fn openapi_document_includes_gateway_paths() {
        let doc = ApiDoc::openapi();
        let paths = doc.paths.paths;
        for path in [
            "/",
            "/openapi.json",
            "/healthz",
            "/v1/models",
            "/v1/pools",
            "/v1/configs/models",
            "/v1/configs/models/{id}/status",
            "/v1/embeddings",
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
        ] {
            assert!(paths.contains_key(path), "{path} missing from OpenAPI doc");
        }
    }

    #[test]
    fn openapi_document_covers_all_gateway_route_methods() {
        let spec = serde_json::to_value(openapi_document()).unwrap();
        let paths = spec["paths"].as_object().unwrap();
        let expected = [
            ("/", &["get"][..]),
            ("/healthz", &["get"][..]),
            ("/readyz", &["get"][..]),
            ("/health", &["get"][..]),
            ("/metrics", &["get"][..]),
            ("/openapi.json", &["get"][..]),
            ("/v1/models", &["get"][..]),
            ("/v1/models/{model}", &["get"][..]),
            ("/v1/pools", &["get", "post"][..]),
            ("/v1/pools/{name}", &["delete", "get"][..]),
            ("/v1/pools/{name}/renew", &["post"][..]),
            ("/v1/configs/models", &["get"][..]),
            ("/v1/configs/models/{id}", &["get"][..]),
            ("/v1/configs/models/{id}/status", &["get"][..]),
            ("/v1/configs/bundles", &["get"][..]),
            ("/v1/configs/bundles/{id}", &["get"][..]),
            ("/v1/configs/resolve", &["post"][..]),
            ("/v1/embeddings", &["post"][..]),
            ("/ws/cluster-status", &["get"][..]),
            ("/v1/encode/{model}", &["post"][..]),
            ("/v1/score/{model}", &["post"][..]),
            ("/v1/extract/{model}", &["post"][..]),
        ];

        let actual_paths: std::collections::BTreeSet<_> =
            paths.keys().map(String::as_str).collect();
        let expected_paths: std::collections::BTreeSet<_> =
            expected.iter().map(|(path, _)| *path).collect();
        assert_eq!(actual_paths, expected_paths);

        for (path, methods) in expected {
            let path_item = paths[path].as_object().unwrap();
            let actual_methods: std::collections::BTreeSet<_> =
                path_item.keys().map(String::as_str).collect();
            let expected_methods: std::collections::BTreeSet<_> = methods.iter().copied().collect();
            assert_eq!(actual_methods, expected_methods, "{path} methods differ");
        }
    }

    #[test]
    fn openapi_document_does_not_advertise_internal_engine_header() {
        let spec = serde_json::to_value(openapi_document()).unwrap();
        for path_item in spec["paths"].as_object().unwrap().values() {
            for operation in path_item.as_object().unwrap().values() {
                let Some(parameters) = operation.get("parameters").and_then(|p| p.as_array())
                else {
                    continue;
                };
                for parameter in parameters {
                    let name = parameter["name"].as_str().unwrap_or_default();
                    assert_ne!(name.to_ascii_lowercase(), "x-sie-engine");
                }
            }
        }
    }

    #[test]
    fn openapi_document_covers_inference_error_statuses() {
        let spec = serde_json::to_value(openapi_document()).unwrap();
        for path in [
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
        ] {
            let responses = spec["paths"][path]["post"]["responses"]
                .as_object()
                .unwrap();
            for status in [
                "200", "202", "400", "404", "409", "413", "500", "502", "503", "504",
            ] {
                assert!(responses.contains_key(status), "{path} missing {status}");
            }
        }

        let emb = &spec["paths"]["/v1/embeddings"]["post"]["responses"]
            .as_object()
            .unwrap();
        for status in [
            "200", "202", "400", "401", "404", "409", "413", "500", "502", "503", "504",
        ] {
            assert!(emb.contains_key(status), "/v1/embeddings missing {status}");
        }
    }

    #[test]
    fn openapi_document_documents_inference_error_unions() {
        let spec = serde_json::to_value(openapi_document()).unwrap();
        let internal_error =
            &spec["components"]["schemas"]["InferenceInternalServerErrorResponse"]["oneOf"];
        assert_eq!(
            internal_error,
            &json!([
                {"$ref": "#/components/schemas/AllItemsFailedResponse"},
                {"$ref": "#/components/schemas/StandardApiError"},
            ])
        );

        let service_unavailable =
            &spec["components"]["schemas"]["InferenceServiceUnavailableResponse"]["oneOf"];
        assert_eq!(
            service_unavailable,
            &json!([
                {"$ref": "#/components/schemas/GatewayErrorResponse"},
                {"$ref": "#/components/schemas/StandardApiError"},
                {"$ref": "#/components/schemas/GpuNotConfiguredError"},
            ])
        );

        for path in [
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
            "/v1/embeddings",
        ] {
            assert_eq!(
                spec["paths"][path]["post"]["responses"]["500"]["content"]["application/json"]
                    ["schema"]["$ref"],
                "#/components/schemas/InferenceInternalServerErrorResponse"
            );
            assert_eq!(
                spec["paths"][path]["post"]["responses"]["503"]["content"]["application/json"]
                    ["schema"]["$ref"],
                "#/components/schemas/InferenceServiceUnavailableResponse"
            );
        }
    }

    #[test]
    fn openapi_export_adds_msgpack_content_and_unauthorized_responses() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        for path in [
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
        ] {
            let post = &spec["paths"][path]["post"];
            let content = &post["responses"]["200"]["content"];
            assert!(
                content.get("application/json").is_some(),
                "{path} missing application/json 200"
            );
            assert!(
                content.get("application/x-msgpack").is_some(),
                "{path} missing application/x-msgpack 200"
            );
            assert!(
                content.get("application/msgpack").is_some(),
                "{path} missing application/msgpack 200"
            );
            assert!(
                content.get("application/vnd.msgpack").is_some(),
                "{path} missing application/vnd.msgpack 200"
            );
            let req_content = &post["requestBody"]["content"];
            assert!(req_content.get("application/x-msgpack").is_some());
            assert!(req_content.get("application/msgpack").is_some());
            assert!(req_content.get("application/vnd.msgpack").is_some());

            let responses = post["responses"].as_object().unwrap();
            assert!(responses.contains_key("401"), "{path} missing injected 401");
            assert!(
                responses.contains_key("500"),
                "{path} missing auth-misconfigured 500"
            );
            assert!(
                responses["200"]["headers"]
                    .get("X-SIE-Request-Id")
                    .is_some(),
                "{path} missing documented request id header"
            );
            assert!(
                responses["200"]["headers"].get("X-Queue-Time").is_some(),
                "{path} missing documented queue timing header"
            );
        }

        let pool_post = &spec["paths"]["/v1/pools"]["post"]["responses"];
        assert!(pool_post.as_object().unwrap().contains_key("401"));
        assert!(pool_post.as_object().unwrap().contains_key("403"));
        assert!(pool_post.as_object().unwrap().contains_key("500"));

        let config_resolve = &spec["paths"]["/v1/configs/resolve"]["post"]["responses"];
        assert!(config_resolve.as_object().unwrap().contains_key("401"));
        assert!(config_resolve.as_object().unwrap().contains_key("403"));
        assert!(config_resolve.as_object().unwrap().contains_key("500"));

        let embeddings_202_headers =
            &spec["paths"]["/v1/embeddings"]["post"]["responses"]["202"]["headers"];
        assert!(embeddings_202_headers.get("Retry-After").is_some());
        assert!(embeddings_202_headers.get("X-SIE-Version").is_some());
        assert!(embeddings_202_headers.get("X-SIE-Server-Version").is_some());

        let embeddings_200_headers =
            &spec["paths"]["/v1/embeddings"]["post"]["responses"]["200"]["headers"];
        for name in [
            "X-SIE-Request-Id",
            "X-SIE-Worker",
            "X-Queue-Publish-Time",
            "X-Queue-Wait-Time",
            "X-Queue-Time",
            "X-Inference-Time",
            "X-Tokenization-Time",
            "X-Postprocessing-Time",
            "X-Payload-Fetch-Time",
        ] {
            assert!(
                embeddings_200_headers.get(name).is_some(),
                "/v1/embeddings 200 missing documented {name} header"
            );
        }
    }

    #[test]
    fn openapi_export_documents_server_parity_cleanup() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();

        for schema in ["EncodeResponse", "ExtractResponse", "ScoreResponse"] {
            assert!(
                spec["components"]["schemas"][schema]["properties"]
                    .get("errors")
                    .is_none(),
                "{schema} must not document gateway-only partial error envelope"
            );
        }

        let model_param = &spec["paths"]["/v1/encode/{model}"]["post"]["parameters"][0];
        assert!(model_param.get("allowReserved").is_none());
        assert!(model_param["description"]
            .as_str()
            .unwrap()
            .contains("percent-encode slashes"));
        assert_eq!(
            spec["paths"]["/v1/encode/{model}"]["post"]["x-sie-axum-catch-all"],
            "/v1/encode/{*model}"
        );

        let model_load_failed = &spec["components"]["schemas"]["GatewayModelLoadFailedResponse"];
        assert!(model_load_failed["properties"].get("error").is_some());
        assert!(model_load_failed["properties"].get("detail").is_none());

        let embeddings_request = &spec["components"]["schemas"]["OpenAIEmbeddingRequest"];
        assert_eq!(
            embeddings_request["properties"]["input"]["$ref"],
            "#/components/schemas/OpenAIEmbeddingInput"
        );
        assert!(embeddings_request["properties"]["encoding_format"]["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .any(|schema| schema["$ref"] == "#/components/schemas/OpenAIEmbeddingEncodingFormat"));
        assert_eq!(
            embeddings_request["properties"]["dimensions"]["type"],
            json!(["integer", "null"])
        );
        assert_eq!(
            embeddings_request["properties"]["user"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(
            spec["components"]["schemas"]["OpenAIEmbeddingEncodingFormat"]["enum"],
            json!(["float", "base64"])
        );

        for path in [
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
        ] {
            assert!(spec["paths"][path]["post"]["description"]
                .as_str()
                .unwrap()
                .contains("Mixed-success batches return 200"));
        }
        assert!(spec["paths"]["/v1/embeddings"]["post"]["description"]
            .as_str()
            .unwrap()
            .contains("partial or truncated internal encode success"));

        let headers = &spec["paths"]["/v1/embeddings"]["post"]["responses"]["502"]["headers"];
        assert!(headers.get("X-SIE-Error-Code").is_some());
        assert!(headers.get("X-SIE-Error-Version").is_some());
    }
}
