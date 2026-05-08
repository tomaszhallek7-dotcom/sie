//! Canonical JSON error bodies (FastAPI-style ``{"detail": {"code", "message", ...}}``).
//! SDK-stable responses (``502 MODEL_LOAD_FAILED``, ``503`` retryable ``{"error":{...}}``)
//! intentionally stay on their existing shapes.

use serde_json::{json, Map, Value};

/// Stable ``detail.code`` values for gateway-generated errors.
pub mod code {
    pub const MODEL_NOT_FOUND: &str = "MODEL_NOT_FOUND";
    pub const BUNDLE_NOT_FOUND: &str = "BUNDLE_NOT_FOUND";
    pub const POOL_NOT_FOUND: &str = "POOL_NOT_FOUND";
    pub const INVALID_REQUEST: &str = "INVALID_REQUEST";
    pub const PAYLOAD_TOO_LARGE: &str = "PAYLOAD_TOO_LARGE";
    pub const QUEUE_UNAVAILABLE: &str = "QUEUE_UNAVAILABLE";
    pub const GPU_NOT_CONFIGURED: &str = "GPU_NOT_CONFIGURED";
    pub const BUNDLE_ROUTING_CONFLICT: &str = "BUNDLE_ROUTING_CONFLICT";
    pub const CONFIG_RESOLVE_BUNDLE_CONFLICT: &str = "BUNDLE_ROUTING_CONFLICT";
    pub const INTERNAL_ERROR: &str = "INTERNAL_SERVER_ERROR";
    pub const GATEWAY_TIMEOUT: &str = "GATEWAY_TIMEOUT";
    pub const SERIALIZATION_ERROR: &str = "SERIALIZATION_ERROR";
    pub const UNAUTHORIZED: &str = "UNAUTHORIZED";
    pub const FORBIDDEN: &str = "FORBIDDEN";
    pub const GATEWAY_AUTH_MISCONFIGURED: &str = "GATEWAY_AUTH_MISCONFIGURED";
    pub const DEFAULT_POOL_DELETE_FORBIDDEN: &str = "DEFAULT_POOL_DELETE_FORBIDDEN";
    pub const POOL_OPERATION_FORBIDDEN: &str = "POOL_OPERATION_FORBIDDEN";
    pub const PROVISIONING: &str = "PROVISIONING";
}

/// ``{"detail": {"code", "message"}}``
pub fn json_detail(code: &str, message: impl Into<String>) -> Value {
    json!({
        "detail": {
            "code": code,
            "message": message.into(),
        }
    })
}

/// ``{"detail": { "code", "message", ...extra }}`` — merge extra keys into ``detail``.
pub fn json_detail_merge(
    code: &str,
    message: impl Into<String>,
    extra: Map<String, Value>,
) -> Value {
    let mut d = Map::new();
    d.insert("code".to_string(), json!(code));
    d.insert("message".to_string(), json!(message.into()));
    d.extend(extra);
    json!({ "detail": Value::Object(d) })
}
