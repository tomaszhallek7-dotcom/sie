use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};

/// Fields mirrored from model YAML for ``/v1/models`` wire parity with
/// ``sie_server`` ``ModelInfo`` (inputs, outputs, dims, profiles, …).
#[derive(Debug, Clone, Default)]
pub struct ModelInfoExtras {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub dims: HashMap<String, i64>,
    pub max_sequence_length: Option<u64>,
}

impl ModelInfoExtras {
    /// Best-effort extraction from a raw model YAML document (same files as
    /// ``sie_server`` / ``sie-config``). Missing sections fall back to
    /// conservative defaults so the JSON shape stays valid.
    pub fn from_yaml_raw(raw: &serde_yaml::Value) -> Self {
        let mut extras = Self::default();

        if let serde_yaml::Value::Mapping(m) = raw.get("inputs").unwrap_or(&serde_yaml::Value::Null)
        {
            for (k, v) in m {
                let Some(ks) = k.as_str() else {
                    continue;
                };
                if matches!(v, serde_yaml::Value::Bool(true)) {
                    extras.inputs.push(ks.to_string());
                }
            }
        }
        if extras.inputs.is_empty() {
            extras.inputs.push("text".to_string());
        }

        extras.max_sequence_length = raw.get("max_sequence_length").and_then(|v| v.as_u64());

        let tasks = raw.get("tasks");
        if let Some(enc) = tasks.and_then(|t| match t.get("encode")? {
            serde_yaml::Value::Mapping(m) => Some(m),
            _ => None,
        }) {
            for (k, v) in enc {
                let Some(key) = k.as_str() else {
                    continue;
                };
                if key.is_empty() {
                    continue;
                }
                match v {
                    serde_yaml::Value::Mapping(vm) => {
                        if let Some(dim) = vm.get("dim").and_then(|d| {
                            d.as_u64().or_else(|| d.as_i64().map(|i| i.max(0) as u64))
                        }) {
                            extras.dims.insert(key.to_string(), dim as i64);
                            extras.outputs.push(key.to_string());
                        } else if !vm.is_empty() {
                            extras.outputs.push(key.to_string());
                        }
                    }
                    serde_yaml::Value::Null => {}
                    _ => {
                        extras.outputs.push(key.to_string());
                    }
                }
            }
        }

        let tasks_absent_or_empty = match tasks {
            None | Some(serde_yaml::Value::Null) => true,
            Some(serde_yaml::Value::Mapping(m)) => m.is_empty(),
            _ => false,
        };
        if extras.outputs.is_empty() && tasks_absent_or_empty {
            extras.outputs.push("dense".to_string());
        }

        extras
    }

    pub fn from_model_config(config: &ModelConfig) -> Self {
        match serde_yaml::to_value(config) {
            Ok(v) => Self::from_yaml_raw(&v),
            Err(_) => Self {
                inputs: vec!["text".to_string()],
                outputs: vec!["dense".to_string()],
                dims: HashMap::new(),
                max_sequence_length: config.max_sequence_length,
            },
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    #[serde(alias = "sie_id")]
    pub name: String,
    #[serde(default)]
    pub adapter_module: Option<String>,
    #[serde(default)]
    pub default_bundle: Option<String>,
    #[serde(default)]
    pub profiles: HashMap<String, ProfileConfig>,
    /// Model YAML ``inputs:`` map (e.g. ``text: true``).
    #[serde(default)]
    pub inputs: Option<HashMap<String, bool>>,
    #[serde(default)]
    pub max_sequence_length: Option<u64>,
    /// Full ``tasks:`` tree from YAML (encode outputs / dims).
    #[serde(default)]
    pub tasks: Option<serde_yaml::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProfileConfig {
    #[serde(default)]
    pub adapter_path: Option<String>,
    #[serde(default)]
    pub max_batch_tokens: Option<u32>,
    #[serde(default)]
    pub compute_precision: Option<String>,
    #[serde(default)]
    pub adapter_options: Option<serde_json::Value>,
    #[serde(default)]
    pub extends: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub bundles: Vec<String>,
    pub adapter_modules: HashSet<String>,
    pub profile_names: HashSet<String>,
    pub profile_configs: HashMap<String, CanonicalProfile>,
    pub info_extras: ModelInfoExtras,
}

impl ModelEntry {
    /// JSON shaped like ``sie_server.api.models.ModelInfo`` for HTTP clients.
    pub fn to_model_info_value(&self, loaded: bool) -> Value {
        let state = if loaded { "loaded" } else { "available" };
        let mut profiles = Map::new();
        for pname in &self.profile_names {
            profiles.insert(pname.clone(), json!({ "is_default": pname == "default" }));
        }
        json!({
            "name": self.name,
            "inputs": self.info_extras.inputs,
            "outputs": self.info_extras.outputs,
            "dims": self.info_extras.dims,
            "loaded": loaded,
            "state": state,
            "last_error": Value::Null,
            "max_sequence_length": self.info_extras.max_sequence_length,
            "profiles": Value::Object(profiles),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalProfile {
    pub adapter_path: Option<String>,
    pub max_batch_tokens: Option<u32>,
    pub compute_precision: Option<String>,
    pub adapter_options: Option<serde_json::Value>,
}

impl CanonicalProfile {
    pub fn from_profile(profile: &ProfileConfig) -> Self {
        // Python-compat normalization. `sie_config.model_registry`
        // canonicalizes `adapter_options` via `not any(values)`, which in
        // Python treats every falsy scalar (None, 0, 0.0, False, "", [],
        // {}) as "empty". If we don't mirror that here, the gateway's
        // `compute_bundle_config_hash` can diverge from the config
        // service's hash for data like `{"flag": 0}` — the config
        // service would strip the field and hash `{}`, the gateway would
        // keep it and hash `{"flag": 0}`, and every worker in that
        // bundle would sit in `pending_workers` forever because its
        // advertised `bundle_config_hash` never matches the gateway's
        // expected hash. See `canonicalize_adapter_options` for the
        // exact predicate.
        let adapter_options = profile
            .adapter_options
            .clone()
            .and_then(canonicalize_adapter_options);

        Self {
            adapter_path: profile.adapter_path.clone(),
            max_batch_tokens: profile.max_batch_tokens,
            compute_precision: profile.compute_precision.clone(),
            adapter_options,
        }
    }
}

/// Mirror of Python's `not any(adapter_opts.values())` falsy check.
///
/// Returns `None` if `opts` is an object whose values are ALL Python-falsy
/// (null, `false`, `0`, `0.0`, `""`, empty array, empty object). Otherwise
/// returns `Some(opts)` unchanged.
fn canonicalize_adapter_options(opts: serde_json::Value) -> Option<serde_json::Value> {
    if let serde_json::Value::Object(ref map) = opts {
        let all_falsy = map.values().all(|v| match v {
            serde_json::Value::Null => true,
            serde_json::Value::Bool(b) => !*b,
            serde_json::Value::Number(n) => {
                n.as_f64().map(|f| f == 0.0).unwrap_or(false)
                    || n.as_i64().map(|i| i == 0).unwrap_or(false)
                    || n.as_u64().map(|u| u == 0).unwrap_or(false)
            }
            serde_json::Value::String(s) => s.is_empty(),
            serde_json::Value::Array(a) => a.is_empty(),
            serde_json::Value::Object(o) => o.is_empty(),
        });
        if all_falsy {
            return None;
        }
    }
    Some(opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_profile_basic() {
        let profile = ProfileConfig {
            adapter_path: Some("module:Adapter".into()),
            max_batch_tokens: Some(4096),
            compute_precision: Some("float16".into()),
            adapter_options: None,
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert_eq!(canonical.adapter_path, Some("module:Adapter".into()));
        assert_eq!(canonical.max_batch_tokens, Some(4096));
        assert_eq!(canonical.compute_precision, Some("float16".into()));
        assert!(canonical.adapter_options.is_none());
    }

    #[test]
    fn test_canonical_profile_strips_null_only_options() {
        let profile = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: None,
            compute_precision: None,
            adapter_options: Some(serde_json::json!({"key": null})),
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert!(canonical.adapter_options.is_none());
    }

    #[test]
    fn test_canonical_profile_strips_false_only_options() {
        let profile = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: None,
            compute_precision: None,
            adapter_options: Some(serde_json::json!({"enabled": false})),
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert!(canonical.adapter_options.is_none());
    }

    #[test]
    fn test_canonical_profile_keeps_meaningful_options() {
        let opts = serde_json::json!({"batch_size": 32, "key": null});
        let profile = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: None,
            compute_precision: None,
            adapter_options: Some(opts.clone()),
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert_eq!(canonical.adapter_options, Some(opts));
    }

    #[test]
    fn test_canonical_profile_python_falsy_parity() {
        // Config service uses Python `not any(values)` which treats 0,
        // 0.0, "", [], {}, None, False as empty. Gateway must match or
        // `compute_bundle_config_hash` diverges and every worker in the
        // affected bundle sits in `pending_workers` forever. Regression
        // for the bug where gateway-side canonicalization only stripped
        // null/false.
        let cases = vec![
            serde_json::json!({"x": 0}),
            serde_json::json!({"x": 0.0}),
            serde_json::json!({"x": ""}),
            serde_json::json!({"x": []}),
            serde_json::json!({"x": {}}),
            serde_json::json!({"x": null, "y": false, "z": 0}),
        ];
        for opts in cases {
            let profile = ProfileConfig {
                adapter_path: Some("mod:A".into()),
                max_batch_tokens: None,
                compute_precision: None,
                adapter_options: Some(opts.clone()),
                extends: None,
            };
            let canonical = CanonicalProfile::from_profile(&profile);
            assert!(
                canonical.adapter_options.is_none(),
                "Expected falsy-only options {opts:?} to be stripped to None for Python parity"
            );
        }
    }

    #[test]
    fn test_canonical_profile_keeps_nonzero_numbers() {
        let profile = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: None,
            compute_precision: None,
            adapter_options: Some(serde_json::json!({"x": 1})),
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert!(canonical.adapter_options.is_some());
    }

    #[test]
    fn test_canonical_profile_equality() {
        let p1 = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: Some(4096),
            compute_precision: None,
            adapter_options: None,
            extends: None,
        };
        let p2 = p1.clone();
        assert_eq!(
            CanonicalProfile::from_profile(&p1),
            CanonicalProfile::from_profile(&p2)
        );
    }

    #[test]
    fn test_model_config_yaml_deserialization() {
        let yaml = r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "module:Adapter"
    max_batch_tokens: 4096
"#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "BAAI/bge-m3");
        assert_eq!(config.profiles.len(), 1);
        assert_eq!(
            config.profiles["default"].adapter_path,
            Some("module:Adapter".into())
        );
    }

    #[test]
    fn test_model_config_sie_id_alias() {
        let yaml = r#"
sie_id: my/model
profiles: {}
"#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "my/model");
    }

    #[test]
    fn test_model_config_json_roundtrip() {
        let config = ModelConfig {
            name: "test/model".into(),
            adapter_module: Some("mod".into()),
            default_bundle: None,
            profiles: HashMap::new(),
            inputs: None,
            max_sequence_length: None,
            tasks: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "test/model");
        assert_eq!(back.adapter_module, Some("mod".into()));
    }

    #[test]
    fn test_model_info_extras_defaults_dense_when_tasks_absent() {
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: example/model
inputs:
  text: true
"#,
        )
        .unwrap();

        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert_eq!(extras.outputs, vec!["dense"]);
    }

    #[test]
    fn test_model_info_extras_does_not_invent_dense_for_non_encode_tasks() {
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: example/reranker
inputs:
  text: true
tasks:
  score:
    relevance:
      dim: 1
"#,
        )
        .unwrap();

        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert!(extras.outputs.is_empty());
        assert!(extras.dims.is_empty());
    }
}
