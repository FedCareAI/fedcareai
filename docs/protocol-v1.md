# FedCareAI interchange protocol v1

This specification documents the existing `/api/v1` protocol and local numeric
update format used by library release 0.1.0. It does not add a new wire field.
Incompatible wire changes require a new API major version; compatible optional
fields may be added within v1. Model authors and participants must use compatible
Python/framework versions and identical preprocessing.

## Local inputs and numeric updates

Data contracts are UTF-8 JSON matching `fedcareai.schemas.data_contract.DataContract`.
`columns` defines ordered model inputs excluding `target_column`. Tabular data
is CSV. `prediction_type` is `classification` or `regression`; the requested
metric must match that type. Regression requires a numeric target declaration.
Image data is not supported by the runners.

Model artifacts are sklearn `.joblib`/`.pkl` or Keras `.keras`/`.h5` files. Only
trusted sources may supply model objects; the server stores them without loading
them. Participants load them locally.

Training updates are `.npz` archives loaded with `allow_pickle=False`. Each tensor
must be a nonempty finite integer or floating-point array. Sklearn incremental
linear models use `coef` and `intercept`; Keras uses `weight_0`, `weight_1`, etc.
Names and shapes must agree across sites and with the base model on export.
Updates contain complete locally trained weights, not weight deltas.

For visible positive counts, tensor averaging is
`sum(site_tensor * site_count / sum(counts))`. Hidden or mixed hidden/visible
counts use equal site weighting. Offline callers select equal weighting by
omitting counts. CSVs and patient-level predictions are not part of this format.

## Optional platform adapter

The configured base URL includes `/api/v1`. Password login uses
`POST /login/access-token`; subsequent requests use a bearer token.

| Operation | Endpoint | Payload |
| --- | --- | --- |
| List tasks | `GET /tasks` | Available task metadata |
| Fetch task | `GET /tasks/{id}` | `TaskMetadata` JSON |
| Fetch base model | `GET /tasks/{id}/model` | Model artifact bytes |
| Fetch latest aggregate | `GET /tasks/{id}/weights` | Numeric `.npz` bytes |
| Submit validation | `POST /tasks/{id}/results` | JSON result |
| Submit training | `POST /tasks/{id}/results` | Multipart `result` JSON and `artifact` file |

Task metadata includes `id`, `project_id`, `name`, `task_type`, `status`, `model`,
`model_filename`, `data_contract`, `current_round`, and `rounds`. Model metadata
includes its SHA-256, byte size, framework, Python version, and filename. The
`weights_available` and `weights_sha256` fields identify the latest aggregate.
The CLI verifies downloaded base and aggregate checksums before use.

Validation results contain `task_id`, `round`, `result_type=external_validation`,
`metrics`, `local_sample_count`, and `local_sample_count_visibility`. Training
results use `result_type=federated_aggregation_update`, `artifact_sha256`, and
metadata including framework, epochs, `raw_data_uploaded=false`, and
`weight_format=numeric_npz`. `artifact_path` is a local-only file reference;
the adapter removes it before sending JSON and uploads the file separately.

Visibility is `exact`, `rounded` (nearest ten), or `hidden` (null count). Visible
counts must remain positive. Pull each new round before training: the platform
adapter applies neighboring downloaded `weights.npz` files and clears stale
result payloads on pull. Standalone workflows never apply neighboring weights
implicitly; export the aggregate to a complete model before the next round.
