# Optional FedCareAI platform integration

This guide is for participants using a FedCareAI platform instance to coordinate tasks across sites. The platform is a separate application, not a requirement for the Python library's standalone workflows.

## Install the library

Install from the public GitHub repository using the [README instructions](../README.md#installation). For Keras tasks, install the `keras` extra. You do not need to download the library through the platform. Use Python and framework versions compatible with the model author.

## Connect to your platform

Register an account on your platform instance and ask the project owner to add your registered email to the project. Use the API URL provided by your administrator, including `/api/v1`:

```powershell
fedcare auth login --api-url https://your-platform.example/api/v1 --email you@example.org
```

For a locally running platform, use `http://127.0.0.1:8000/api/v1` instead. Login prompts for your password. An existing access token can be supplied with `--token`. Settings and the access token are saved under `~/.fedcareai`; downloaded tasks live under `~/.fedcareai/tasks`.

## External validation

```powershell
fedcare task list
fedcare task pull TASK_ID
fedcare task schema-check TASK_ID --data ./local_data.csv
fedcare task validate TASK_ID --data ./local_data.csv
fedcare task preview-submit TASK_ID
fedcare task submit TASK_ID
```

Follow the task's data contract and preprocessing instructions. Feature order follows the contract and extra CSV columns are excluded. Validation uses the task's prediction type and metric, including continuous regression metrics when applicable. Only load models from authors you trust. Downloaded base models and aggregate weights are checked against their SHA-256 checksums.

## Federated training

For a federated training task, use `task aggregate` to train locally and export a numeric update:

```powershell
fedcare task pull TASK_ID
fedcare task aggregate TASK_ID --data ./local_data.csv --epochs 1 --sample-count-visibility exact
fedcare task preview-submit TASK_ID
fedcare task submit TASK_ID --result-type federated_aggregation_update
```

The project owner averages contributions on the platform. Pull again before each new round: the library applies the latest aggregate to the original model and clears stale result payloads. Raw CSV data stays local; numeric weights are visible to the platform. This protocol does not provide cryptographic secure aggregation or differential privacy.

Sample counts can be `exact`, `rounded` (the default, nearest ten), or `hidden`. Visible positive counts determine site weighting. If any count is hidden, all sites have equal weight. For fewer than six samples, use exact or hidden counts to avoid a zero rounded count.

After the owner completes the final round, pull and export the complete model:

```powershell
fedcare task pull TASK_ID
fedcare task export-model TASK_ID --output ./aggregated_model.joblib
```

Use `.keras` or `.h5` for Keras. See [supported models](../README.md#supported-models) for the estimators that can train through this protocol.

## Synthetic platform example

From the cloned library repository:

```powershell
python examples/create_demo.py --output ./demo
```

Upload `demo/model.joblib` to a project and create a task with target `outcome` and ordered numeric features `age, bmi`. Separate participant accounts can use `demo/site_a.csv` and `demo/site_b.csv`. These files contain synthetic data only.

The [v1 interchange specification](protocol-v1.md) documents the API and numeric update format.
