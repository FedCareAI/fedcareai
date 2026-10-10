# FedCareAI Python library

FedCareAI is a Python library for federated machine learning and external model validation, designed for healthcare research and applicable to other domains. It supports local training and validation of tabular models, aggregation of model updates across participating sites, and export of complete models while keeping raw datasets local.

Use it directly from Python or through the `fedcare local` CLI without an account or server. The optional `fedcare task` workflow connects to the separately maintained FedCareAI platform for coordinated collaboration across sites.

## Standalone installation

From this repository, using Python 3.10 or newer:

```powershell
python -m pip install .
```

The default installation supports sklearn. For Keras/TensorFlow, use a TensorFlow-supported Python version and install the extra:

```powershell
python -m pip install ".[keras]"
```

## Offline workflow

Run these commands from the library repository. The example uses synthetic data and an already fitted incremental classifier. Each output path must be new.

```powershell
python examples/create_demo.py --output ./demo
fedcare local schema-check --data ./demo/site_a.csv --contract ./examples/contract.json
fedcare local validate --model ./demo/model.joblib --data ./demo/site_a.csv --contract ./examples/contract.json
fedcare local train --model ./demo/model.joblib --data ./demo/site_a.csv --contract ./examples/contract.json --output ./demo/site_a.npz
fedcare local train --model ./demo/model.joblib --data ./demo/site_b.csv --contract ./examples/contract.json --output ./demo/site_b.npz
fedcare local aggregate ./demo/site_a.npz ./demo/site_b.npz --sample-count 100 --sample-count 100 --output ./demo/average.npz
fedcare local export-model --model ./demo/model.joblib --weights ./demo/average.npz --output ./demo/final.joblib
fedcare local validate --model ./demo/final.joblib --data ./demo/site_a.csv --contract ./examples/contract.json
```

Omit all `--sample-count` options for equal site weighting. Otherwise provide one positive integer per update, in input order. Use `--framework keras` on validation, training, and export commands for Keras models. For another training round, train each site from the exported model and aggregate the new updates.

Contracts are JSON documents matching `DataContract`; see [the example](examples/contract.json). Feature order follows `columns`, excluding the target. All declared model columns must exist during validation and training, and preprocessing must already be applied. For regression, set `prediction_type` to `regression`, declare a numeric target, and choose `mae`, `mse`, `rmse`, or `r2`.

Standalone commands do not read platform settings, task caches, or neighboring `weights.npz` files. Use `export-model` to apply a specific update. Only load trusted model files: sklearn joblib/pickle deserialization can execute code. Numeric updates are not encrypted, and this implementation does not provide cryptographic secure aggregation or differential privacy.

## Python API

```python
from pathlib import Path
from fedcareai.api import validate_model, train_model, aggregate_updates, export_model
from fedcareai.schemas.data_contract import DataContract

contract = DataContract.model_validate_json(Path("examples/contract.json").read_text())
metrics = validate_model(Path("demo/model.joblib"), Path("demo/site_a.csv"), contract, "sklearn")
training = train_model(Path("demo/model.joblib"), Path("demo/site_a.csv"), contract,
                       "sklearn", Path("demo/python_site.npz"), epochs=1)
aggregate_updates([Path("demo/python_site.npz")], Path("demo/python_average.npz"))
export_model(Path("demo/model.joblib"), Path("demo/python_average.npz"),
             "sklearn", Path("demo/python_final.joblib"))
```

`validate_model` returns metric values and `row_count`. `train_model` returns `row_count`, `artifact_path`, `framework`, and `epochs`. Aggregation and export return the output `Path`. Inputs are local files; none of these functions connects to a platform. They refuse to overwrite outputs by default. See [the versioned interchange specification](docs/protocol-v1.md).

## License and project boundary

This repository's source, tests, examples, and documentation are licensed under **GNU GPL version 3 only**, `GPL-3.0-only`. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The FedCareAI platform's frontend and backend are maintained separately and are not included in this package. See [the separation guide](docs/project-boundary.md).

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m pip install build
python -m build
```

The tests include standalone Python and CLI workflows with platform configuration and network connections disabled. Keras adapter tests use lightweight model doubles; running real TensorFlow models requires the `keras` extra.

## Install and connect

Download **FedCareAI source ZIP** from the platform's **Connect Locally** tab and extract it. Activate a Python virtual environment, then open a terminal in the extracted `fedcareai` folder:

```powershell
python -m pip install .
fedcare auth login --api-url http://127.0.0.1:8000/api/v1 --email you@example.org
```

Use your platform's URL in place of `http://127.0.0.1:8000` when connecting to a hosted platform. Use Python 3.10 or newer and Python/framework versions compatible with the model author. Installation downloads the package's dependencies; FedCareAI itself is installed from the extracted source.

For development from the folder containing both projects:

```powershell
pip install -e ./fedcareai
fedcare auth login --api-url http://127.0.0.1:8000/api/v1 --email you@example.org
```

Login prompts for your platform password. An existing access token can be supplied with `--token`. The URL must include `/api/v1`. Settings and the access token are stored under `~/.fedcareai`; logout removes the saved token. Task files live under `~/.fedcareai/tasks`.

Ask the owner to add your registered account email to the project. List and run tasks:

```powershell
fedcare task list
fedcare task pull TASK_ID
fedcare task schema-check TASK_ID --data ./local_data.csv
fedcare task validate TASK_ID --data ./local_data.csv
fedcare task preview-submit TASK_ID
fedcare task submit TASK_ID
```

Follow the project's preprocessing instructions. Numeric features use the order declared in the data contract; extra CSV columns are excluded. The target column and required features must be present. The library validates downloaded model and aggregate checksums. Only pull and load models from authors you trust: sklearn joblib/pickle files can execute code when loaded.

## Continuous prediction (regression)

Create a task with **Regression (continuous values)** as the prediction type and a numeric target column. The same `fedcare task validate` command evaluates sklearn regressors or single-output Keras regression models using MAE, MSE, RMSE, and R². Continuous predictions are preserved without class thresholds. MAE/MSE/RMSE are better when lower; R² is better when higher and can be negative. R² requires at least two rows. Targets must be finite numbers without missing values; multi-output regression is not supported.

For federated regression, use an already fitted incremental linear regressor such as `SGDRegressor`, or a Keras model compiled with an appropriate regression loss. Use the same preprocessing, units, and framework versions at every site.

## Federated training

For a federated training task, use `aggregate` instead of `validate`:

```powershell
fedcare task pull TASK_ID
fedcare task aggregate TASK_ID --data ./local_data.csv --epochs 1 --sample-count-visibility exact
fedcare task preview-submit TASK_ID
fedcare task submit TASK_ID --result-type federated_aggregation_update
```

The local command trains the model and exports numeric `.npz` weights. The project owner averages contributions on the platform. Pull the task again for each new round; pulling clears previous local result payloads to prevent accidental reuse. The original model is combined with the latest aggregated weights when it is loaded.

Supported training models are Keras/TensorFlow models and already fitted incremental sklearn linear estimators with `partial_fit`, `coef_`, and `intercept_`, such as `SGDClassifier` or `SGDRegressor`. Other sklearn estimators, including tree ensembles, support validation only. Keras requires compatible TensorFlow/Python versions and preprocessing. This version supports classification and single-target regression on tabular data.

Sample counts can be `exact`, `rounded` (the default, nearest ten), or `hidden`. The server uses visible positive counts for weighting; if any count is hidden, all sites have equal weight. For fewer than six samples, use exact or hidden counts to avoid a zero rounded weight.

After the owner completes the final round, pull and export the complete model:

```powershell
fedcare task pull TASK_ID
fedcare task export-model TASK_ID --output ./aggregated_model.joblib
```

Use `.keras` or `.h5` for Keras. You can upload this exported model to a new platform validation task. The platform does not load Python model objects to aggregate them. Numeric weights are visible to the platform, and this protocol does not provide cryptographic secure aggregation or differential privacy.

## Synthetic example

```powershell
python fedcareai/examples/create_demo.py --output ./demo
```

Upload `demo/model.joblib`, create a task with target `outcome` and numeric features `age, bmi`, and use `demo/site_a.csv` and `demo/site_b.csv` on separate registered accounts. The files contain synthetic data only.
