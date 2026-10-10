# FedCareAI Python library

FedCareAI is a Python library for federated machine learning and external model validation, designed for healthcare research and applicable to other domains. It supports local training and validation of tabular models, aggregation of model updates across participating sites, and export of complete models while keeping raw datasets local.

Use it directly from Python or through the `fedcare local` CLI without an account or server.

## Installation

Use Python 3.10 or newer. Clone the public repository and install the library, preferably in a virtual environment:

```powershell
git clone https://github.com/FedCareAI/fedcareai.git
cd fedcareai
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

## Supported models

Validation supports sklearn estimators and Keras/TensorFlow models on preprocessed tabular data. Federated training supports Keras models and already fitted incremental sklearn linear estimators with `partial_fit`, `coef_`, and `intercept_`, such as `SGDClassifier` and `SGDRegressor`. Other sklearn estimators, including tree ensembles, support validation only.

Classification and single-target regression are supported. Regression preserves continuous predictions and reports MAE, MSE, RMSE, and R-squared. Error metrics are better when lower; R-squared is better when higher and can be negative. Targets must be finite numbers without missing values; R-squared requires at least two rows. Keras training models must be compiled with an appropriate loss. All sites must use the same preprocessing, feature order, units, and compatible framework versions.

## Optional integration

The `fedcare task` commands can connect to a separately maintained FedCareAI platform for coordination across sites. This is optional and is not needed for the workflows above. See the [platform integration guide](docs/platform-integration.md) for authentication, task downloads, and result submission.

## License

This repository's source, tests, examples, and documentation are licensed under **GNU GPL version 3 only**, `GPL-3.0-only`. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The FedCareAI platform's frontend and backend are maintained separately and are not included in this package. See [the separation guide](docs/project-boundary.md).

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m pip install build
python -m build
```

The tests include standalone Python and CLI workflows with platform configuration and network connections disabled. Keras adapter tests use lightweight model doubles; running real TensorFlow models requires the `keras` extra.
