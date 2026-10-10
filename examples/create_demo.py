# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

"""Create a fitted incremental classifier and two synthetic collaborator datasets."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('demo'))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

def dataset(n):
    age = rng.normal(50, 12, n)
    bmi = rng.normal(27, 5, n)
    outcome = (0.04 * (age - 50) + 0.1 * (bmi - 27) + rng.normal(0, .7, n) > 0).astype(int)
    return pd.DataFrame({'age': age, 'bmi': bmi, 'outcome': outcome})

train = dataset(300)
model = SGDClassifier(loss='log_loss', random_state=42, learning_rate='constant', eta0=0.0001, max_iter=1000)
model.fit(train[['age', 'bmi']], train['outcome'])
joblib.dump(model, args.output/'model.joblib')
for site in ('site_a', 'site_b'):
    dataset(100).to_csv(args.output/f'{site}.csv', index=False)
print(f'Synthetic model and CSVs saved in {args.output.resolve()}')
