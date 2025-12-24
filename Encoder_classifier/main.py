import os
import warnings
from time import time
from typing import List

import numpy as np
import torch
from joblib import Memory
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

from classifier import ClassifierBinary
from modelEncoderDecoderAndvancedV3VAE import MIEOVAE
from utilsData import load_dataset, set_cpu

warnings.filterwarnings("ignore", category=UserWarning)


CLF_GRID_COMMON = {
    "clf__lr": [1e-3, 3e-4],
    "clf__max_epochs": [50],
    "clf__batch_size": [128],
}

VAE_GRID_COMMON = {
    "vae__lr": [1e-3, 3e-4],
    "vae__max_epochs": [50],
    "vae__batch_size": [256],
    "vae__module__mask_percentage": [0.1, 0.2],
    "vae__beta": [0.1, 1.0],
    "vae__binary_weight": [0.5, 1.0],
}


def _merged(*parts: dict) -> dict:
    merged = {}
    for part in parts:
        merged.update(part)
    return merged


def build_param_grid(latent_dims: List[int], baseline_input_dim: int) -> List[dict]:
    """
    Tie classifier input size to VAE latent size by creating one grid dict per latent_dim.
    """
    grid_list = []
    for ld in latent_dims:
        grid_list.append(
            _merged(
                VAE_GRID_COMMON,
                CLF_GRID_COMMON,
                {
                    "vae__module__latent_dim": [ld],
                    "clf__module__inputSize": [ld],
                },
            )
        )
    # Baseline: no encoder, classifier sees [values | null_mask] directly.
    grid_list.append(
        _merged(
            CLF_GRID_COMMON,
            {
                "vae": ["passthrough"],
                "clf__module__inputSize": [baseline_input_dim],
            },
        )
    )
    return grid_list

def run():
    device = set_cpu()
    torch.manual_seed(42)

    # ----------------------------------------------------------------------
    # Load full labeled/unlabeled tensors.
    #
    # The split/CV strategy is defined here (not in the loader) so you can easily switch between:
    # - holdout validation (PredefinedSplit)
    # - StratifiedKFold / RepeatedStratifiedKFold
    # - nested CV, etc.
    # ----------------------------------------------------------------------
    data = load_dataset(years=8, test_size=0.2, random_state=42,unlabeled=True)
 
    X_dev = data["X_dev"]
    y_dev = data["y_dev"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_unlabeled = data["X_unlabeled"]
    binary_cols = data["binary_cols"]

    cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pos = float(y_dev.sum())
    neg = float(len(y_dev) - pos)
    pos_weight = neg / max(pos, 1.0)

    memory = Memory(location="_pipe_cache_", verbose=0)

    vae_estimator = MIEOVAE(
        module__data_dim=X_dev.shape[1],
        module__mask_dim=X_dev.shape[1],
        module__binary=binary_cols,
        module__hidden_dims=[256, 128, 64],
        optimizer=torch.optim.Adam,
        lr=1e-3,
        batch_size=256,
        max_epochs=50,
        device=device,
        verbose=0,
    )

    clf_estimator = NeuralNetClassifier(
        module=ClassifierBinary,
        module__inputSize=X_dev.shape[1],  # overridden by grid
        optimizer=torch.optim.Adam,
        lr=1e-3,
        batch_size=128,
        max_epochs=50,
        device=device,
        criterion=torch.nn.BCEWithLogitsLoss,
        criterion__pos_weight=torch.tensor([pos_weight], device=device),
        iterator_train__shuffle=True,
        callbacks=[("early_stopping", EarlyStopping(patience=8))],
        verbose=0,
    )

    latent_dims = [8, 16, 32]
    param_grid = build_param_grid(latent_dims, baseline_input_dim=X_dev.shape[1])

    pipe = Pipeline(steps=[("vae", vae_estimator), ("clf", clf_estimator)], memory=memory)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv_split,
        scoring="balanced_accuracy",
        n_jobs=1,  # safer for DL workloads
        refit=True,
        verbose=1,
    )

    begin = time()
    grid.fit(X_dev, y_dev, vae__X_unlabeled=X_unlabeled)
    fit_time = time() - begin

    best_model = grid.best_estimator_
    y_pred_test = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)

    os.makedirs("./Encoder_classifier/gridResults", exist_ok=True)
    output = {
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "test_report": report,
        "fit_time_sec": fit_time,
    }
    save_path = "./Encoder_classifier/gridResults/last_results.json"
    with open(save_path, "w") as f:
        import json

        json.dump(output, f, indent=4)

    print(f"Fit completed in {fit_time/60:.1f} min")
    print(f"Best CV balanced accuracy: {grid.best_score_:.4f}")
    print("Best params:", grid.best_params_)
    print("Test report:")
    for k, v in report.items():
        if isinstance(v, dict) and "f1-score" in v:
            print(f"  {k}: f1={v['f1-score']:.3f}, precision={v['precision']:.3f}, recall={v['recall']:.3f}")


if __name__ == "__main__":
    run()
