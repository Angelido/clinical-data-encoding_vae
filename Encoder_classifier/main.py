import os
import warnings
from time import time
from typing import List

import numpy as np
import torch
from joblib import Memory
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

from classifier import ClassifierBinary
from modelEncoderDecoderAndvancedV3VAE import MIEOVAE
from utilsData import load_known_unknown, preprocess_known_unknown_split, set_cpu

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


def _pos_weight_from_train_labels(y_train: torch.Tensor) -> float:
    pos = float(y_train.sum().item())
    neg = float(y_train.shape[0] - pos)
    return neg / max(pos, 1.0)


def run():
    device = set_cpu()
    torch.manual_seed(42)

    # ----------------------------------------------------------------------
    # Load raw labeled/unlabeled data, then define the split strategy here.
    #
    # Keeping the split logic in main makes it easy to switch to StratifiedKFold in the future:
    # - first create a hold-out test split on labeled data
    # - then run StratifiedKFold on the remaining labeled dev set
    # - call preprocess_known_unknown_split per fold (or move preprocessing into a Pipeline step)
    # ----------------------------------------------------------------------
    years = 8
    dataset_known, dataset_unknown = load_known_unknown(years=years)
    y_all = dataset_known.iloc[:, -1].to_numpy()
    all_idx = np.arange(len(y_all))
    dev_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=42, stratify=y_all
    )
    train_idx, val_idx = train_test_split(
        dev_idx, test_size=0.2, random_state=42, stratify=y_all[dev_idx]
    )
    data_dict = preprocess_known_unknown_split(
        dataset_known=dataset_known,
        dataset_unknown=dataset_unknown,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    tr_data = data_dict["tr_data"]
    tr_out = data_dict["tr_out"]
    val_data = data_dict["val_data"]
    val_out = data_dict["val_out"]
    test_data = data_dict["test_data"]
    test_out = data_dict["test_out"]
    X_unlabeled = data_dict.get("tr_unlabled")
    binary_cols = data_dict["bin_col"]

    input_dim = tr_data.shape[1]
    data_dim = input_dim // 2  # because preprocessing appends a same-size mask

    # Build dev set (train + val) with a predefined split for CV
    X_dev = torch.cat((tr_data, val_data), dim=0).cpu().numpy()
    y_dev = torch.cat((tr_out, val_out), dim=0).cpu().numpy()
    test_fold = np.concatenate(
        (
            np.full(tr_data.shape[0], -1, dtype=int),  # train indices
            np.zeros(val_data.shape[0], dtype=int),  # validation fold = 0
        )
    )
    cv_split = PredefinedSplit(test_fold=test_fold)

    X_test = test_data.cpu().numpy()
    y_test = test_out.cpu().numpy()
    X_unlabeled_np = None if X_unlabeled is None else X_unlabeled.cpu().numpy()

    pos_weight = _pos_weight_from_train_labels(tr_out)

    memory = Memory(location="_pipe_cache_", verbose=0)

    vae_estimator = MIEOVAE(
        module__data_dim=data_dim,
        module__mask_dim=data_dim,
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
        module__inputSize=data_dim,  # overridden by grid
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
    param_grid = build_param_grid(latent_dims, baseline_input_dim=input_dim)

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
    grid.fit(X_dev, y_dev, vae__X_unlabeled=X_unlabeled_np)
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
