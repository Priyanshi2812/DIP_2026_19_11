# Shared evaluation for SAR, optical and fused models.
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE = r"C:\Users\Hp\OneDrive\Documents\sar_project"


def load_labels():
    master = pd.read_csv(os.path.join(BASE, "data", "master_table.csv"))
    folds = pd.read_csv(os.path.join(BASE, "data", "folds.csv"))
    return master[["field_id", "country", "label"]].merge(
        folds[["field_id", "fold"]], on="field_id"
    )


def evaluate_models(feats, cols, models, tag):
    """Evaluate each model on the provided spatial folds and save its predictions."""
    data = load_labels().merge(
        feats[["field_id"] + cols], on="field_id"
    ).reset_index(drop=True)

    results_dir = os.path.join(BASE, "results")
    os.makedirs(results_dir, exist_ok=True)

    labels = sorted(data.label.unique())
    rows = []

    for name, model_factory in models.items():
        predictions = np.empty(len(data), dtype=object)

        for fold in sorted(data.fold.unique()):
            train_mask = data.fold != fold
            test_mask = data.fold == fold
            model = model_factory()
            model.fit(data.loc[train_mask, cols], data.loc[train_mask, "label"])
            predictions[test_mask.to_numpy()] = model.predict(data.loc[test_mask, cols])

        prediction_table = pd.DataFrame({
            "field_id": data["field_id"],
            "label": data["label"],
            "pred": predictions,
            "fold": data["fold"],
        })
        prediction_table.to_csv(
            os.path.join(results_dir, f"pred_{tag}_{name}.csv"), index=False
        )

        row = {
            "set": tag,
            "model": name,
            "n_fields": len(data),
            "accuracy": accuracy_score(data.label, predictions),
            "macro_f1": f1_score(data.label, predictions, average="macro"),
            "weighted_f1": f1_score(data.label, predictions, average="weighted"),
        }
        rows.append(row)

        confusion = confusion_matrix(data.label, predictions, labels=labels)
        pd.DataFrame(confusion, index=labels, columns=labels).to_csv(
            os.path.join(results_dir, f"confusion_{tag}_{name}.csv")
        )
        print(
            f"{tag:8s} {name:20s} acc {row['accuracy']:.3f}  "
            f"macroF1 {row['macro_f1']:.3f}  weightedF1 {row['weighted_f1']:.3f}"
        )

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(os.path.join(results_dir, f"results_{tag}.csv"), index=False)
    return result_frame
