# Shared evaluation for SAR, Optical and Fusion models
import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_models(data, cols, models, tag):
    """
    Evaluate models using 5-fold spatial cross-validation.
    Save predictions, metrics, confusion matrices and classification reports.
    """

    data = data.copy().reset_index(drop=True)

    labels = sorted(data["label"].unique())
    rows = []

    # Validate input
    assert data["field_id"].is_unique, "Duplicate field IDs found!"
    assert data["fold"].notna().all(), "Missing fold assignments!"
    assert data["label"].notna().all(), "Missing labels!"

    for name, model_factory in models.items():

        predictions = np.empty(len(data), dtype=object)

        # Five-fold spatial cross-validation
        for fold in sorted(data["fold"].unique()):

            train_mask = data["fold"] != fold
            test_mask = data["fold"] == fold

            model = model_factory()

            # Imputation and scaling are performed inside the model pipeline
            model.fit(
                data.loc[train_mask, cols],
                data.loc[train_mask, "label"]
            )

            predictions[test_mask.to_numpy()] = model.predict(
                data.loc[test_mask, cols]
            )

        y_true = data["label"].to_numpy()

        # Save out-of-fold predictions
        prediction_table = pd.DataFrame({
            "field_id": data["field_id"],
            "label": y_true,
            "pred": predictions,
            "fold": data["fold"]
        })

        prediction_table.to_csv(
            os.path.join(
                RESULTS_DIR,
                f"pred_{tag}_{name}.csv"
            ),
            index=False
        )

        # Calculate evaluation metrics
        row = {
            "Approach": tag,
            "Model": name,
            "Fields": len(data),
            "Accuracy": accuracy_score(y_true, predictions),
            "Precision": precision_score(
                y_true, predictions,
                average="macro",
                zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions,
                average="macro",
                zero_division=0
            ),
            "Macro_F1": f1_score(
                y_true, predictions,
                average="macro"
            ),
            "Weighted_F1": f1_score(
                y_true, predictions,
                average="weighted"
            )
        }

        rows.append(row)

        # Save confusion matrix
        confusion = confusion_matrix(
            y_true, predictions, labels=labels
        )

        pd.DataFrame(
            confusion,
            index=labels,
            columns=labels
        ).to_csv(
            os.path.join(
                RESULTS_DIR,
                f"confusion_{tag}_{name}.csv"
            )
        )

        # Save detailed classification report
        report = classification_report(
            y_true,
            predictions,
            labels=labels,
            zero_division=0,
            output_dict=True
        )

        pd.DataFrame(report).transpose().to_csv(
            os.path.join(
                RESULTS_DIR,
                f"classification_{tag}_{name}.csv"
            )
        )

        print(
            f"{tag} | {name} | "
            f"Accuracy: {row['Accuracy']:.4f} | "
            f"Precision: {row['Precision']:.4f} | "
            f"Recall: {row['Recall']:.4f} | "
            f"Macro-F1: {row['Macro_F1']:.4f} | "
            f"Weighted-F1: {row['Weighted_F1']:.4f}"
        )

    # Save combined results
    result_frame = pd.DataFrame(rows)

    result_frame.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"results_{tag}.csv"
        ),
        index=False
    )

    return result_frame