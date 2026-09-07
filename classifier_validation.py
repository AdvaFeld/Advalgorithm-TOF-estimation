import os
import csv

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")  # needed on the cluster: save plots without opening a window
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset, ConcatDataset

from tof_dataset import TOFTrainingDataset
from model import Classifier_MLP
from model_config import DEVICE, CHECKPOINT_CLASSIFIER_DIR, TRAIN_LOG_NAME

# ============================================================
# SETTINGS
# ============================================================

RUN_NAME = "classifier_cart_90K_sigmoid"

# We want to compare:
# epoch 5  -> lowest validation loss during validation
# epoch 31 -> local minimum for validation and lower loss in train than 5
EPOCHS_TO_EVALUATE = [5, 31]

NUM_TRAIN_PER_GROUP = 40 * 1000
NUM_VAL_PER_GROUP = 5 * 1000
TOTAL_PER_GROUP = NUM_TRAIN_PER_GROUP + NUM_VAL_PER_GROUP

VALIDATION_BATCH_SIZE = 256 #16*16


# ============================================================
# CREATE THE EXACT SAME VALIDATION SET USED DURING TRAINING
# ============================================================
def create_validation_loader():

    # Non-baseline dataset:
    # moderate + close + very_close
    dataset_nonbaseline = TOFTrainingDataset(
        data_dir="dataset_rest_of_scenarios",
        data_size=TOTAL_PER_GROUP)

    # Baseline dataset
    dataset_baseline = TOFTrainingDataset(
        data_dir="dataset_baseline",
        data_size=TOTAL_PER_GROUP)

    # Same indices as in model_train.py:
    # first 40k = training
    # last 5k  = validation
    validation_ids = list(
        range(NUM_TRAIN_PER_GROUP, TOTAL_PER_GROUP))

    val_nonbaseline = Subset(
        dataset_nonbaseline,
        validation_ids)

    val_baseline = Subset(
        dataset_baseline,
        validation_ids)

    # Same idea as training:
    # 5k nonbaseline + 5k baseline = 10k total validation samples
    validation_dataset = ConcatDataset(
        [val_nonbaseline, val_baseline])

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False)

    print("Validation set:")
    print("  Non-baseline samples:", len(val_nonbaseline))
    print("  Baseline samples:", len(val_baseline))
    print("  Total samples:", len(validation_dataset))
    print()

    return validation_loader


# ============================================================
# EVALUATE ONE CHECKPOINT
# ============================================================

def evaluate_checkpoint(epoch, validation_loader):

    run_path = os.path.join(
        CHECKPOINT_CLASSIFIER_DIR,
        RUN_NAME)

    checkpoint_path = os.path.join(
        run_path,
        "epoch_{}.pt".format(epoch))

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Checkpoint does not exist: {}".format(checkpoint_path))

    print("=" * 60)
    print("Evaluating epoch {}".format(epoch))
    print("Checkpoint:", checkpoint_path)

    # Create classifier architecture
    model = Classifier_MLP()

    # Load the trained weights
    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    # Confusion matrix:
    #
    #                 predicted
    #                0         1
    # true 0       [0,0]     [0,1]
    # true 1       [1,0]     [1,1]
    #
    # class 0 = baseline
    # class 1 = non-baseline
    confusion_matrix = np.zeros((2, 2), dtype=np.int64)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in validation_loader:

            y_noisy = batch["y_noisy"].to(DEVICE)
            true_labels = batch["scenario_label"].to(DEVICE)

            # Classifier output:
            # shape = [batch_size, 1]
            logits = model(y_noisy)

            # Convert the single logit to P(non-baseline)
            p_non_bl = torch.sigmoid(logits)

            # p >= 0.5 -> non-baseline (1)
            # p <  0.5 -> baseline (0)
            predicted_labels = (p_non_bl >= 0.5).long().squeeze(1)

            total_correct += (predicted_labels == true_labels).sum().item()

            total_samples += true_labels.shape[0]

            true_np = true_labels.detach().cpu().numpy()
            pred_np = predicted_labels.detach().cpu().numpy()

            for true_label, predicted_label in zip(
                true_np,
                pred_np):
                confusion_matrix[
                    int(true_label),
                    int(predicted_label)
                ] += 1

    # ========================================================
    # ACCURACIES
    # ========================================================

    overall_accuracy = total_correct / float(total_samples)

    baseline_total = confusion_matrix[0, :].sum()
    nonbaseline_total = confusion_matrix[1, :].sum()

    baseline_accuracy = (
        confusion_matrix[0, 0] / float(baseline_total)
        if baseline_total > 0
        else 0.0
    )

    nonbaseline_accuracy = (
        confusion_matrix[1, 1] / float(nonbaseline_total)
        if nonbaseline_total > 0
        else 0.0
    )

    print()
    print("Results for epoch {}".format(epoch))
    print("--------------------------------")
    print(
        "Overall validation accuracy: {:.4f} ({:.2f}%)".format(
            overall_accuracy,
            100.0 * overall_accuracy
        )
    )

    print(
        "Baseline accuracy:           {:.4f} ({:.2f}%)".format(
            baseline_accuracy,
            100.0 * baseline_accuracy
        )
    )

    print(
        "Non-baseline accuracy:       {:.4f} ({:.2f}%)".format(
            nonbaseline_accuracy,
            100.0 * nonbaseline_accuracy
        )
    )

    print()
    print("Confusion matrix:")
    print("Rows = true class")
    print("Columns = predicted class")
    print()
    print(confusion_matrix)

    print()
    print("Interpretation:")
    print(
        "True baseline -> predicted baseline:       {}".format(
            confusion_matrix[0, 0]
        )
    )
    print(
        "True baseline -> predicted non-baseline:   {}".format(
            confusion_matrix[0, 1]
        )
    )
    print(
        "True non-baseline -> predicted baseline:   {}".format(
            confusion_matrix[1, 0]
        )
    )
    print(
        "True non-baseline -> predicted non-baseline: {}".format(
            confusion_matrix[1, 1]
        )
    )

    # ========================================================
    # SAVE CONFUSION MATRIX AS CSV
    # ========================================================

    csv_path = os.path.join(
        run_path,
        "confusion_matrix_epoch_{}.csv".format(epoch)
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "",
            "Predicted baseline",
            "Predicted nonbaseline"
        ])

        writer.writerow([
            "True baseline",
            confusion_matrix[0, 0],
            confusion_matrix[0, 1]
        ])

        writer.writerow([
            "True nonbaseline",
            confusion_matrix[1, 0],
            confusion_matrix[1, 1]
        ])

    # ========================================================
    # SAVE CONFUSION MATRIX FIGURE
    # ========================================================

    figure_path = os.path.join(
        run_path,
        "confusion_matrix_epoch_{}.png".format(epoch)
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(confusion_matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Baseline",
        "Non-baseline"
    ])

    ax.set_yticklabels([
        "Baseline",
        "Non-baseline"
    ])

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_title(
        "Classifier Confusion Matrix - Epoch {}".format(epoch)
    )

    # Put the numerical values inside the matrix
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(confusion_matrix[row, col]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print()
    print("Saved:")
    print(" ", csv_path)
    print(" ", figure_path)
    print()

    return {
        "epoch": epoch,
        "overall_accuracy": overall_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "nonbaseline_accuracy": nonbaseline_accuracy
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("Classifier validation")
    print("Device:", DEVICE)
    print()

    validation_loader = create_validation_loader()

    all_results = []

    for epoch in EPOCHS_TO_EVALUATE:

        result = evaluate_checkpoint(
            epoch,
            validation_loader
        )

        all_results.append(result)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    for result in all_results:

        print(
            "Epoch {:2d}: overall={:.2f}% | baseline={:.2f}% | nonbaseline={:.2f}%".format(
                result["epoch"],
                100.0 * result["overall_accuracy"],
                100.0 * result["baseline_accuracy"],
                100.0 * result["nonbaseline_accuracy"]
            )
        )

    # Save comparison summary
    run_path = os.path.join(
        CHECKPOINT_CLASSIFIER_DIR,
        RUN_NAME
    )

    loss_log_path = os.path.join(run_path, TRAIN_LOG_NAME)
    df = pd.read_csv(loss_log_path)
    plt.plot(df['epoch'], df['avg_train_loss'], color='tab:blue', label='avg_train_loss', linewidth=1.5, marker='o')
    plt.plot(df['epoch'], df['avg_val_loss'], color='tab:red', label='avg_val_loss', linewidth=1.5, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('avg_loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    figure_path = os.path.join(run_path, "train_val_loss_per_epoch.png")
    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    summary_path = os.path.join(
        run_path,
        "classifier_validation_summary.csv"
    )

    with open(summary_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "overall_accuracy",
            "baseline_accuracy",
            "nonbaseline_accuracy"
        ])

        for result in all_results:

            writer.writerow([
                result["epoch"],
                result["overall_accuracy"],
                result["baseline_accuracy"],
                result["nonbaseline_accuracy"]
            ])

    print()
    print("Summary saved to:")
    print(summary_path)


if __name__ == "__main__":
    main()