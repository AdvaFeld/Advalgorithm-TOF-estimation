

'''
Phase 2:
load the three models trained in Phase 1 and fine-tune them jointly.

The classifier produces one logit for each sample.
A sigmoid converts the logit into

    p_non_bl = P(non-baseline | y)

and

    p_bl = 1 - p_non_bl.

Both DU experts are evaluated and their outputs are softly combined:

    p_hat = p_bl * p_hat_baseline
          + p_non_bl * p_hat_super_resolution

The ToF loss is computed from the mixed output p_hat.
The classifier is directly supervised using BCEWithLogitsLoss.
Because the routing is differentiable, the ToF loss can also
propagate gradients through the classifier probabilities.

'''

#Differentiable unified model for phase 2:
import csv
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm

from tof_dataset import TOFTrainingDataset
from model import DeepUnfoldedSPICE, DeepUnfoldedSPICE_piecewise, Classifier_MLP, UnifiedModel
from loss_func import p_rmse_windowed_tof_loss
from model_config import (DEVICE, CHECKPOINT_DIR, CHECKPOINT_CLASSIFIER_DIR,
                          CHECKPOINT_UNIFIED_DIR, TRAIN_LOG_NAME,
                          VALIDATION_SET_NAME, model_max_iterations, g)


def train_phase2():

    # Phase-1 checkpoints
    baseline_run_name = "40K_baseline_small_q_p_rmse"
    baseline_run_epoch = 49

    super_res_run_name = "40K_piecewise_q_p_rmse"
    super_res_run_epoch = 49

    classifier_run_name = "classifier_cart_90K_sigmoid"
    classifier_run_epoch = 5

    # Phase-2 settings
    run_name = "phase2_joint_soft_routing_sigmoid"
    batch_size = 16
    validation_batch_size = 16 * 16
    num_epochs = 30
    du_lr = 1e-4
    classifier_lr = 1e-4

    num_samp_train = 40 * 1000
    num_samp_val = 5 * 1000
    num_tot = num_samp_train + num_samp_val

    dataset_baseline = TOFTrainingDataset(
        data_dir="dataset_baseline",
        data_size=num_tot
    )

    dataset_nonbaseline = TOFTrainingDataset(
        data_dir="dataset_rest_of_scenarios",
        data_size=num_tot
    )

    train_ids = list(range(num_samp_train))
    val_ids = list(range(num_samp_train, num_tot))

    train_baseline = Subset(dataset_baseline, train_ids)
    train_nonbaseline = Subset(dataset_nonbaseline, train_ids)

    val_baseline = Subset(dataset_baseline, val_ids)
    val_nonbaseline = Subset(dataset_nonbaseline, val_ids)

    train_dataset = ConcatDataset([
        train_baseline,
        train_nonbaseline
    ])

    val_dataset = ConcatDataset([
        val_baseline,
        val_nonbaseline
    ])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False
    )

    # Phase-2 checkpoint folder
    check_dir = f"{CHECKPOINT_UNIFIED_DIR}/{run_name}"
    os.makedirs(check_dir, exist_ok=True)
    print("Phase-2 checkpoint directory:", check_dir)

    # Save the validation file names
    valid_file_names = (
        [dataset_baseline.file_paths[i] for i in val_ids]
        + [dataset_nonbaseline.file_paths[i] for i in val_ids]
    )

    with open(f"{check_dir}/{VALIDATION_SET_NAME}", "wb") as file:
        pickle.dump(valid_file_names, file)

    # Create the three model architectures
    baseline_model = DeepUnfoldedSPICE(
        max_iter=model_max_iterations
    )

    super_res_model = DeepUnfoldedSPICE_piecewise(
        max_iter=model_max_iterations
    )

    classifier_model = Classifier_MLP()

    # Load the trained Phase-1 parameters
    baseline_checkpoint = (
        f"{CHECKPOINT_DIR}/{baseline_run_name}/"
        f"epoch_{baseline_run_epoch}.pt"
    )

    super_res_checkpoint = (
        f"{CHECKPOINT_DIR}/{super_res_run_name}/"
        f"epoch_{super_res_run_epoch}.pt"
    )

    classifier_checkpoint = (
        f"{CHECKPOINT_CLASSIFIER_DIR}/{classifier_run_name}/"
        f"epoch_{classifier_run_epoch}.pt"
    )

    baseline_model.load_state_dict(
        torch.load(baseline_checkpoint, map_location=DEVICE)
    )

    super_res_model.load_state_dict(
        torch.load(super_res_checkpoint, map_location=DEVICE)
    )

    classifier_model.load_state_dict(
        torch.load(classifier_checkpoint, map_location=DEVICE)
    )

    # Combine the three pretrained models into the complete system
    model = UnifiedModel(
        baseline_model,
        super_res_model,
        classifier_model
    ).to(DEVICE)

    classifier_loss_fn = nn.BCEWithLogitsLoss()

# Separate optimizers allow independent learning rates for the DU experts and classifier.
# The DU experts are updated through
# the power-profile loss, while the classifier receives gradients from both the power-profile loss and BCE.
    optimizer_du = optim.Adam(
        list(model.baseline_model.parameters())
        + list(model.super_res_model.parameters()),
        lr=du_lr
    )

    optimizer_classifier = optim.Adam(
        model.classifier.parameters(),
        lr=classifier_lr
    )

    log_file_path = f"{check_dir}/{TRAIN_LOG_NAME}"

    with open(log_file_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "avg_train_tof_loss",
            "avg_train_classifier_loss",
            "train_classifier_accuracy",
            "train_mean_confidence",
            "avg_val_tof_loss",
            "avg_val_classifier_loss",
            "val_classifier_accuracy",
            "val_mean_confidence"
        ])

    print("Started Phase-2 Training")

    for epoch in range(num_epochs):

        model.train()

        train_tof_sum = 0.0
        train_cls_sum = 0.0
        train_correct = 0
        train_confidence_sum = 0.0
        train_num_samples = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}"
        )

        for batch in progress_bar:

            optimizer_du.zero_grad()
            optimizer_classifier.zero_grad()

            p_hat, logits, p_non_bl = model(
                batch["y_noisy"],
                batch["A_aug"],
                batch["p0"],
                return_aux=True
            )

            tof_loss = p_rmse_windowed_tof_loss(
                p_hat=p_hat,
                P_true_sig=batch["P_true_sig"],
                true_tau_idx=batch["true_tau_idx"]
            )
            targets=(batch["scenario_label"].float().unsqueeze(1))
            classifier_loss = classifier_loss_fn(logits,targets)

            # The power-profile loss trains both DU experts through the soft fusion
            # and also backpropagates through the classifier probability.
            # BCE provides additional direct supervision to the classifier.
            loss = tof_loss + classifier_loss

            loss.backward()

            optimizer_du.step()
            optimizer_classifier.step()

            current_batch_size = batch["y_noisy"].shape[0]

            train_tof_sum += tof_loss.item() * current_batch_size
            train_cls_sum += classifier_loss.item() * current_batch_size

            preds = (p_non_bl>=0.5).long().squeeze(1)

            train_correct += (
                preds == batch["scenario_label"]
            ).sum().item()

            confidence = torch.maximum(p_non_bl,1.0-p_non_bl).squeeze(1)
            train_confidence_sum += confidence.sum().item()

            train_num_samples += current_batch_size

        avg_train_tof_loss = train_tof_sum / train_num_samples
        avg_train_classifier_loss = train_cls_sum / train_num_samples
        train_classifier_accuracy = train_correct / train_num_samples
        train_mean_confidence = train_confidence_sum / train_num_samples

        model.eval()

        val_tof_sum = 0.0
        val_cls_sum = 0.0
        val_correct = 0
        val_confidence_sum = 0.0
        val_num_samples = 0

        with torch.no_grad():

            for batch in val_loader:

                p_hat, logits, p_non_bl = model(
                    batch["y_noisy"],
                    batch["A_aug"],
                    batch["p0"],
                    return_aux=True
                )

                tof_loss = p_rmse_windowed_tof_loss(
                    p_hat=p_hat,
                    P_true_sig=batch["P_true_sig"],
                    true_tau_idx=batch["true_tau_idx"]
                )
                targets=batch["scenario_label"].float().unsqueeze(1)
                classifier_loss = classifier_loss_fn(logits,targets)
                current_batch_size = batch["y_noisy"].shape[0]

                val_tof_sum += tof_loss.item() * current_batch_size
                val_cls_sum += classifier_loss.item() * current_batch_size

                preds =(p_non_bl>=0.5).long().squeeze(1)

                val_correct += (
                    preds == batch["scenario_label"]
                ).sum().item()

                confidence = torch.maximum(p_non_bl,1.0-p_non_bl).squeeze(1)
                val_confidence_sum += confidence.sum().item()

                val_num_samples += current_batch_size

        avg_val_tof_loss = val_tof_sum / val_num_samples
        avg_val_classifier_loss = val_cls_sum / val_num_samples
        val_classifier_accuracy = val_correct / val_num_samples
        val_mean_confidence = val_confidence_sum / val_num_samples

        # Save all three fine-tuned models together
        save_path = f"{check_dir}/epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path)

        with open(log_file_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                avg_train_tof_loss,
                avg_train_classifier_loss,
                train_classifier_accuracy,
                train_mean_confidence,
                avg_val_tof_loss,
                avg_val_classifier_loss,
                val_classifier_accuracy,
                val_mean_confidence
            ])

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train ToF: {avg_train_tof_loss:.6f} | "
            f"Train Cls: {avg_train_classifier_loss:.6f} | "
            f"Train Acc: {100*train_classifier_accuracy:.2f}% | "
            f"Val ToF: {avg_val_tof_loss:.6f} | "
            f"Val Cls: {avg_val_classifier_loss:.6f} | "
            f"Val Acc: {100*val_classifier_accuracy:.2f}% | "
            f"Val Confidence: {val_mean_confidence:.4f}"
        )


if __name__ == "__main__":
    train_phase2()