import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from tof_dataset import TOFTrainingDataset
from model import DeepUnfoldedSPICE, DeepUnfoldedSPICE_piecewise

from Adva_algo_implementation import run_likes
from Adva_algo_implementation import run_slim
from Adva_algo_implementation import run_spice
from Adva_algo_implementation import run_likes30m
from Adva_algo_implementation import run_slim5

from model_config import (model_max_iterations, TRAIN_LOG_NAME, DEVICE, RESULT_LOG_NAME, CHECKPOINT_DIR,)
from tof_eval_utils import p_hat2tau_hat


# ONLY PART TO CHANGE BETWEEN RUNS
model_type = "baseline"       # "baseline" or "piecewise"

if model_type == "baseline":
    model_class = DeepUnfoldedSPICE
    run_name = "40K_baseline_small_q_p_rmse"
    dataset_name = "dataset_baseline"

elif model_type == "piecewise":
    model_class = DeepUnfoldedSPICE_piecewise
    run_name = "40K_piecewise_q_p_rmse"
    dataset_name = "dataset_rest_of_scenarios"

else:
    raise ValueError("model_type must be 'baseline' or 'piecewise'")

run_epoch = 49
override_result = True

# DATA SPLIT

# samples:
# 0-39999       -> training
# 40000-44999   -> validation
# 45000+        -> test

test_start_idx = 45 * 1000

# PATHS
run_path = CHECKPOINT_DIR + "/" + run_name

checkpoint_path = f"{run_path}/epoch_{run_epoch}.pt"
loss_log_path = run_path + "/" + TRAIN_LOG_NAME
result_log_path = run_path + "/" + RESULT_LOG_NAME

plot_dir = os.path.join(run_path, "plots")
os.makedirs(plot_dir, exist_ok=True)


# ============================================================
# EVALUATION SETTINGS
test_batch_size = 16 * 16

models_names = ["unfolding", "spice", "likes", "slim", "likes30m", "slim5"]

# FIND TEST FILES
def get_test_file_paths():
    test_file_paths = []

    for file_name in os.listdir(dataset_name):

        if not file_name.endswith(".pkl"):
            continue

        # file names are expected to be:
        # 0.pkl, 1.pkl, ..., 45000.pkl, ...
        sample_idx = int(os.path.splitext(file_name)[0])

        if sample_idx >= test_start_idx:
            test_file_paths.append(os.path.join(dataset_name, file_name))

    # numerical sort:
    # 45000.pkl, 45001.pkl, ...
    test_file_paths.sort(
        key=lambda path: int(
            os.path.splitext(
                os.path.basename(path)
            )[0]
        )
    )

    if len(test_file_paths) == 0:
        raise RuntimeError(
            f"No test samples were found in {dataset_name} "
            f"with index >= {test_start_idx}"
        )

    print(
        f"Using {len(test_file_paths)} TEST samples "
        f"from '{dataset_name}'"
    )

    return test_file_paths


# ============================================================
# RUN MODELS ON TEST SET

def calc_models_test(model):
    """
    Runs the trained deep-unfolded model and the classical
    comparison algorithms on the held-out TEST samples.

    Test samples are all files whose index is >= 45000.
    """
    test_file_paths = get_test_file_paths()

    test_data = TOFTrainingDataset(
        file_paths=test_file_paths
    )

    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size,
        shuffle=False
    )

    # We are recalculating the entire test result file,
    # so remove the previous one first.
    if os.path.exists(result_log_path):
        os.remove(result_log_path)

    should_append = False

    with torch.no_grad():

        progress_bar = tqdm(
            test_loader,
            desc="Test"
        )

        for batch in progress_bar:

            models_result = dict()

            # ------------------------------------------------
            # Deep unfolded model

            models_result["unfolding_p"] = (
                model(
                    batch["y_noisy"],
                    batch["A_aug"],
                    batch["p0"]
                )
                .detach()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Classical algorithms

            models_result["spice_p"] = (
                run_spice(
                    batch["y_noisy"],
                    batch["A_aug"],
                    batch["p0"],
                    max_iter=200,
                    tol=1e-3
                )[0]
                .detach()
                .cpu()
                .numpy()
            )

            models_result["likes_p"] = (
                run_likes(
                    batch["y_noisy"],
                    batch["A_aug"],
                    batch["p0"],
                    max_iter=200,
                    tol=1e-3
                )[0]
                .detach()
                .cpu()
                .numpy()
            )

            models_result["slim_p"] = (
                run_slim(
                    batch["y_noisy"],
                    batch["A_aug"],
                    batch["p0"],
                    max_iter=200,
                    tol=1e-3
                )[0]
                .detach()
                .cpu()
                .numpy()
            )

            models_result["likes30m_p"] = (
                run_likes30m(
                    batch["y_noisy"],
                    batch["A_aug"],
                    batch["p0"],
                    max_iter=200,
                    tol=1e-3
                )[0]
                .detach()
                .cpu()
                .numpy()
            )

            models_result["slim5_p"] = (
                run_slim5(
                    batch["y_noisy"],
                    batch["A_aug"],
                    batch["p0"],
                    tol=1e-3
                )[0]
                .detach()
                .cpu()
                .numpy()
            )

            # Move batch information to CPU / NumPy
            batch = {
                k: (
                    v.detach().cpu().numpy()
                    if torch.is_tensor(v)
                    else v
                )
                for k, v in batch.items()
            }

            # ------------------------------------------------
            # Convert p estimates to ToF estimates
            # and calculate RMSE

            for model_name in models_names:

                models_result[f"{model_name}_rmse_tof"] = []
                models_result[f"{model_name}_rmse_first_path"] = []

                for i in range(batch["true_tau_ns"].shape[0]):
                    true_tau_ns = np.sort(batch["true_tau_ns"][i])
                    num_paths = true_tau_ns.shape[0]
                    tau, _ = p_hat2tau_hat(
                        models_result[f"{model_name}_p"][i],
                        batch["tau_grid"][i],
                        batch["K"][i],
                        num_paths)

                    tau_ns = np.sort(tau * 1e9)
                    models_result[f"{model_name}_rmse_tof"].append(np.sqrt(np.mean((tau_ns - true_tau_ns) ** 2)))
                    models_result[f"{model_name}_rmse_first_path"].append(np.sqrt((tau_ns[0] - true_tau_ns[0]) ** 2))

                # We do not need to store the complete p vector
                # in the parquet result file.
                del models_result[f"{model_name}_p"]

            # ------------------------------------------------
            # Metadata

            models_result["snr_db"] = batch["snr_db"]
            models_result["scenario_type"] = batch["scenario_type"]
            models_result["sample_idx"] = batch["sample_idx"]
            models_result = {
                k: list(v) if isinstance(v, np.ndarray) else v
                for k, v in models_result.items()}

            # ------------------------------------------------
            # Save result chunk
            chunk_df = pd.DataFrame(models_result)

            chunk_df.to_parquet(
                result_log_path,
                engine="fastparquet",
                append=should_append)

            should_append = True


# ============================================================
# PLOT TEST RESULTS

def calc_and_print_rmse(df):
    """
    Receives the TEST results of one scenario and calculates
    the mean RMSE for every SNR value.
    """

    scenario_name = str(df.name).replace(" ", "_")


    # --------------------------------------------------------
    # ALL-PATH RMSE

    plt.figure(figsize=(6, 5))

    for model_name in models_names:

        rmse_tof_per_snr = (
            df.groupby("snr_db")[f"{model_name}_rmse_tof"]
            .mean()
            .to_frame("rmse_tof")
            .reset_index()
        )

        plt.plot(
            rmse_tof_per_snr["snr_db"],
            rmse_tof_per_snr["rmse_tof"],
            label=model_name,
            marker="o"
        )

    plt.xlabel("SNR [dB]")
    plt.ylabel("Average ToF RMSE [ns]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.title(f"{df.name} - All Paths")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir,
            f"{scenario_name}_rmse_all_paths.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            plot_dir,
            f"{scenario_name}_rmse_all_paths.pdf"
        ),
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


    # --------------------------------------------------------
    # FIRST-PATH RMSE

    plt.figure(figsize=(6, 5))

    for model_name in models_names:

        rmse_first_path_per_snr = (
            df.groupby("snr_db")[
                f"{model_name}_rmse_first_path"
            ]
            .mean()
            .to_frame("rmse_first_path")
            .reset_index()
        )

        plt.plot(
            rmse_first_path_per_snr["snr_db"],
            rmse_first_path_per_snr["rmse_first_path"],
            label=model_name,
            marker="o"
        )

    plt.xlabel("SNR [dB]")
    plt.ylabel("Average First-Path RMSE [ns]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.title(f"{df.name} - First Path")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir,
            f"{scenario_name}_rmse_first_path.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            plot_dir,
            f"{scenario_name}_rmse_first_path.pdf"
        ),
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# MAIN RESULT ANALYSIS

def result_analysis():

    # --------------------------------------------------------
    # Load correct model architecture

    model = model_class(
        max_iter=model_max_iterations
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()


    # --------------------------------------------------------
    # Print learned q values

    print("Learned softmax q for all layers:")

    for i in range(model.max_iter):

        print(
            f"layer {i}: "
            f"{torch.softmax(model.q_logits[i], dim=0).detach()}"
        )


    # --------------------------------------------------------
    # Calculate TEST results

    if override_result or (
        not os.path.exists(result_log_path)
    ):
        calc_models_test(model)

    # --------------------------------------------------------
    # TRAINING / VALIDATION LOSS PLOT

    df_loss = pd.read_csv(loss_log_path)

    plt.figure(figsize=(6, 5))

    plt.plot(
        df_loss["epoch"],
        df_loss["avg_train_loss"],
        color="tab:blue",
        label="Train Loss",
        linewidth=1.5,
        marker="o"
    )

    plt.plot(
        df_loss["epoch"],
        df_loss["avg_val_loss"],
        color="tab:red",
        label="Validation Loss",
        linewidth=1.5,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.title("Train and Validation Loss per Epoch")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir,
            "train_validation_loss.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            plot_dir,
            "train_validation_loss.pdf"
        ),
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # --------------------------------------------------------
    # LOAD AND DISPLAY TEST RESULTS
    df_results = pd.read_parquet(
        result_log_path
    )

    # Important sanity check:
    # prints number of TEST samples for each
    # scenario and SNR.
    samples_count_table = pd.crosstab(
        df_results["scenario_type"],
        df_results["snr_db"]
    )

    print("\nNumber of TEST samples per scenario/SNR:")
    print(samples_count_table)


    # Separate result plots for every scenario
    df_results.groupby(
        "scenario_type",
        group_keys=False
    ).apply(calc_and_print_rmse)


if __name__ == "__main__":
    result_analysis()
