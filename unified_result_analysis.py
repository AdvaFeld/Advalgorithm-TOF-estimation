import os
import pickle
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset, ConcatDataset
from tqdm import tqdm

from tof_dataset import TOFTrainingDataset
from model import DeepUnfoldedSPICE, UnifiedModel, DeepUnfoldedSPICE_piecewise, Classifier_MLP
from Adva_algo_implementation import run_likes
from Adva_algo_implementation import run_slim
from Adva_algo_implementation import run_spice
from Adva_algo_implementation import run_likes30m
from Adva_algo_implementation import run_slim5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_config import model_max_iterations, DEVICE, RESULT_LOG_NAME, CHECKPOINT_UNIFIED_DIR
from tof_eval_utils import p_hat2tau_hat


override_result = True
# Name for saving the Phase-2 test results
result_run_name = "phase2_joint_soft_routing_sigmoid"

run_path = CHECKPOINT_UNIFIED_DIR + "/" + result_run_name
epoch = 29
checkpoint_path = f"{run_path}/epoch_{epoch}.pt"
result_log_path = run_path + "/" + RESULT_LOG_NAME
Path(run_path).mkdir(parents=True, exist_ok=True)


test_batch_size = 16 * 16

max_iter=200
tol=1e-3

# our existing models
models_names = ["unfolding", "spice", "likes", "slim", "likes30m", "slim5"]


def calc_models_test(model):
    """
    loads saved data and model, calculates models on test data and saves result to log file
    """

    # Load held-out test samples not used in training or validation:
    # 7K baseline + 21K non-baseline
    dataset_nonbaseline = TOFTrainingDataset(
        data_dir="dataset_rest_of_scenarios",
        data_size=66 * 1000)

    dataset_baseline = TOFTrainingDataset(
        data_dir="dataset_baseline",
        data_size=52 * 1000)

    # Samples 0:39999 were used for training.
    # Samples 40000:44999 were used for validation.
    # Baseline test:     45000:51999
    # Non-baseline test: 45000:65999
    baseline_test_ids = list(range(45 * 1000, 52 * 1000))
    non_baseline_test_ids = list(range(45 * 1000, 66 * 1000))

    test_nonbaseline = Subset(
        dataset_nonbaseline,
        non_baseline_test_ids)

    test_baseline = Subset(
        dataset_baseline,
        baseline_test_ids)

    test_data = ConcatDataset([
        test_nonbaseline,
        test_baseline])

    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size,
        shuffle=False)

    print("Unified test set:")
    print("  baseline:", len(test_baseline))
    print("  non-baseline:", len(test_nonbaseline))
    print("  total:", len(test_data))

    if os.path.exists(result_log_path):
        os.remove(result_log_path)

    should_append = False

    with torch.no_grad():
        progress_bar = tqdm(test_loader)

        for batch in progress_bar:
            models_result = dict()

            # DU model - remains batched exactly as before
            models_result["unfolding_p"] = model(batch["y_noisy"], batch["A_aug"], batch["p0"]).detach().cpu().numpy()

            # Benchmarks - run one sample at a time
            benchmark_funcs = {
                "spice": lambda y, A, p: run_spice(y, A, p, max_iter=max_iter, tol=tol),
                "likes": lambda y, A, p: run_likes(y, A, p, max_iter=max_iter, tol=tol),
                "slim": lambda y, A, p: run_slim(y, A, p, max_iter=max_iter, tol=tol),
                "likes30m": lambda y, A, p: run_likes30m(y, A, p, max_iter=max_iter, tol=tol),
                "slim5": lambda y, A, p: run_slim5(y, A, p, tol=tol)}

            for name, func in benchmark_funcs.items():
                p_list, iter_list, conv_list, runtime_list = [], [], [], []

                for i in range(batch["y_noisy"].shape[0]):
                    p_out, n_iter, converged, runtime = func(batch["y_noisy"][i:i+1], batch["A_aug"][i:i+1], batch["p0"][i:i+1])
                    p_list.append(p_out.detach().cpu().numpy())
                    iter_list.append(n_iter)
                    conv_list.append(converged)
                    runtime_list.append(runtime)

                    #progress_bar.set_postfix(benchmark=name, iteration=n_iter, converged=converged)

                models_result[f"{name}_p"] = np.concatenate(p_list, axis=0)
                models_result[f"{name}_iterations"] = iter_list
                models_result[f"{name}_converged"] = conv_list
                models_result[f"{name}_runtime_s"] = runtime_list

            batch = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in batch.items()}

            for model_name in models_names:
                models_result[f"{model_name}_mse_tof"] = []
                models_result[f"{model_name}_se_first_path"] = []

                for i in range(batch["true_tau_ns"].shape[0]):
                    true_tau_ns = np.sort(batch["true_tau_ns"][i])
                    num_paths = true_tau_ns.shape[0]

                    tau, _ = p_hat2tau_hat(
                        models_result[f"{model_name}_p"][i],
                        batch["tau_grid"][i],
                        batch["K"][i],
                        num_paths)

                    tau_ns = np.sort(tau * 1e9)

                    models_result[f"{model_name}_mse_tof"].append(
                        np.mean((tau_ns - true_tau_ns) ** 2))

                    models_result[f"{model_name}_se_first_path"].append(
                        (tau_ns[0] - true_tau_ns[0]) ** 2)

                del models_result[f"{model_name}_p"]

            models_result["snr_db"] = batch["snr_db"]
            models_result["scenario_type"] = batch["scenario_type"]
            models_result["sample_idx"] = batch["sample_idx"]

            models_result = {k: list(v) if isinstance(v, np.ndarray) else v for k, v in models_result.items()}

            chunk_df = pd.DataFrame(models_result)
            chunk_df.to_parquet(result_log_path, engine='fastparquet', append=should_append)
            should_append = True



def calc_and_print_rmse(df):
    """
    Calculates RMSE versus SNR for one scenario and saves the plots.
    """

    scenario_name = str(df.name).replace(" ", "_")

    # ========================================================
    # RMSE OF ALL TOF PATHS
    # ========================================================

    plt.figure(figsize=(6, 5))

    for model_name in models_names:

        rmse_tof_per_snr = (
        np.sqrt(
            df.groupby("snr_db")[f"{model_name}_mse_tof"].mean()
        )
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
    plt.ylabel("RMSE ToF [ns]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.title(str(df.name))
    plt.tight_layout()

    tof_plot_path = os.path.join(
        run_path,
        "{}_rmse_tof.png".format(scenario_name)
    )

    plt.savefig(
        tof_plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", tof_plot_path)

    # ========================================================
    # FIRST-PATH RMSE
    # ========================================================

    plt.figure(figsize=(6, 5))

    for model_name in models_names:

        rmse_first_path_per_snr = (
        np.sqrt(
            df.groupby("snr_db")[f"{model_name}_se_first_path"].mean()
        )
        .to_frame("rmse_first_path")
        .reset_index())

        plt.plot(
            rmse_first_path_per_snr["snr_db"],
            rmse_first_path_per_snr["rmse_first_path"],
            label=model_name,
            marker="o"
        )

    plt.xlabel("SNR [dB]")
    plt.ylabel("First-path RMSE [ns]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.title(str(df.name))
    plt.tight_layout()

    first_path_plot_path = os.path.join(
        run_path,
        "{}_rmse_first_path.png".format(scenario_name)
    )

    plt.savefig(
        first_path_plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", first_path_plot_path)


def result_analysis():
    baseline_model = DeepUnfoldedSPICE(model_max_iterations)
    super_res_model = DeepUnfoldedSPICE_piecewise(model_max_iterations)
    classifier_model = Classifier_MLP()

    model = UnifiedModel(baseline_model, super_res_model, classifier_model).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    # calcs the result logs if not calculated before
    if override_result or (not os.path.exists(result_log_path)):
        calc_models_test(model)

    # display saved results
    df = pd.read_parquet(result_log_path)
    # print number of samples per snr and scenario_type
    samples_count_table = pd.crosstab(df['scenario_type'], df['snr_db'])
    print(samples_count_table)
    df.groupby('scenario_type', group_keys=False).apply(calc_and_print_rmse)


if __name__ == "__main__":
    result_analysis()

