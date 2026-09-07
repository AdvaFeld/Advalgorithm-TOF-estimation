import os
import pickle

import numpy as np

from ToA_DataGenerator import generate_toa_data
from model_config import global_seed, scenario_type_dict, DATASET_DIR, snr_db_choices, TEST_SET_DIR


def generate_dataset(num_samples, dataset_name=DATASET_DIR, M=64, bw=40e6,
                         f0=0.0, tau_max=200e-9, delay_step=0.5e-9, num_paths=3,
                         snr_db_choices=snr_db_choices,
                         train_seed=global_seed):
    """ Generates training data for the deep unfolding model.
     The function keeps the same simulation's structure because of the use of the generate_toa_data function.
      the simulation's generate_toa_data function remains the raw sample generator (mainly)
    returns a list of dictionaries, where each dictionary. each entry contains one training sample and its labels\metadata.
    For each training sample, it should:
        choose a scenario type
        choose exact delays for that scenario
        choose SNR
        choose path powers
        call generate_toa_data(...)
        build A_aug
        build p0
        build P_true_sig
        pack everything into a sample dictionary
    """
    rng = np.random.default_rng(train_seed)  # for reproducibility
    # the line from above means that within one run, every sample is different, but across repeated runs, the entire sequence is reproduced.

    delay_step_ns = delay_step * 1e9  # convert to ns (grid spacing)
    tau_max_ns = tau_max * 1e9  # convert to ns
    K = int(round(tau_max / delay_step) + 1)  # number of signal-grid bins
    for sample_idx in range(num_samples):
        scenario_type = rng.choice(list(scenario_type_dict.keys()))
        # choosing the exact delays for the paths based on the scenario type

        scenario_conf = scenario_type_dict[scenario_type]
        min_gap, max_gap = scenario_conf["min_gap"], scenario_conf["max_gap"]  # in ns

        valid_sample = False
        while not valid_sample:
            # generate random delays for the paths, all in ns units
            gap1 = rng.uniform(min_gap, max_gap)  # gap between path 1 and 2
            gap2 = rng.uniform(min_gap, max_gap)  # gap between path 2 and 3
            max_start = tau_max_ns - (
                    gap1 + gap2)  # max starting delay for path 1 to ensure all paths fit within tau_max
            if max_start <= 0:
                continue  # invalid scenario, regenerate (skipping the while iter and go back and redraw gap1,2)
            tau1_ns = rng.uniform(0, max_start)  # delay of path 1
            # this means even within one scenario family, the whole 3-path group
            # can appear in many different regions of the delay axis (to avoid bias)
            tau2_ns = tau1_ns + gap1  # delay of path 2
            tau3_ns = tau2_ns + gap2  # delay of path 3
            # tau1<tau2<tau3 is guaranteed by construction
            tau_ns = np.array([tau1_ns, tau2_ns, tau3_ns], dtype=float)

            # converting to the index:
            # round for making it on grid (rounding to indx), and then clip to ensure it's within the valid range of indices (0 to K-1)
            true_tau_idx = np.round(tau_ns / delay_step_ns).astype(int)  # true delay indices for the paths
            true_tau_idx = np.clip(true_tau_idx, 0, K - 1)  # clip to valid range (that it fits the interval [0, K-1])

            if len(np.unique(true_tau_idx)) == num_paths:
                # Only accept the sample if the three paths occupy three distinct grid bins.
                valid_sample = True  # valid means all 3 paths fit under tau_max and after converting to grid indices, they occupy 3 distinct bins
                # (it keeps getting random samples until they correspond to the gap demand)
            if (sample_idx % 1000 == 0):
                print(f"Generating sample {sample_idx}/{num_samples} for scenario '{scenario_type}' with delays {tau_ns} ns and indices {true_tau_idx}.")
        # choosing SNR:
        snr_db = float(snr_db_choices[sample_idx % len(snr_db_choices)])
        # Choosing the path powers, mostly decresing (physically reasonable), but not always:
        raw_powers = rng.uniform(0.5, 3.0, size=num_paths)
        if rng.random() < 0.75:
            P_true = np.sort(raw_powers)[::-1]  # sort in decreasing order with 75% probability (earlier paths stronger than later paths)
            if rng.random() < 0.20:
                swap_idx = rng.integers(0, 2)
                P_true[swap_idx], P_true[swap_idx + 1] = P_true[swap_idx + 1], P_true[swap_idx]

        else:  # 25% the data doesnt have to obey physics, to be manage to be able to avoid biasing the model
            P_true = raw_powers.copy()
            rng.shuffle(P_true)

        # generate the raw sample using the provided function
        y_noisy, A, tau_grid, true_tau, true_gamma, P_true_returned, SNR_dB_used, noise_power, signal_power = generate_toa_data(
            M=M,
            bw=bw,
            f0=f0,
            tau_max=tau_max,
            true_tau_idx=true_tau_idx,
            P_true=P_true,
            SNR_dB=snr_db,
            delay_step=delay_step,
            truth_mode="on_grid",
            random_seed=train_seed + sample_idx,
        )

        # A_aug:
        I_M = np.eye(A.shape[0], dtype=complex)
        A_aug = np.hstack((A, I_M))  # shape (M, K+M)
        # p0:
        K = A.shape[1]  # number of columns in A (signal grid size)
        M = A.shape[0]  # number of measurements
        p0 = np.zeros(K + M, dtype=float)  # initial guess for augmented power vector
        for k in range(K + M):  # total number of columns in A_aug
            a_k = A_aug[:, k:k + 1]  # shape (M, 1)
            numerator = np.abs(a_k.conj().T @ y_noisy) ** 2  # shape (1, 1)
            denominator = np.linalg.norm(a_k) ** 4
            p0[k] = numerator.item() / denominator  # initial power estimate for grid bin k

        P_true_sig = np.zeros(K, dtype=float)
        output_true_tau_idx = np.round(true_tau / delay_step).astype(
            int)  # true delay indices for the paths (should match true_tau_idx)

        for path_idx in range(len(P_true_returned)):  # for the label vec
            grid_idx = output_true_tau_idx[path_idx]
            P_true_sig[grid_idx] = P_true_returned[path_idx]
        # wrapping it up nicely:
        sample = {
            "sample_idx": sample_idx,
            "scenario_type": scenario_type,
            "snr_db": snr_db,
            "y_noisy": y_noisy,
            "A": A,
            "A_aug": A_aug,
            "tau_grid": tau_grid,
            "true_tau": true_tau,
            "true_tau_idx": output_true_tau_idx,
            "true_gamma": true_gamma,
            "P_true": P_true_returned,
            "P_true_sig": P_true_sig,
            "p0": p0,
            "K": K,
            "M": M,
            "noise_power": noise_power,
            "signal_power": signal_power,
        }
        os.makedirs(dataset_name, exist_ok=True)
        full_file_path = f"{dataset_name}/{sample_idx}.pkl"
        with open(full_file_path, 'wb') as file:
            pickle.dump(sample, file)

if __name__ == "__main__":
    generate_dataset(66*1000)
