import numpy as np
import matplotlib.pyplot as plt
from ToA_DataGenerator import generate_toa_data
from Adva_algo_implementation import run_likes
from Adva_algo_implementation import run_slim 
from Adva_algo_implementation import run_spice
from Adva_algo_implementation import run_likes30m
from scipy.signal import find_peaks
from Adva_algo_implementation import run_slim5
'''
This function contains the main loop of the algorithm comparison, it will run a nested loop for the examination,
where the outer loop will iterate over the SNR values, and the inner loop will iterate over the trials, going through all the 3 algorithms of the specific snr.
SNR_dB_range is a list of SNR values in dB that we want to test, and algorithms is a list of algorithm instances that we want to compare.

SIMULATION_MODE = "debug_single":
    num trials=1
    debug prints enabled
    same behavior as the backup code, but through a different structure
SIMULATION_MODE = "mc_average":
    num trials=50
    debug prints disabled
    we will take the average of the RMSE values across the 50 trials for each SNR value, and plot the average RMSE vs SNR for each algorithm.
'''

SNR_dB_range = [0, 5, 10, 15, 20, 25, 30] # Example SNR values in dB
SIMULATION_MODE = "debug_single"   # "debug_single" or "mc_average"
NUM_TRIALS = 1 if SIMULATION_MODE == "debug_single" else 50
DEBUG_PRINTS = (SIMULATION_MODE == "debug_single")#for the comparison afterwards:

#bw modification try out:
BW=40e6 #40MHz
# in debug mode, run only the hard scenarios
DEBUG_SCENARIOS = {"moderate", "close", "very_close"}

# Tau scenarios (true tau values):
tau_scenarios={
    "baseline":[20.0, 80.0, 149.5] , #in nanoseconds, corresponds to 6m and 24m
    "moderate": [20.0, 35.0, 50.0],
    "close":[20.0, 24.0, 28.0] ,
    "very_close":[20.0, 21.0, 22.0]
}



# helper function: converts tau values from ns to grid indices
# because delay_step = 0.5 ns, for example:
# 20 ns -> 20/0.5 = 40
# 24 ns -> 24/0.5 = 48
def ns_to_idx(tau_ns_list, delay_step_ns=0.5):
    return [int(round(t_ns / delay_step_ns)) for t_ns in tau_ns_list]


tof_rmse_spice_vs_snr = []
tof_rmse_likes_vs_snr = []
tof_rmse_slim_vs_snr = []
tof_rmse_likes30m_vs_snr = []
tof_rmse_slim5_vs_snr = []

first_path_rmse_spice_vs_snr = []
first_path_rmse_likes_vs_snr = []
first_path_rmse_slim_vs_snr = []
first_path_rmse_likes30_vs_snr = []
first_path_rmse_slim5_vs_snr = []

runtime_spice_vs_snr = []
runtime_likes_vs_snr = []
runtime_slim_vs_snr = []
runtime_likes30m_vs_snr = []
runtime_slim5_vs_snr= []

#for later on use in finding tau from the algorithms' output p vectors:
def pick_top_peaks_scipy(p_sig, num_peaks, min_sep_bins=2, prominence=None):
    peaks, properties = find_peaks(p_sig, distance=min_sep_bins, prominence=prominence)
    #if we found atleast C peaks:
    if len(peaks) >= num_peaks:
        peak_heights = p_sig[peaks]
        best_order = np.argsort(peak_heights)[::-1]
        selected = peaks[best_order[:num_peaks]]
        return np.sort(selected)
    
    #if we found less than C peaks, then we will take the C strongest peaks that are at least min_sep_bins apart, even if they dont satisfy the prominence condition:
    selected = list(peaks)
    sorted_idx = np.argsort(p_sig)[::-1]

    for idx in sorted_idx:
        if idx not in selected:
            if all(abs(idx - s) > min_sep_bins for s in selected):
                selected.append(idx)
        if len(selected) == num_peaks:
            break

    return np.sort(np.array(selected))



for scenario_name in tau_scenarios:
    if SIMULATION_MODE == "debug_single" and scenario_name not in DEBUG_SCENARIOS:
        continue
    tau_ns_list = tau_scenarios[scenario_name]
    true_tau_idx = ns_to_idx(tau_ns_list, delay_step_ns=0.5)

    print(f"\nRunning scenario: {scenario_name}")
    print(f"tau values [ns]: {tau_ns_list}")
    print(f"true_tau_idx   : {true_tau_idx}")

    # these lists must be re-created for EACH scenario
    tof_rmse_spice_vs_snr = []
    tof_rmse_likes_vs_snr = []
    tof_rmse_slim_vs_snr = []
    tof_rmse_likes30m_vs_snr = []
    tof_rmse_slim5_vs_snr = []

    first_path_rmse_spice_vs_snr = []
    first_path_rmse_likes_vs_snr = []
    first_path_rmse_slim_vs_snr = []
    first_path_rmse_likes30_vs_snr = []
    first_path_rmse_slim5_vs_snr = []

    runtime_spice_vs_snr = []
    runtime_likes_vs_snr = []
    runtime_slim_vs_snr = []
    runtime_likes30m_vs_snr = []
    runtime_slim5_vs_snr = []

    for snr_db in SNR_dB_range:
        spice_rmse_trials = []
        likes_rmse_trials = []
        slim_rmse_trials = []
        likes30m_rmse_trials = []
        slim5_rmse_trials = []

        spice_fpe_trials = []
        likes_fpe_trials = []
        slim_fpe_trials = []
        likes30m_fpe_trials = []
        slim5_fpe_trials = []

        spice_runtime_trials = []
        likes_runtime_trials = []
        slim_runtime_trials = []
        likes30m_runtime_trials = []
        slim5_runtime_trials = []

        for trial_idx in range(NUM_TRIALS):
            y_noisy, A, tau_grid, true_tau, true_gamma, P_true, SNR_dB_used, noise_power, signal_power = generate_toa_data(
                M=64,
                bw=BW,
                f0=0.0,
                tau_max=200e-9,
                SNR_dB=snr_db,
                delay_step=0.5e-9,
                truth_mode="on_grid",
                true_tau_idx=true_tau_idx,   # NEW: pass the scenario tau indices
                random_seed=1 if SIMULATION_MODE == "debug_single" else trial_idx
            )

            if not DEBUG_PRINTS and (trial_idx % 5 == 0):
                print(f"Scenario={scenario_name} | SNR={snr_db} dB, trial {trial_idx+1}/{NUM_TRIALS}")
            if DEBUG_PRINTS:
                empirical_snr_db = 10 * np.log10(signal_power / noise_power)
                print(f"\nScenario={scenario_name}, trial={trial_idx+1}, target SNR={snr_db} dB")
                print(f"SNR_dB_used from generator = {SNR_dB_used}")
                print(f"signal_power = {signal_power:.6e}")
                print(f"noise_power  = {noise_power:.6e}")
                print(f"empirical SNR from powers = {empirical_snr_db:.3f} dB")
            I_M = np.eye(A.shape[0])
            A_aug = np.hstack([A, I_M])  # dim M x (K+M), y_noisy is M x 1, tau_grid is K x 1
            K = A.shape[1]  # number of columns in A, which is the num of delay grid steering columns
            M = A.shape[0]  # number of rows in A, which is the num of measurements\subcarriers
            tot_cols = A_aug.shape[1]  # total number of columns in A_aug, which is K+M

            # A[:,0]=steering vec for first delay in tau_grid, A[:,1]=steering vec for second delay in tau_grid, ..., A[:,K-1]=steering vec for last delay in tau_grid
            # A_aug=[A I_M], so first K columns are steering vecs, and last M columns are identity matrix columns (noise columns)
            p0 = np.zeros(tot_cols, dtype=float)  # initial guess for the optimization, which is a zero vector of length K+M

            # initializing p0 for all the algorithms in advance:
            for k in range(tot_cols):
                a_k = A_aug[:, k:k+1]  # k-th column, shape is (M,1)
                numerator = np.abs(a_k.conj().T @ y_noisy)**2  # squared magnitude of the inner product
                denominator = np.linalg.norm(a_k)**4  # squared norm of a_k
                p0[k] = numerator.item() / denominator  # initial power estimate as describe in stoica's algo

            if DEBUG_PRINTS:  # debug check:
                print(f"Scenario={scenario_name}, SNR (dB): {snr_db}, Initial 5 power estimates (p0): {p0[:5]}, A_shape: {A.shape}, A_aug_shape: {A_aug.shape}, p0_shape: {p0.shape}")

            # initial power estimate for each algorithm
            p_spice = p0.copy()
            p_likes = p0.copy()
            p_slim = p0.copy()
            p_likes30m = p0.copy()
            p_slim5 = p0.copy()

            # running the 5 algorithms for the current SNR value:
            max_iter = 200
            tol = 1e-3
            
            
            # weighted-spice:
            p_hat_spice, iter_spice, conv_spice, time_spice = run_spice(y_noisy, A_aug, p_spice, max_iter, tol)
            print(f"trial {trial_idx+1} | SPICE done | iter={iter_spice} | time={time_spice:.2f}s")

            # likes:
            p_hat_likes, iter_likes, conv_likes, time_likes = run_likes(y_noisy, A_aug, p_likes, max_iter, tol)
            print(f"trial {trial_idx+1} | LIKES done | iter={iter_likes} | time={time_likes:.2f}s")

            # slim:
            p_hat_slim, iter_slim, conv_slim, time_slim = run_slim(y_noisy, A_aug, p_slim, max_iter, tol)
            print(f"trial {trial_idx+1} | SLIM done | iter={iter_slim} | time={time_slim:.2f}s")

            # likes30m:
            p_hat_likes30m, iter_likes30m, conv_likes30m, time_likes30m = run_likes30m(y_noisy, A_aug, p_likes30m, max_iter, tol)
            print(f"trial {trial_idx+1} | LIKES30M done | iter={iter_likes30m} | time={time_likes30m:.2f}s")

            # slim5:
            p_hat_slim5, iter_slim5, conv_slim5, time_slim5 = run_slim5(y_noisy, A_aug, p_slim5, max_iter=5, tol=tol)
            print(f"trial {trial_idx+1} | SLIM5 done | iter={iter_slim5} | time={time_slim5:.2f}s")
            if DEBUG_PRINTS:
                print(f"SNR={snr_db} | SPICE: iter={iter_spice}, conv={conv_spice}, runtime={time_spice:.4f}s")
                print(f"SNR={snr_db} | LIKES: iter={iter_likes}, conv={conv_likes}, runtime={time_likes:.4f}s")
                print(f"SNR={snr_db} | SLIM : iter={iter_slim}, conv={conv_slim}, runtime={time_slim:.4f}s")
                print(f"SNR={snr_db} | Likes30: iter={iter_likes30m}, conv={conv_likes30m}, runtime={time_likes30m:.4f}s")
                print(f"SNR={snr_db} | SLIM5 : iter={iter_slim5}, conv={conv_slim5}, runtime={time_slim5:.4f}s")
                # NaN check:
                print("NaNs in SPICE:", np.isnan(p_hat_spice).any())
                print("NaNs in LIKES:", np.isnan(p_hat_likes).any())
                print("NaNs in SLIM :", np.isnan(p_hat_slim).any())
                print("NaNs in LIKES30M :", np.isnan(p_hat_likes30m).any())
                print("NaNs in SLIM5 :", np.isnan(p_hat_slim5).any())

            # peak extraction from the power estimates (p)
            p_sig_spice = p_hat_spice[:K]
            p_sig_likes = p_hat_likes[:K]
            p_sig_slim = p_hat_slim[:K]
            p_sig_likes30m = p_hat_likes30m[:K]
            p_sig_slim5 = p_hat_slim5[:K]

            #for debugging the rmse:
            p_noise_spice = p_hat_spice[K:]
            p_noise_likes = p_hat_likes[K:]
            p_noise_slim = p_hat_slim[K:]
            p_noise_likes30m = p_hat_likes30m[K:]
            p_noise_slim5 = p_hat_slim5[K:]



            if DEBUG_PRINTS:
                topL = 8

                def show_top_bins(name, p_sig):
                    top_idx = np.argsort(p_sig)[::-1][:topL]
                    print(f"\n{name} top-{topL} bins:")
                    for idx in top_idx:
                        print(f"  idx={idx:3d}, tau={tau_grid[idx]*1e9:7.2f} ns, p={p_sig[idx]:.3e}")

                print(f"\nDetailed spectrum debug | Scenario={scenario_name}, SNR={snr_db} dB")
                print("true_tau_idx =", np.sort(true_tau_idx))
                print("true_tau_ns  =", np.sort(true_tau * 1e9))

                show_top_bins("SPICE   ", p_sig_spice)
                show_top_bins("LIKES   ", p_sig_likes)
                show_top_bins("SLIM    ", p_sig_slim)
                show_top_bins("LIKES30M", p_sig_likes30m)
                show_top_bins("SLIM5   ", p_sig_slim5)

                print("\nSignal/noise sums:")
                print(f"SPICE    : signal_sum={np.sum(p_sig_spice):.3e}, noise_sum={np.sum(p_noise_spice):.3e}")
                print(f"LIKES    : signal_sum={np.sum(p_sig_likes):.3e}, noise_sum={np.sum(p_noise_likes):.3e}")
                print(f"SLIM     : signal_sum={np.sum(p_sig_slim):.3e}, noise_sum={np.sum(p_noise_slim):.3e}")
                print(f"LIKES30M : signal_sum={np.sum(p_sig_likes30m):.3e}, noise_sum={np.sum(p_noise_likes30m):.3e}")
                print(f"SLIM5    : signal_sum={np.sum(p_sig_slim5):.3e}, noise_sum={np.sum(p_noise_slim5):.3e}")
            # because we defined the true tof to be on the grid
            C = len(true_tau)  # number of true paths

            peak_idx_spice = pick_top_peaks_scipy(p_sig_spice, C, min_sep_bins=2)
            peak_idx_likes = pick_top_peaks_scipy(p_sig_likes, C, min_sep_bins=2)
            peak_idx_slim = pick_top_peaks_scipy(p_sig_slim, C, min_sep_bins=2)
            peak_idx_likes30m = pick_top_peaks_scipy(p_sig_likes30m, C, min_sep_bins=2)
            peak_idx_slim5 = pick_top_peaks_scipy(p_sig_slim5, C, min_sep_bins=2)

            # conversion of the indices into actual delay values: (* 1e9 to convert from seconds to nanoseconds)
            tau_hat_spice = tau_grid[peak_idx_spice] * 1e9
            tau_hat_likes = tau_grid[peak_idx_likes] * 1e9
            tau_hat_slim = tau_grid[peak_idx_slim] * 1e9
            tau_hat_likes30m = tau_grid[peak_idx_likes30m] * 1e9
            tau_hat_slim5 = tau_grid[peak_idx_slim5] * 1e9

            # converting true_tau to nanoseconds for the error calculation:
            true_tau_ns = true_tau * 1e9

            if DEBUG_PRINTS:
                print(f"Scenario={scenario_name}, SNR={snr_db}")
                print("true_tau_idx         =", np.sort(true_tau_idx))
                print("peak_idx_spice       =", np.sort(peak_idx_spice))
                print("peak_idx_likes       =", np.sort(peak_idx_likes))
                print("peak_idx_slim        =", np.sort(peak_idx_slim))
                print("peak_idx_likes30m    =", np.sort(peak_idx_likes30m))
                print("peak_idx_slim5       =", np.sort(peak_idx_slim5))

                print("true_tau_ns          =", np.sort(true_tau_ns))
                print("tau_hat_spice_ns     =", np.sort(tau_hat_spice))
                print("tau_hat_likes_ns     =", np.sort(tau_hat_likes))
                print("tau_hat_slim_ns      =", np.sort(tau_hat_slim))
                print("tau_hat_likes30m_ns  =", np.sort(tau_hat_likes30m))
                print("tau_hat_slim5_ns     =", np.sort(tau_hat_slim5))
                print("-" * 50)

            # RMSE over all paths
            mse_spice = np.mean((tau_hat_spice - true_tau_ns)**2)
            mse_likes = np.mean((tau_hat_likes - true_tau_ns)**2)
            mse_slim = np.mean((tau_hat_slim - true_tau_ns)**2)
            mse_likes30m = np.mean((tau_hat_likes30m - true_tau_ns)**2)
            mse_slim5 = np.mean((tau_hat_slim5 - true_tau_ns)**2)

            rmse_spice = np.sqrt(mse_spice)
            rmse_likes = np.sqrt(mse_likes)
            rmse_slim = np.sqrt(mse_slim)
            rmse_likes30m = np.sqrt(mse_likes30m)
            rmse_slim5 = np.sqrt(mse_slim5)

            # another comparison option- comparing only the first path estimation errors:
            fpe_spice = (tau_hat_spice[0] - true_tau_ns[0])**2
            fpe_likes = (tau_hat_likes[0] - true_tau_ns[0])**2
            fpe_slim = (tau_hat_slim[0] - true_tau_ns[0])**2
            fpe_slim5 = (tau_hat_slim5[0] - true_tau_ns[0])**2
            fpe_likes30m = (tau_hat_likes30m[0] - true_tau_ns[0])**2

            rmse_fpe_spice = np.sqrt(fpe_spice)
            rmse_fpe_slim5 = np.sqrt(fpe_slim5)
            rmse_fpe_likes = np.sqrt(fpe_likes)
            rmse_fpe_slim = np.sqrt(fpe_slim)
            rmse_fpe_likes30m = np.sqrt(fpe_likes30m)

            # adding the values to the lists for plotting later:
            spice_rmse_trials.append(rmse_spice)
            likes_rmse_trials.append(rmse_likes)
            slim_rmse_trials.append(rmse_slim)
            likes30m_rmse_trials.append(rmse_likes30m)
            slim5_rmse_trials.append(rmse_slim5)

            spice_fpe_trials.append(rmse_fpe_spice)
            likes_fpe_trials.append(rmse_fpe_likes)
            slim_fpe_trials.append(rmse_fpe_slim)
            likes30m_fpe_trials.append(rmse_fpe_likes30m)
            slim5_fpe_trials.append(rmse_fpe_slim5)

            spice_runtime_trials.append(time_spice)
            likes_runtime_trials.append(time_likes)
            slim_runtime_trials.append(time_slim)
            likes30m_runtime_trials.append(time_likes30m)
            slim5_runtime_trials.append(time_slim5)

        # average over trials for this SNR
        tof_rmse_spice_vs_snr.append(np.mean(spice_rmse_trials))
        tof_rmse_likes_vs_snr.append(np.mean(likes_rmse_trials))
        tof_rmse_slim_vs_snr.append(np.mean(slim_rmse_trials))
        tof_rmse_likes30m_vs_snr.append(np.mean(likes30m_rmse_trials))
        tof_rmse_slim5_vs_snr.append(np.mean(slim5_rmse_trials))

        first_path_rmse_spice_vs_snr.append(np.mean(spice_fpe_trials))
        first_path_rmse_likes_vs_snr.append(np.mean(likes_fpe_trials))
        first_path_rmse_slim_vs_snr.append(np.mean(slim_fpe_trials))
        first_path_rmse_likes30_vs_snr.append(np.mean(likes30m_fpe_trials))
        first_path_rmse_slim5_vs_snr.append(np.mean(slim5_fpe_trials))

        runtime_spice_vs_snr.append(np.mean(spice_runtime_trials))
        runtime_likes_vs_snr.append(np.mean(likes_runtime_trials))
        runtime_slim_vs_snr.append(np.mean(slim_runtime_trials))
        runtime_likes30m_vs_snr.append(np.mean(likes30m_runtime_trials))
        runtime_slim5_vs_snr.append(np.mean(slim5_runtime_trials))

    # plotting the results for THIS scenario

    # plot 1 RMSE vs SNR for the five algorithms:
    plt.figure(figsize=(8, 5))
    plt.plot(SNR_dB_range, tof_rmse_spice_vs_snr, marker='o', label='SPICE')
    plt.plot(SNR_dB_range, tof_rmse_likes_vs_snr, marker='s', label='LIKES')
    plt.plot(SNR_dB_range, tof_rmse_slim_vs_snr, marker='^', label='SLIM')
    plt.plot(SNR_dB_range, tof_rmse_likes30m_vs_snr, marker='d', label='LIKES30M')
    plt.plot(SNR_dB_range, tof_rmse_slim5_vs_snr, marker='x', label='SLIM5')
    plt.xlabel('SNR [dB]')
    plt.ylabel('ToF RMSE [ns]')
    plt.title(f'ToF RMSE vs SNR - {scenario_name}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot 2 Runtime vs SNR for the five algorithms:
    plt.figure(figsize=(8, 5))
    plt.plot(SNR_dB_range, runtime_spice_vs_snr, marker='o', label='SPICE')
    plt.plot(SNR_dB_range, runtime_likes_vs_snr, marker='s', label='LIKES')
    plt.plot(SNR_dB_range, runtime_slim_vs_snr, marker='^', label='SLIM')
    plt.plot(SNR_dB_range, runtime_likes30m_vs_snr, marker='d', label='LIKES30M')
    plt.plot(SNR_dB_range, runtime_slim5_vs_snr, marker='x', label='SLIM5')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Runtime [s]')
    plt.title(f'Runtime vs SNR - {scenario_name}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot 3 first-path RMSE vs SNR:
    plt.figure(figsize=(8, 5))
    plt.plot(SNR_dB_range, first_path_rmse_spice_vs_snr, marker='o', label='SPICE')
    plt.plot(SNR_dB_range, first_path_rmse_likes_vs_snr, marker='s', label='LIKES')
    plt.plot(SNR_dB_range, first_path_rmse_slim_vs_snr, marker='^', label='SLIM')
    plt.plot(SNR_dB_range, first_path_rmse_likes30_vs_snr, marker='d', label='LIKES30M')
    plt.plot(SNR_dB_range, first_path_rmse_slim5_vs_snr, marker='x', label='SLIM5')
    plt.xlabel('SNR [dB]')
    plt.ylabel('First-path RMSE [ns]')
    plt.title(f'First-path ToF RMSE vs SNR - {scenario_name}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()