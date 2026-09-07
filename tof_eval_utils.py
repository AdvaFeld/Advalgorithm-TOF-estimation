from scipy.signal import find_peaks
import numpy as np



def pick_top_peaks_scipy(p_sig, num_peaks, min_sep_bins=2, prominence=None):
    '''Finds the best num_peaks peak indices in the signal spectrum p_sig, with a minimum
    separation and a fallback if too few peaks are detected. This is the same peak-picking
    logic already used in the simulation script
    '''
    
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
            if all(abs(idx - s) >= min_sep_bins for s in selected):
                selected.append(idx)
        if len(selected) == num_peaks:
            break

    return np.sort(np.array(selected))


def p_hat2tau_hat(p_hat, tau_grid, K, num_paths, min_sep_bins=2, prominence=None):
    """
    Takes one algorithm output p_hat, keeps only the first K entries (p_hat[:K]),
    finds the peak indices using pick_top_peaks_scipy, and converts those indices to
    estimated delay values using tau_grid.
    p_hat
    """

    p_sig = np.asarray(p_hat[:K], dtype=float)

    peak_idx = pick_top_peaks_scipy(
        p_sig=p_sig,
        num_peaks=num_paths,
        min_sep_bins=min_sep_bins,
        prominence=prominence
    )

    tau_hat = np.sort(tau_grid[peak_idx])
    return tau_hat, peak_idx


