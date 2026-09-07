import torch
from model_config import scenario_type_dict

from model_config import DEVICE, scenario_type_dict


def p_rmse_tof_loss(p_hat, P_true_sig):
    return torch.sqrt(torch.mean((P_true_sig - p_hat[:, :P_true_sig.shape[1]])**2))

p_win_size = 5 #from each side->overall 11bins

def p_rmse_windowed_tof_loss(p_hat, P_true_sig, true_tau_idx):
    #creating triangles around the tof location, to make the "missing" less brutal, to make the loss more forgiving for small errors in the tof estimation
    P_true_sig_win = P_true_sig.clone()
    for b in range(P_true_sig.shape[0]): #P_true_sig is shape (B,K) and true_tau_idx is shape (B,C)
        for i in true_tau_idx[b]:
            i=int(i)
            peak=P_true_sig[b, i]
            left_start = max(i - p_win_size, 0)
            right_end = min(i + p_win_size, P_true_sig.shape[1] - 1)

            left_len = i - left_start
            right_len = right_end - i

            if left_len > 0:
                P_true_sig_win[b, left_start:i] = torch.linspace(
                    0,
                    peak,
                    steps=left_len + 1,
                    device=P_true_sig.device)[:-1]

            P_true_sig_win[b, i] = peak

            if right_len > 0:
                P_true_sig_win[b, i + 1:right_end + 1] = torch.linspace(
                    peak,
                    0,
                    steps=right_len + 1,
                    device=P_true_sig.device)[1:]
            '''
            linespace_left = torch.linspace(0, P_true_sig[b, i], steps=p_win_size)
            linespace_right = torch.linspace(P_true_sig[b, i], 0, steps=p_win_size)
            p_win_start = max(i - p_win_size, 0)
            win_start = max(0, p_win_size - (i - p_win_start))
            P_true_sig_win[b, p_win_start:i] = linespace_left[win_start:]

            p_win_end = min(i + p_win_size, P_true_sig.shape[1])
            win_end = min(p_win_size, (p_win_end - i))
            P_true_sig_win[b, i:p_win_end] = linespace_right[:win_end]
            '''
    return torch.sqrt(torch.mean((P_true_sig_win - p_hat[:, :P_true_sig.shape[1]])**2))


def local_softargmax_tof_loss(
        p_hat, true_tau_idx, tau_grid_ns, scenario, beta=30.0
):
    '''     Differentiable ToF-domain loss using local softargmax windows.
    The model outputs p_hat with shape (B, K+M).
    The first K entries correspond to the ToF/delay grid.
    The last M entries correspond to the noise part from A_aug = [A I_M].

    This function:
    1. takes the signal part p_hat_sig = p_hat[:, :K],
    2. converts it into log-power scores,
    3. creates a local delay-bin window around each true ToF index,
    4. applies softmax only inside each local window,
    5. computes one differentiable ToF estimate per path,
    6. compares tau_hat to true_tau using an RMSE-style loss.

    Shapes:
    p_hat.shape         = (B, K+M)
    true_tau_idx.shape  = (B, C)
    tau_grid_ns.shape   = (B, K)

    Final outputs:
    tau_hat.shape       = (B, C)
    true_tau.shape      = (B, C)
    loss                = scalar
    '''
    C = true_tau_idx.shape[1]
    epsilon = 1e-8
    B = p_hat.shape[0]  # batch size
    K = tau_grid_ns.shape[1]  # number of grid bins

    # shape (B,K) because tau_grid_ns is length K, we only care about the signal part of p_hat for the loss calculation
    p_hat_sig = p_hat[:, :K]
    # The clamp in the score is to prevent log(0)
    score = torch.log(torch.clamp(p_hat_sig, min=epsilon))  # shape (B,K) because tau_grid_ns is length K
    # extracting true tau: (representation: sample 0: tau_grid_ns[0, [tau1, tau2, tau3]]). tau_grid.shape=(B,K) and true_tau_idx.shape=(B,C)
    true_tau_idx = true_tau_idx.long()
    true_tau = torch.gather(tau_grid_ns, dim=1, index=true_tau_idx)  # shape (B,C) , in ns units

    # the local window:
    # window length in bins, should be odd to be symmetric around the true index
    tau_hat = torch.zeros((B, C), dtype=tau_grid_ns.dtype, device=DEVICE)

    for b in range(B):
        curr_win_size = scenario_type_dict[scenario[b]]['loss_win_size']
        L = 2 * int(curr_win_size) + 1
        local_idx = torch.zeros((C, L), dtype=torch.long, device=DEVICE)  # to store the local window indices
        for c in range(C):
            k_c = true_tau_idx[b, c]
            k_start = max(int(k_c) - int(curr_win_size), 0)  # start of the window, ensuring it doesnt go out of bounds
            k_end = min(int(k_c) + int(curr_win_size), K - 1)  # end of the window, ensuring it doesnt go out of bounds
            window_indices = torch.arange(k_start, k_end + 1, device=DEVICE)  # indices of the local window around the true index
            if len(window_indices) < L:
                pad_len = L - len(window_indices)
                if k_start == 0:
                    padding = torch.zeros(pad_len, dtype=torch.long, device=DEVICE)
                    window_indices = torch.cat((padding, window_indices))  # adds padding to the beginning of the window
                else:
                    padding = torch.full((pad_len,), K - 1, dtype=torch.long, device=DEVICE)  # full-a tensor with a specific shape, filled with the same value
                    window_indices = torch.cat((window_indices, padding))  # adds padding to the end of the window

            local_idx[c, :] = window_indices
        local_score = torch.zeros((C, L), dtype=score.dtype, device=DEVICE)
        for c in range(C):
            local_score[c, :] = score[b, local_idx[c, :]]  # shape (C,L) the scores of the local windows around each true index
        # local softmax weights calc:
        alpha_local = torch.softmax(beta * local_score, dim=1)  # shape (C,L) after the softmax, the weights for the local window around each true index

        # the tau grid values inside each local window:
        local_tau = torch.zeros((C, L), dtype=tau_grid_ns.dtype, device=DEVICE)

        for c in range(C):
            local_tau[c, :] = tau_grid_ns[b, local_idx[c, :]]  # shape (C,L) the tau grid values of the local windows around each true index

        # differentiable tau estimate for each path\sample:
        tau_hat[b, :] = torch.sum(alpha_local * local_tau, dim=1)  # shape (B,C)
    # calculating ToF-domain RMSE-style loss:
    err2 = (tau_hat - true_tau) ** 2  # shape (B,C)
    loss_per_sample = torch.sqrt(torch.mean(err2, dim=1) + epsilon)  # shape (B,)
    loss = torch.mean(loss_per_sample)  # scalar

    return loss, tau_hat, true_tau
