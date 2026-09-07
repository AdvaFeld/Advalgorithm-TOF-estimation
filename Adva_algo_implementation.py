import time
import torch
from model_config import DEVICE

"""
This is the script for the algorithm implementations and simulations for the ToA estimation problem. It generates the data, initializes the optimization variables, and runs the optimization for each algorithm in the list of algorithms. The results are being sent to the main function (thesis-Adva Simulation code.py) for evaluation and plotting.
The algorithms are ToF optimizers:
1. Weighted-SPICE
2. LIKES
3. SLIM
each one of them:
inputs:
y_noisy, A_aug, p_0, settings?
iter_num
outputs:
p_hat, iter_num, convergence_metric
maybe also runtime

*the plotting and peak finder will be executed in the main function (thesis-Adva Simulation code.py)
each one of the algorithms is implemented in the following way, only with different weights\penalty, and therefore a different p will be returned:
#1- R=A_aug*diag(p_curr)*A_aug^H
#2 inverting R
#3 computing the new p=p_next
#4 checking convergence according to the formulas: ||(p_next-p_curr)||_2/(||p_curr||_2)<tol
#5 return final p_hat, iterations, convergence flag, runtime
"""
 

# Weighted SPICE: for now its spice A update
## SPICEa update from the weighted-SPICE paper, with SPICE weights w_k = ||a_k||_2^2
def run_spice(y_noisy, A_aug, p0, max_iter=1000, tol=1e-3):
    p_curr = p0.clone()
    converged = False  # flag to stop iterations if convergence is reached
    w = torch.sum(torch.abs(A_aug) ** 2, dim=1)  # w_k=||a_k||_2^2

    A_aug_H = A_aug.conj().transpose(1, 2)

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for iter_num in range(max_iter):
        # covariance matrix of the received signal, which is used in the SPICE algorithm to compute the new power estimates
        # OPTIMIZATION:
        # Instead of A @ diag(p) @ A.H, we use element-wise broadcasting:
        # R = (A * p) @ A.H
        # This is mathematically identical but avoids O(N^2) memory for a diagonal matrix.
        R = (A_aug * p_curr.unsqueeze(1)) @ A_aug_H
        z = torch.linalg.solve(R, y_noisy)  # z=R^-1y
        r = A_aug_H @ z  # r[k]=a_k^*z from SPICE article (the first one)
        p_next = p_curr * torch.abs(r.squeeze(-1)) / torch.sqrt(w)  # spice A update

    #convergence check:
        relative_change=torch.linalg.norm(p_next-p_curr, dim=1)/torch.clamp(torch.linalg.norm(p_curr, dim=1), min=1e-12)
        #updating p before stopping:
        p_curr=p_next
        if torch.all(relative_change < tol).item():
            converged=True
            break

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    

    runtime = time.perf_counter() - t0
    return p_curr, iter_num + 1, converged, runtime


def run_likes(y_noisy, A_aug, p0, max_iter=1000, tol=1e-3, update_iter=1):
    """
    in spice A update, the weights are w_k=||a_k||_2^2, in likes, the weights looks different (thats the main difference)
    w_k(likes)=a_k^*R^-1a_k
    and the power update (Likes A) is:
    p_k^i+1=p_k^i*|a_k^*R^-1y|/sqrt(w_k(likes))
    => the same update as spice, only the denominator changes from the fixed sqrt(w_k)
    to the dynamic sqrt(w_k(likes)) which depends on the adaptive covariance matrix R^-1
    **Important note: in the weighted spice article, they initialize likes with spice's output,
    but here to do a fair comparison, we will initialize likes with the same initial power estimates as spice (p0)
    ** in their simulation they updated the weightes only once in 30 iterations.
    thats why they have 2 indices of R in the update formula.
    """
    p_curr = p0.clone()
    #n_cols = A_aug.shape[1]  # =(B,M,K+M) so n_cols=K+M
    converged = False  # flag to stop iterations if convergence is reached
    A_aug_H = A_aug.conj().transpose(1, 2)
    A_aug_conj = torch.conj(A_aug)

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # likes weights:
    w = None
    for iter_num in range(max_iter):
        # covariance matrix of the received signal
        # Optimized R calculation: R = (A * p) @ A.H
        R = (A_aug * p_curr.unsqueeze(1)) @ A_aug_H
        z = torch.linalg.solve(R, y_noisy)  # z=R^-1y
        r = A_aug_H @ z  # r[k]=a_k^*z from spice article (the first one)
        if iter_num % update_iter == 0:  # update the weights only once every 30 iterations
            # VECORIZED OPTIMIZATION:
            # Instead of looping through k columns, we solve RU = A_aug.
            # U becomes a matrix where each column k is R^-1 @ a_k.
            U = torch.linalg.solve(R, A_aug)

            # The weights w_k are the diagonal elements of (A_aug.H @ U).
            # We can compute this efficiently using element-wise multiplication and sum:
            # w_k = sum_over_rows(conj(A_aug_ik) * U_ik)
            w = torch.real(torch.sum(A_aug_conj * U, dim=1))

        # Stability: Prevent division by tiny numbers
        w = torch.clamp(w, min=1e-12)
        # Likes A update:
        p_next = p_curr * torch.abs(r.squeeze(-1)) / torch.sqrt(w)

        #convergence check:
        relative_change=((torch.linalg.norm(p_next-p_curr, dim=1))/torch.clamp(torch.linalg.norm(p_curr,dim=1),min=1e-12))
        p_curr = p_next
        if torch.all(relative_change<tol).item():
            converged=True
            break

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    runtime = time.perf_counter() - t0
    return p_curr, iter_num + 1, converged, runtime


def run_likes30m(y_noisy, A_aug, p0, max_iter=1000, tol=1e-3):
    """
    LIKES with paused weight updates:
    - The LIKES weights w_k = a_k^* R^{-1} a_k
    are NOT updated every iteration, instead they are updated only once every 30 iterations, and kept fixed in between.
    """
    return run_likes(y_noisy, A_aug, p0, max_iter=max_iter, tol=tol, update_iter=30)


def run_slim(y_noisy, A_aug, p0, max_iter=1000, tol=1e-3):
    """
    now the weights are:
    w_k(slim A)=1/p_k
    the SLIM A update is: (according to weighted-SPICE paper):
    p_k^(i+1)=(P_k^i)^(3/2)|a_k^*R_i^-1y|
    Important note:
    In the weighted-SPICE article, SLIM was run for only 5 iterations
    in the numerical evaluations.
    Here, for fair comparison, we use the same input interface and stopping
    rule as the other algorithms.
    """
    # Clone p0 to ensure we don't modify the input tensor outside the function
    p_curr = p0.clone()
    converged = False
    A_aug_H = A_aug.conj().transpose(1, 2)

    # Synchronize at the start for accurate timing
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for iter_num in range(max_iter):
        # 1. Optimized R calculation: (A * p) @ A.H
        # This replaces A @ diag(p) @ A.H for efficiency
        R = (A_aug * p_curr.unsqueeze(1)) @ A_aug_H
        M = R.shape[1]  # R is shaper (B, M, M) so R.shape[1] is M=64
        B = y_noisy.shape[0]
        I_M = torch.eye(M, dtype=R.dtype, device=R.device).unsqueeze(0).expand(B, -1, -1)
        eps = 1e-8
        R =R+eps*I_M

        # 2. Solve Rz = y
        z = torch.linalg.solve(R, y_noisy)

        # 3. Compute r = A^H @ z
        r = A_aug_H @ z

        # 4. SLIM power update: p_next = p_curr^(3/2) * |r|
        # .view(-1) ensures the shape matches p_curr (equivalent to ravel)
        # p_next = torch.pow(torch.clamp(p_curr, min=eps), 1.5) * torch.abs(r.squeeze(-1))
        w_slim = 1.0 / torch.clamp(p_curr, min=eps)
        r_abs = torch.abs(r.squeeze(-1))
        p_next = p_curr * r_abs / torch.sqrt(w_slim)
        p_next = torch.clamp(p_next, min=eps)

        #convergence check:
        relative_change = (torch.linalg.norm(p_next - p_curr, dim=1)/ torch.clamp(torch.linalg.norm(p_curr, dim=1),min=1e-12))
        p_curr = p_next
        if torch.all(relative_change<tol).item():
            converged=True
            break
        

    # Final synchronization to capture the end of GPU execution
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    runtime = time.perf_counter() - t0

    return p_curr, iter_num + 1, converged, runtime


def run_slim5(y_noisy, A_aug, p0, tol=1e-3):
    """
    same as slim algorithm, only now limiting it to only 5 iterations
    now the weights are:
    w_k(slim A)=1/p_k
    the SLIM A update is: (according to weighted-SPICE paper):
    p_k^(i+1)=(P_k^i)^(3/2)|a_k^*R_i^-1y|
    """
    return run_slim(y_noisy, A_aug, p0, max_iter=5, tol=tol)
