import torch
import torch.nn as nn
from model_config import model_max_iterations

'''
the script that gathers all the 3 models that build our entire system
'''
class Classifier_MLP(nn.Module):
    def __init__(self, input_dim=2*64, hidden_dim=128, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    @staticmethod
    def prepare_input(x):
        return torch.cat([x.real, x.imag], dim=1).squeeze(-1)

    def forward(self, x):
        return self.mlp(self.prepare_input(x))


# FINALY- the models:
class DeepUnfoldedSPICE(nn.Module):
    def __init__(self, max_iter=model_max_iterations, eps=1e-8):
        """
        Deep unfolding model for the SPICE/LIKES/SLIM family.
        later on it was decided to represent the smaller model, and the piecewise model inherites
        everything besides the init_q_logits function, which is different for the two models.
        
        Parameters:
        max_iter : int, Number of unfolded layers / iterations. (set to 10 for now).
        eps : float, Small numerical stability constant to avoid division by zero
            and sqrt(0) issues.
        """
        super().__init__()
        self.max_iter = max_iter
        self.eps = eps

        # Raw learnable weights for each unfolded layer. Shape: (max_iter, 3)
        # Each row corresponds to [q1, q2, q3] for one layer,
        # before softmax normalization.
        self.q_logits = self.init_q_logits()

    def init_q_logits(self):
        return nn.Parameter(torch.zeros(self.max_iter, 3, dtype=torch.float32))

    # B is for batch size.
    def forward(self, y_noisy, A_aug, p0):
        """
        Forward pass of the unfolded model.

        Inputs:
        y_noisy: torch.Tensor, shape (B, M, 1), complex
            Batched noisy received signal.

        A_aug: torch.Tensor, shape (B, M, K+M), complex
            Batched augmented dictionary [A, I_M].

        p0 : torch.Tensor, shape (B, K+M), real
            Batched initial power vectors

        Returns:
        p: torch.Tensor, shape (B, K+M), real
            Final unfolded power estimate after max_iter layers. p as for p_hat
        """
        p = p0.clone()

        B = y_noisy.shape[0]  # batch size
        n_cols = A_aug.shape[2]  # K+M

        for i in range(self.max_iter):
            # Building R_i = A_aug diag(p_i) A_aug^H
            # P_diag = torch.diag_embed(p).to(A_aug.dtype)
            A_aug_H = A_aug.conj().transpose(1, 2)
            R = (A_aug * p.unsqueeze(1)) @ A_aug_H

            # Small diagonal loading for numerical stability
            M = R.shape[1]  # R is shaper (B, M, M) so R.shape[1] is M=64
            I_M = torch.eye(M, dtype=R.dtype, device=R.device).unsqueeze(0).expand(B, -1, -1)
            R = R + self.eps * I_M

            # Building z_i = R_i^{-1} y_noisy (as in the 2011 SPICE paper)
            z = torch.linalg.solve(R, y_noisy)

            # Compute r_i = A_aug^H z_i
            # A_aug's shape is (B,M,K+M), z's shape is (B,M,1)
            r = A_aug_H @ z  # r's shape is (B,K+M,1)
            # so I removed the last dimension with the squeeze command
            r_abs = torch.abs(r.squeeze(-1))

            # calculation of w_spice w_spice(k) = ||a_k||_2^2
            w_spice = torch.sum(torch.abs(A_aug) ** 2, dim=1)
            w_spice = torch.clamp(w_spice, min=self.eps)
            # clamp function force values to stay within a chosen range (so it wont be divided by zero).

            # same for w_likes w_likes(k) = a_k^H R^{-1} a_k
            R_inv_A = torch.linalg.solve(R, A_aug)
            w_likes = torch.sum(A_aug.conj() * R_inv_A, dim=1).real
            w_likes = torch.clamp(w_likes, min=self.eps)

            # and for w_slim w_slim(k) = 1 / p_k
            w_slim = 1.0 / torch.clamp(p, min=self.eps)

            # Softmax on the learnable q values of layer i
            # q shape (max_iter, 3) but here its i-th layer
            q = torch.softmax(self.q_logits[i], dim=0)
            # shape(q[i]) is (3,) and it contains the normalized weights for the three regularizers for layer i
            q1 = q[0]
            q2 = q[1]
            q3 = q[2]

            # Mixed regularizator w_mix = q1*w_spice + q2*w_likes + q3*w_slim
            w_mix = q1 * w_spice + q2 * w_likes + q3 * w_slim
            w_mix = torch.clamp(w_mix, min=self.eps)

            # Shared multiplicative power p update p_{i+1} = p_i * |r_i| / sqrt(w_mix)
            p = p * r_abs / torch.sqrt(w_mix)
            p = torch.clamp(p, min=self.eps)

        return p


class DeepUnfoldedSPICE_piecewise(DeepUnfoldedSPICE):
    def __init__(self, max_iter=model_max_iterations, eps=1e-8):
        super().__init__(max_iter, eps)

    def init_q_logits(self):
        return nn.Parameter(torch.zeros(self.max_iter, 3, 465, dtype=torch.float32))




class UnifiedModel(nn.Module):
    def __init__(self, baseline_model, super_res_model, classifier):
        super().__init__()
        self.baseline_model = baseline_model
        self.super_res_model = super_res_model
        self.classifier = classifier

    def forward(self, y_noisy, A_aug, p0, return_aux=False):
        logits = self.classifier(y_noisy)

        p_non_bl=torch.sigmoid(logits) #shape [B,1] #bl=baseline
        p_bl=1-p_non_bl
        output_bl=self.baseline_model(y_noisy,A_aug,p0)
        output_sr=self.super_res_model(y_noisy,A_aug,p0) #sr for super resolution
        p_hat = output_bl*p_bl + p_non_bl*output_sr

        if not return_aux:
            return p_hat

        return p_hat, logits, p_non_bl
    
# We return:
# p_hat -> ToF loss
# logits -> BCEWithLogitsLoss for classifier training
# p_non_bl -> routing probability and classifier confidence