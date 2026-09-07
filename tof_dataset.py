import pickle

import torch
from torch.utils.data import Dataset
from model_config import DATASET_DIR, DEVICE


class TOFTrainingDataset(Dataset):
    """
    this class is aimed to use the data that was generated in the data genrator.
    this function reads the pkl samples and converts them into tensors
    makes the data accesible to training process
    """
    def __init__(self, data_dir=DATASET_DIR, data_size=1000, file_paths=None):
        if not file_paths:
            file_paths = [f"{data_dir}/{i}.pkl" for i in range(data_size)]
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        f_path = self.file_paths[idx]
        with open(f_path, 'rb') as file:
            sample = pickle.load(file)
        scenario_type = sample["scenario_type"]
        scenario_label = 0 if scenario_type == "baseline" else 1
        return {
            "sample_idx": sample["sample_idx"],
            "y_noisy": torch.tensor(sample["y_noisy"], dtype=torch.complex64, device=DEVICE),
            "A_aug": torch.tensor(sample["A_aug"], dtype=torch.complex64, device=DEVICE),
            "p0": torch.tensor(sample["p0"], dtype=torch.float32, device=DEVICE),
            "P_true_sig": torch.tensor(sample["P_true_sig"], dtype=torch.float32, device=DEVICE),
            # for the new tof loss:
            "true_tau_idx": torch.tensor(sample["true_tau_idx"], dtype=torch.long).to(DEVICE),
            "tau_grid_ns": torch.tensor(sample["tau_grid"] * 1e9, dtype=torch.float32).to(DEVICE),
            "tau_grid": torch.tensor(sample["tau_grid"], dtype=torch.float32).to(DEVICE),
            "true_tau_ns": torch.tensor(sample["true_tau"] * 1e9, dtype=torch.float32).to(DEVICE),
            # metadata
            "scenario_type": sample["scenario_type"],
            "scenario_label": torch.tensor(scenario_label, dtype=torch.long, device=DEVICE),
            "snr_db": sample["snr_db"],
            "K": sample["K"],
            "M": sample["M"],
        }
