import os
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tof_dataset import TOFTrainingDataset
from model import Classifier_MLP
import matplotlib.pyplot as plt

from model_config import TRAIN_LOG_NAME, DEVICE, RESULT_LOG_NAME, VALIDATION_SET_NAME, CHECKPOINT_CLASSIFIER_DIR


# only part to change between runs
run_name = "classifier_cart_90K_sigmoid"
run_epoch = 34
override_result = True



run_path = CHECKPOINT_CLASSIFIER_DIR + "/" + run_name
checkpoint_path = f"{run_path}/epoch_{run_epoch}.pt"
loss_log_path = run_path + "/" + TRAIN_LOG_NAME
result_log_path = run_path + "/" + RESULT_LOG_NAME
# test_data_size = 2000
validation_batch_size = 16*16


def calc_models_validation(model):
    """
    loads saved data and model, calculates models on validation data and saves result to log file
    """

    # load validation data and put in loader
    # val_data = TOFTrainingDataset(data_dir=TEST_SET_DIR, data_size=test_data_size)
    validation_files_path = run_path + "/" + VALIDATION_SET_NAME
    with open(validation_files_path, 'rb') as file:
        val_data_files = pickle.load(file)

    val_data = TOFTrainingDataset(file_paths=val_data_files)
    val_loader = DataLoader(val_data, batch_size=validation_batch_size, shuffle=False)

    should_append = False

    with torch.no_grad():
        progress_bar = tqdm(val_loader)
        for batch in progress_bar:

            models_result = dict()
            #models_result["pred"] = torch.argmax(model(batch["y_noisy"]), dim=1)
            logits=model(batch["y_noisy"])
            p_non_bl=torch.sigmoid(logits)
            models_result["pred"]=(p_non_bl>=0.5).long().squeeze(1)
            models_result["accuracy"] = (models_result["pred"] == batch["scenario_label"])*1.0

            models_result["snr_db"] = batch["snr_db"]
            models_result["scenario_type"] = batch["scenario_type"]
            models_result["sample_idx"] = batch["sample_idx"]

            models_result = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in models_result.items()}
            # models_result = {k: list(v) if isinstance(v, np.ndarray) else v for k, v in models_result.items()}

            chunk_df = pd.DataFrame(models_result)
            chunk_df.to_parquet(result_log_path, engine='fastparquet', append=should_append)
            should_append = True


def calc_and_print_rmse(df):
    acc_tof_per_snr = df.groupby("snr_db")['accuracy'].mean().to_frame('mean_accuracy').reset_index()
    plt.plot(acc_tof_per_snr["snr_db"], acc_tof_per_snr['mean_accuracy'], marker='o')
    plt.xlabel('snr_db')
    plt.ylabel('mean_accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.title(df.name)
    plt.show()


def result_analysis():
    model = Classifier_MLP()
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    # calcs the result logs if not calculated before
    if override_result or (not os.path.exists(result_log_path)):
        calc_models_validation(model)

    # display train loss results per epoch
    df = pd.read_csv(loss_log_path)
    plt.plot(df['epoch'], df['avg_train_loss'], color='tab:blue', label='avg_train_loss', linewidth=1.5, marker='o')
    plt.plot(df['epoch'], df['avg_val_loss'], color='tab:red', label='avg_val_loss', linewidth=1.5, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('avg_loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # display saved results
    df = pd.read_parquet(result_log_path)
    # print number of samples per snr and scenario_type
    samples_count_table = pd.crosstab(df['scenario_type'], df['snr_db'])
    print(samples_count_table)
    df.groupby('scenario_type', group_keys=False).apply(calc_and_print_rmse)


if __name__ == "__main__":
    result_analysis()

