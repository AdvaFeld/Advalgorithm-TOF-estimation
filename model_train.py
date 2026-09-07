import csv
import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from torch import nn, optim
from torch.utils.data import ConcatDataset
from tof_dataset import TOFTrainingDataset
from model import DeepUnfoldedSPICE, Classifier_MLP, DeepUnfoldedSPICE_piecewise, UnifiedModel

from model_config import DATASET_DIR, global_seed, TRAIN_LOG_NAME, VALIDATION_SET_NAME, CHECKPOINT_CLASSIFIER_DIR, g,CHECKPOINT_DIR, TRAIN_LOG_NAME, DEVICE
from torch.utils.data import DataLoader, Subset

from loss_func import local_softargmax_tof_loss, p_rmse_tof_loss, p_rmse_windowed_tof_loss
from model_config import global_seed, g, DATASET_DIR, CHECKPOINT_DIR, model_max_iterations, TRAIN_LOG_NAME, DEVICE, \
    VALIDATION_SET_NAME

'''
the training of the DU model
saves the checkpoints for each epoch in the CHECKPOINT_DIR/run_name/epoch_{epoch}.pt
saves info about the training and validation loss in the CHECKPOINT_DIR/run_name/train_log.csv
saves the validation set file names in the CHECKPOINT_DIR/run_name/val_dataset_paths.pkl
'''

#The DU with the smaller amount of parameters q.shape=(3,10), P loss function, with the windowed square around the true tof in the P vec:
def train_small_model():
    train_dataset_ratio = 0.8
    batch_size = 16
    validation_batch_size = 16*16
    num_epochs = 50
    fixed_win_bins = 4
    beta = 30.0
    run_name = "40K_baseline_small_q_p_rmse"
    dataset_name = "dataset_baseline"
    # epoch num to continue training from, if not trained before set as -1
    start_epoch_num = -1
    num_samp_train=40*1000
    num_samp_val=5*1000
    rng = np.random.default_rng(global_seed)
    tot_samples=num_samp_train+num_samp_val
    dataset = TOFTrainingDataset(data_dir=dataset_name, data_size=tot_samples)
    #train_data_size = int(train_dataset_ratio * num_samples)
    #train_data_size = int(80*1000) #40k for baseline, 40k for the rest of the scenarios
    sample_ids = list(range(tot_samples))
    #rng.shuffle(sample_ids)

    train_dataset = Subset(dataset, sample_ids[:num_samp_train])
    val_dataset = Subset(dataset, sample_ids[num_samp_train:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=False)

    # create checkpoint dir
    check_dir = f"{CHECKPOINT_DIR}/{run_name}"
    print(check_dir)
    os.makedirs(check_dir, exist_ok=True)

    # create loss log file if doesnt exist or override an existing one if start training scratch
    log_file_path = f"{check_dir}/{TRAIN_LOG_NAME}"
    if start_epoch_num < 0:
        with open(log_file_path, 'w', newline='') as f:
            csv.writer(f).writerow(["epoch", "avg_train_loss", "avg_val_loss"])

    # create validation names
    valid_file_names = [dataset.file_paths[i] for i in sample_ids[num_samp_train:]]
    with open(check_dir + "/" + VALIDATION_SET_NAME, 'wb') as file:
        pickle.dump(valid_file_names, file)


    model = DeepUnfoldedSPICE(max_iter=model_max_iterations)
    # if we want to continue from a specific epoch of an already trained model
    if start_epoch_num >= 0:
        run_path = CHECKPOINT_DIR + "/" + run_name
        checkpoint_path = f"{run_path}/epoch_{start_epoch_num}.pt"
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)


    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model = model.to(DEVICE)

    print("Started Training")
    for epoch in range(start_epoch_num+1, num_epochs):
        model.train()
        train_loss_sum=0.0
        train_num_batches=0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:

            optimizer.zero_grad()

            p_hat = model(batch["y_noisy"], batch["A_aug"], batch["p0"]) #the forward pass

            # New ToF-domain differentiable loss:
            # loss, tau_hat, true_tau = local_softargmax_tof_loss(
            #     p_hat=p_hat,
            #     true_tau_idx=batch["true_tau_idx"],
            #     tau_grid_ns=batch["tau_grid_ns"],
            #     scenario=batch["scenario_type"],
            #     beta=beta,
            # )
            # loss = p_rmse_tof_loss(p_hat=p_hat, P_true_sig=batch["P_true_sig"])
            loss = p_rmse_windowed_tof_loss(p_hat=p_hat, P_true_sig=batch["P_true_sig"], true_tau_idx=batch["true_tau_idx"])

            loss.backward() #backpropagation
            optimizer.step() #parameter update
            train_loss_sum += loss.item() * batch["y_noisy"].shape[0]
            train_num_batches += batch["y_noisy"].shape[0]
        avg_train_loss = train_loss_sum / train_num_batches

        # saving checkpoint for each epoch
        save_path = f"{check_dir}/epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path)

        #validation part:
        model.eval()
        val_loss_sum=0.0
        val_num_batches=0
        val_progress_bar = tqdm(val_loader, desc=f"validation")
        with torch.no_grad(): #we want to evaluation without building gradient graphs (faster, cleaner, and we dont really need parameters update)
            for batch in val_progress_bar:
                p_hat = model(batch["y_noisy"], batch["A_aug"], batch["p0"])

                # loss, tau_hat, true_tau = local_softargmax_tof_loss(
                #     p_hat=p_hat,
                #     true_tau_idx=batch["true_tau_idx"],
                #     tau_grid_ns=batch["tau_grid_ns"],
                #     scenario=batch["scenario_type"],
                #     beta=beta,
                # )
                # loss = p_rmse_tof_loss(p_hat=p_hat, P_true_sig=batch["P_true_sig"])
                loss = p_rmse_windowed_tof_loss(p_hat=p_hat, P_true_sig=batch["P_true_sig"],
                                                true_tau_idx=batch["true_tau_idx"])
                val_loss_sum += loss.item() * batch["y_noisy"].shape[0]
                val_num_batches += batch["y_noisy"].shape[0]
        avg_val_loss = val_loss_sum / val_num_batches

        # append loss to log file for each epoch
        with open(log_file_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss])

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")



#The DU with the piecewise loss function, q.shape=(3,465,10), with the windowed square around the true tof in the P vec:
def train_piecewise_model():
    train_dataset_ratio = 0.8
    batch_size = 16
    validation_batch_size = 16*16
    num_epochs = 50
    fixed_win_bins = 4
    beta = 30.0
    run_name = "40K_piecewise_q_p_rmse"
    dataset_name = "dataset_rest_of_scenarios"
    # epoch num to continue training from, if not trained before set as -1
    start_epoch_num = -1
    num_samp_train=40*1000
    num_samp_val=5*1000
    rng = np.random.default_rng(global_seed)
    tot_samples=num_samp_train+num_samp_val
    dataset = TOFTrainingDataset(data_dir=dataset_name, data_size=tot_samples)
    #train_data_size = int(train_dataset_ratio * num_samples)
    #train_data_size = int(80*1000) #40k for baseline, 40k for the rest of the scenarios
    sample_ids = list(range(tot_samples))
    #rng.shuffle(sample_ids)

    train_dataset = Subset(dataset, sample_ids[:num_samp_train])
    val_dataset = Subset(dataset, sample_ids[num_samp_train:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=False)

    # create checkpoint dir
    check_dir = f"{CHECKPOINT_DIR}/{run_name}"
    os.makedirs(check_dir, exist_ok=True)
    print(check_dir)
    # create loss log file if doesnt exist or override an existing one if start training scratch
    log_file_path = f"{check_dir}/{TRAIN_LOG_NAME}"
    if start_epoch_num < 0:
        with open(log_file_path, 'w', newline='') as f:
            csv.writer(f).writerow(["epoch", "avg_train_loss", "avg_val_loss"])

    # create validation names
    valid_file_names = [dataset.file_paths[i] for i in sample_ids[num_samp_train:]]
    with open(check_dir + "/" + VALIDATION_SET_NAME, 'wb') as file:
        pickle.dump(valid_file_names, file)


    model = DeepUnfoldedSPICE_piecewise(max_iter=model_max_iterations)
    # if we want to continue from a specific epoch of an already trained model
    if start_epoch_num >= 0:
        run_path = CHECKPOINT_DIR + "/" + run_name
        checkpoint_path = f"{run_path}/epoch_{start_epoch_num}.pt"
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)


    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model = model.to(DEVICE)

    print("Started Training")
    for epoch in range(start_epoch_num+1, num_epochs):
        model.train()
        train_loss_sum=0.0
        train_num_batches=0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:

            optimizer.zero_grad()

            p_hat = model(batch["y_noisy"], batch["A_aug"], batch["p0"]) #the forward pass

            # New ToF-domain differentiable loss:
            # loss, tau_hat, true_tau = local_softargmax_tof_loss(
            #     p_hat=p_hat,
            #     true_tau_idx=batch["true_tau_idx"],
            #     tau_grid_ns=batch["tau_grid_ns"],
            #     scenario=batch["scenario_type"],
            #     beta=beta,
            # )
            # loss = p_rmse_tof_loss(p_hat=p_hat, P_true_sig=batch["P_true_sig"])
            loss = p_rmse_windowed_tof_loss(p_hat=p_hat, P_true_sig=batch["P_true_sig"], true_tau_idx=batch["true_tau_idx"])

            loss.backward() #backpropagation
            optimizer.step() #parameter update
            train_loss_sum += loss.item() * batch["y_noisy"].shape[0]
            train_num_batches += batch["y_noisy"].shape[0]
        avg_train_loss = train_loss_sum / train_num_batches

        # saving checkpoint for each epoch
        save_path = f"{check_dir}/epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path)

        #validation part:
        model.eval()
        val_loss_sum=0.0
        val_num_batches=0
        val_progress_bar = tqdm(val_loader, desc=f"validation")
        with torch.no_grad(): #we want to evaluation without building gradient graphs (faster, cleaner, and we dont really need parameters update)
            for batch in val_progress_bar:
                p_hat = model(batch["y_noisy"], batch["A_aug"], batch["p0"])

                # loss, tau_hat, true_tau = local_softargmax_tof_loss(
                #     p_hat=p_hat,
                #     true_tau_idx=batch["true_tau_idx"],
                #     tau_grid_ns=batch["tau_grid_ns"],
                #     scenario=batch["scenario_type"],
                #     beta=beta,
                # )
                # loss = p_rmse_tof_loss(p_hat=p_hat, P_true_sig=batch["P_true_sig"])
                loss = p_rmse_windowed_tof_loss(p_hat=p_hat, P_true_sig=batch["P_true_sig"],
                                                true_tau_idx=batch["true_tau_idx"])
                val_loss_sum += loss.item() * batch["y_noisy"].shape[0]
                val_num_batches += batch["y_noisy"].shape[0]
        avg_val_loss = val_loss_sum / val_num_batches

        # append loss to log file for each epoch
        with open(log_file_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss])

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")



def train_classifier():
    train_dataset_ratio = 0.8
    batch_size = 16
    validation_batch_size = 16*16
    num_epochs = 40
    run_name = "classifier_cart_90K_sigmoid" #cart for cartesian (real&imagenary)
    dataset_name = "dataset_total" #the dataset that contains all the scenarios
    # epoch num to continue training from, if not trained before set as -1

    dataset1 = TOFTrainingDataset(data_dir="dataset_rest_of_scenarios", data_size=45*1000)
    dataset2 = TOFTrainingDataset(data_dir="dataset_baseline", data_size=45*1000)
    num_samp_train=40*1000
    num_samp_val=5*1000
    num_tot = num_samp_train + num_samp_val # 40k for baseline, 40k for the rest of the scenarios, 10k for validation
    sample_ids = list(range(num_tot))
    train_dataset1 = Subset(dataset1, sample_ids[:num_samp_train])
    val_dataset1 = Subset(dataset1, sample_ids[num_samp_train:])
    train_dataset2 = Subset(dataset2, sample_ids[:num_samp_train])
    val_dataset2 = Subset(dataset2, sample_ids[num_samp_train:])

    # Total length will be len(dataset1) + len(dataset2)
    dataset_train =ConcatDataset([train_dataset1, train_dataset2])
    dataset_val = ConcatDataset([val_dataset1, val_dataset2])
    
    # Example usage:
    # Index 0 returns dataset1[0]
    # Index len(dataset1) returns dataset2[0]
    start_epoch_num = -1 #starting from beginning, if we want to continue from a specific epoch of an already trained model, set it to the epoch number
    loss_fn = nn.BCEWithLogitsLoss()

    #rng.shuffle(sample_ids)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(dataset_val, batch_size=validation_batch_size, shuffle=False)

    # create checkpoint dir
    check_dir = f"{CHECKPOINT_CLASSIFIER_DIR}/{run_name}"
    os.makedirs(check_dir, exist_ok=True)
    print(check_dir)
    # create loss log file if doesnt exist or override an existing one if start training scratch
    log_file_path = f"{check_dir}/{TRAIN_LOG_NAME}"
    if start_epoch_num < 0:
        with open(log_file_path, 'w', newline='') as f:
            csv.writer(f).writerow(["epoch", "avg_train_loss", "avg_val_loss"])


    model = Classifier_MLP() #we have 4 scenarios, but we use a binary prespective: baselines\not baseline, binary classifier: one output logit
    # if we want to continue from a specific epoch of an already trained model
    if start_epoch_num >= 0:
        run_path = CHECKPOINT_CLASSIFIER_DIR + "/" + run_name
        checkpoint_path = f"{run_path}/epoch_{start_epoch_num}.pt"
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model = model.to(DEVICE)

    print("Started Training Classifier")
    for epoch in range(start_epoch_num+1, num_epochs):
        model.train()
        train_loss_sum=0.0
        train_num_batches=0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            optimizer.zero_grad()
            logits = model(batch["y_noisy"]) #the forward pass, shape [B,1]
            targets=(batch["scenario_label"].float().unsqueeze(1)) #shape [B,1]
            # New ToF-domain differentiable loss:
            loss = loss_fn(logits, targets)

            loss.backward() #backpropagation
            optimizer.step() #parameter update
            train_loss_sum += loss.item() * batch["y_noisy"].shape[0]
            train_num_batches += batch["y_noisy"].shape[0]
        avg_train_loss = train_loss_sum / train_num_batches

        # saving checkpoint for each epoch
        save_path = f"{check_dir}/epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path) #in case we stop the running in the middle, we save the parameters of the epoch and can continue from there

        #validation part:
        model.eval()
        val_loss_sum=0.0
        val_num_batches=0
        val_progress_bar = tqdm(val_loader, desc=f"validation")
        with torch.no_grad(): #we want to evaluation without building gradient graphs (faster, cleaner, and we dont really need parameters update)
            for batch in val_progress_bar:
                logits = model(batch["y_noisy"])
                targets=(batch["scenario_label"].float().unsqueeze(1))
                loss = loss_fn(logits, targets)
                val_loss_sum += loss.item() * batch["y_noisy"].shape[0]
                val_num_batches += batch["y_noisy"].shape[0]
        avg_val_loss = val_loss_sum / val_num_batches

        # append loss to log file for each epoch
        with open(log_file_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss])

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")





#running the training of the models:
if __name__ == "__main__":
    #train_small_model()
    #train_piecewise_model()
    train_classifier()