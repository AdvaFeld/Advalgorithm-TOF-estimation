import numpy as np
import torch
import random
import os

# choose which device to run on
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
print(f"Training on: {DEVICE}")


# files and folders names
DATASET_DIR = "dataset_rest_of_scenarios" #train data
TEST_SET_DIR = "test_set"
CHECKPOINT_DIR = "checkpoint"
CHECKPOINT_CLASSIFIER_DIR = "checkpoint_classifier"
CHECKPOINT_UNIFIED_DIR = "checkpoint_unified"
TRAIN_LOG_NAME = "train_log.csv"
RESULT_LOG_NAME = "result_log_test.parquet"
VALIDATION_SET_NAME = "val_dataset_paths.pkl"

# all scenarios and their configs
scenario_type_dict = {
    #"baseline": {"min_gap": 25.0, "max_gap": 70.0, 'loss_win_size': 20},
    "moderate": {"min_gap": 10.0, "max_gap": 20.0, 'loss_win_size': 8},
    "close": {"min_gap": 3.0, "max_gap": 6.0,'loss_win_size': 2},
    "very_close": {"min_gap": 1.0, "max_gap": 2.0, 'loss_win_size': 1},
}

#converting the scenario names to a numbers list, so the mlp classificator will understand them (instead of a string):
scenario_names = list(scenario_type_dict.keys()) #["baseline", "moderate", "close", "very_close"]
num_scenarios = len(scenario_names) #4 for now
#creating a dict that converts a scenario name into a number
scenario_name_to_id = {
    scenario_name: scenario_id
    for scenario_id, scenario_name in enumerate(scenario_names)
}
''' the final dic becomes:{ "baseline": 0, "moderate": 1, "close": 2, "very_close": 3,}'''
scenario_id_to_name = {
    scenario_id: scenario_name
    for scenario_name, scenario_id in scenario_name_to_id.items()
}

scenario_classifier_ckpt = f"{CHECKPOINT_DIR}/scenario_classifier.pt"


model_max_iterations = 10

snr_db_choices = (0, 5, 10, 15, 20, 25, 30)


# sets seed everywhere for deterministic randomness
global_seed = 2026
def seed_everything(seed):
    """
    sets constant seed everywhere in the code
    """
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Optional: Force CuDNN to be deterministic (can slow down performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
seed_everything(global_seed)
# torch.manual_seed(global_seed)
#  Creates a generator for the DataLoader for deterministic shuffle
g = torch.Generator()
g.manual_seed(global_seed)
