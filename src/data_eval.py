import os
import json
import hydra
import pickle
 
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.utils.majority_vote import majority_vote
from src.utils.metrics import save_metrics
from src.classification import initialize_data

def output_statistics(data_loader):
    # TODO: Count the number of labels for each dataset used
    labels = ["Normal", "Depression", "Suicidal", "Anxiety", "Stress", "Bipolar", "Personality disorder"]

    # TODO: Create a histogram for data using 2 different seeds: 42 & 489149
    # TODO: Creata a 3D histogram showing number of label samples used per split (train, test, val) for 2 different seeds

@hydra.main(version_base=None, config_path="../config", config_name="data_eval.yaml")
def main(cfg: DictConfig):
    parser_obj = initialize_data(cfg.data)
    train_loader, test_loader, val_loader = parser_obj.fill_dataloaders()

if __name__ == "__main__":
    main()