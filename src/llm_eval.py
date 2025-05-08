import os
import json
import hydra
import pickle
 
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.utils.majority_vote import majority_vote
from src.utils.metrics import save_metrics
from src.classification import initialize_data

def clean_responses(json_file):
    with open(json_file, "r") as f:
        dirty_responses = json.load(f) 
    
    possible_classes = ["Normal", "Depression", "Suicidal", "Anxiety", "Stress", "Bipolar", "Personality disorder"]
    cleaned_responses = {}
    for sample_num, all_responses in dirty_responses.items():
        dirty_cnt = 0
        cleaned_responses[sample_num] = {} 
        for classifier, response in all_responses.items():
            for classification in possible_classes:
                if not (classification.lower() in response.strip().lower()):
                    dirty_cnt += 1
                else:
                    cleaned_responses[sample_num][classifier] = classification.lower()
        if dirty_cnt == 21:
            del cleaned_responses[sample_num]

    return cleaned_responses

@hydra.main(version_base=None, config_path="../config", config_name="llm_eval_config.yaml")
def main(cfg: DictConfig):
    output_dir = cfg.results_dir
    json_file_path = os.path.join(output_dir, "llm_responses.json")

    # Initialize required items
    parser_obj = initialize_data(cfg.data)

    # TODO: Add a config variable detailing which loader to compare data to
    _, test_loader, _ = parser_obj.fill_dataloaders()
    cleaned_responses = clean_responses(json_file_path)
    mv_responses = majority_vote(cleaned_responses)

    # Get metrics
    true_labels = [label[0] for _, label in test_loader]
    mv_key_indices = list(mv_responses.keys())

    y_true = [true_labels[int(i)].lower() for i in mv_key_indices]
    y_pred = [mv_responses[i]["majority"] for i in mv_key_indices]
    
    save_metrics(y_true, y_pred, output_dir)

if __name__ == "__main__":
    main()