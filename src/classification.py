import os
import json
import hydra
import pickle
import torch

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.classifier.softmax import Softmax_Classifier
from src.utils.majority_vote import generate_annotations
from src.utils.metrics import save_metrics

from matplotlib import pyplot as plt
import numpy as np

# TODO:
# 1. Tune hyperparameters with validation set **DONE
# 2. Setup softmax baseline

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().run.dir

    # Initialize data
    parser_obj = initialize_data(cfg.data)
    train_loader, test_loader, val_loader = parser_obj.fill_dataloaders()

    if "softmax" in cfg.classifier._target_:
        text_vocab, label_vocab = parser_obj.build_vocab(train_loader, test_loader, val_loader)

        train_loader = parser_obj.vectorize_dataloaders(train_loader)
        test_loader = parser_obj.vectorize_dataloaders(test_loader)
        val_loader = parser_obj.vectorize_dataloaders(val_loader)

    # Initialize and train classifiers
    classifier = None
    if "naive_bayes" in cfg.classifier._target_:
        classifier = initialize_naive_bayes(cfg.classifier)
        classifier.train(train_loader)
    elif "prompting" in cfg.classifier._target_:
        llms = initialize_llms(n_classifiers=cfg.n_classifiers,
                               llm_cfg=cfg.llm)
        classifier = initialize_classifiers(classifier_engines=llms,
                                            classifier_cfg=cfg.classifier)
    elif "softmax" in cfg.classifier._target_:
        classifier = initialize_softmax(cfg.classifier, text_vocab, label_vocab)
        epoch_losses = classifier.train(train_loader)

        plt.plot(np.arange(len(epoch_losses)), epoch_losses)
        plt.show()

    # Run and store predictions
    if "naive_bayes" in cfg.classifier._target_:
        # Predictions stored right after
        text_test_labels = [label[0] for _, label in test_loader]
        test_labels = classifier.label_encoder.transform(text_test_labels)
        test_pred = classifier.predict(test_loader)

        test_labels = classifier.label_encoder.inverse_transform(test_labels)
        test_pred = classifier.label_encoder.inverse_transform(test_pred)
        save_metrics(test_labels, test_pred, output_dir)
    elif "prompting" in cfg.classifier._target_:
        # Predictions must be ran with with the llm_eval.py file (check readme.md)
        text_test_samples = [text[0] for text, _ in test_loader]
        _ = generate_annotations(classifiers=classifier,
                                 text_samples=text_test_samples,
                                 output_dir=output_dir)
    elif "softmax" in cfg.classifier._target_:
        pass    

def initialize_data(data_cfg):
    parser_obj = hydra.utils.instantiate(
        data_cfg
    )
    return parser_obj

def initialize_llms(n_classifiers,
                    llm_cfg):
    # Extract LLMs
    llms = []
    cnt = 0
    for llm_config in llm_cfg.config.llm_configs.values():
        if(cnt < n_classifiers):
            llm_instance = hydra.utils.instantiate(
                {
                    "_target_": llm_cfg._target_,
                    "api_key": llm_cfg.api_key,
                    "config": llm_config,
                }
            )
            cnt += 1
            llms.append(llm_instance)
    return llms

def initialize_softmax(softmax_cfg, text_vocab, label_vocab):
    print("INITIALIZING SOFTMAX")
    classifier = hydra.utils.instantiate(softmax_cfg)
    classifier.finish_initialization(text_vocab, label_vocab)
    return classifier

def initialize_naive_bayes(naive_bayes_cfg):
    classifier = hydra.utils.instantiate(naive_bayes_cfg)
    return classifier

def initialize_classifiers(classifier_engines,
                           classifier_cfg):
    # Extract engines
    classifiers = []
    for engine in classifier_engines:
        OmegaConf.resolve(classifier_cfg)
        classifier = hydra.utils.instantiate(
            {
                "_target_": classifier_cfg._target_,
                "classifier": engine,
                "config": classifier_cfg.config
            }
        )
        classifiers.append(classifier)
    return classifiers

if __name__ == "__main__":
    main()