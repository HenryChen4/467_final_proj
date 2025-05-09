Installation instructions:

Upon cloning this repository, first make the following directories in the root directory of this project:

./data

./data_results

./results

The data used in this project: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

Download the Combined Data.csv and drag and stick it into the ./data directory you created. Rename the csv file to "main_data.csv"

```
conda create --name henry_final_proj python=3.10
conda activate henry_final_proj
pip install -r requirements.txt
```

Commands:

### To run the classification script for naive bayes and LSTM-Softmax:

```
python -m src.classification classifier=<classifier name> data=emotion_1
```

`The above command with train the classifier and evaluate its scores which are stored in the ./results directory under a directory named ./timestamp_classification_<classifier name>`

Available classifier names:
- naive_bayes
- softmax

Example:

```
python -m src.classification classifier=naive_bayes data=emotion_1
```

### To run the classification script for majority vote LLMs:

```
python -m src.classification llm=gemini classifier=prompting prompt=prompt_1 data=emotion_1 n_classifiers=3
```

Sometimes the LLM will take awhile to run. If you want to evaluate its current metrics, run:

```
python -m src.llm_eval results_dir=./results/<results_dir> data=emotion_1
```

`<results_dir> is the latest classification run directory that will be automatically generated in the ./results directory`

### To evaluate the data:

```
python -m src.data_eval data=emotion_1
```

### To run tuning scripts:

```
python -m src.tuning results_dir=./results/<classifier_name>_tuning classifier=<classifier_name> data=emotion_1
```

`## Note: The directory ./results/<classifier_name>_tuning must already exist`

If you want to try out your own hyperparameters, the yaml files for each classifier and the data contain values to edit.
