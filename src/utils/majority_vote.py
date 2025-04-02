import os
import json

from tqdm import tqdm
from dask import delayed
from dask.distributed import Client
from tqdm import tqdm
from collections import Counter

def generate_annotations(classifiers,
                         text_samples,
                         output_dir):
    responses = {}

    client = Client(
        n_workers=3,
        threads_per_worker=1
    )

    cnt = 0
    for text in tqdm(text_samples):
        responses[str(cnt)] = {}

        delayed_responses = [delayed(classifier.inference)(text) for classifier in classifiers]
        futures = client.compute(delayed_responses)
        results = client.gather(futures)

        for i, result in enumerate(results):
            responses[str(cnt)][f"classifier_{i}"] = result

        with open(os.path.join(output_dir, "llm_responses.json"), "w") as f:
            json.dump(responses, f, indent=4)

        cnt += 1

    client.close()
    return responses

def majority_vote(cleaned_responses):
    for key, value in cleaned_responses.items():
        classifier_keys = [k for k in value.keys() if k.startswith("classifier_")]
        labels = [value[k] for k in classifier_keys]

        label_counts = Counter(labels)
        majority_label, count = label_counts.most_common(1)[0]

        # Handle tie case (all different)
        if count == 1 and len(set(labels)) == len(labels):
            majority_label = value[classifier_keys[0]] 

        cleaned_responses[key]["majority"] = majority_label
    return cleaned_responses