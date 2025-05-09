import numpy as np

from src.classifier.classifier_base import Classifier_Base

class Prompting_Classifier(Classifier_Base):
    def __init__(self, 
                 classifier,
                 config):
        super().__init__(classifier, config)
        self.prompt = config.prompt

        if(config.use_fewshot):
            self.seed = config.random_example_seed

            rng = np.random.default_rng(seed=self.seed)
        
            self.fewshot_descriptions = config.fewshot_desc
            self.fewshot_examples = config.fewshot_examples
            
            self.prompt = ""

            for label, description in self.fewshot_descriptions.items():
                random_example = str(self.fewshot_examples[label][int(rng.choice([0, 1, 2]))])
                self.prompt += "\"" + random_example + "\"\n\n"
                self.prompt += description[0] + "\n\n"
            
            self.prompt += str(config.prompt)

    # For LLM use only
    def inference(self, text):
        self.prompt = self.prompt.format(text=text)
        return self.classifier_engine.inference(self.prompt)