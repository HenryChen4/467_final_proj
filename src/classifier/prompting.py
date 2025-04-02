from src.classifier.classifier_base import Classifier_Base

class Prompting_Classifier(Classifier_Base):
    def __init__(self, 
                 classifier,
                 config):
        super().__init__(classifier, config)
        self.prompt = config.prompt

    # For LLM use only
    def inference(self, text):
        self.prompt = self.prompt.format(text=text)
        return self.classifier_engine.inference(self.prompt)