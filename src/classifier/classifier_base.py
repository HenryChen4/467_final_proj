from abc import ABC, abstractmethod

class Classifier_Base(ABC):
    @abstractmethod
    def __init__(self, classifier_engine, config):
        super().__init__()
        self.classifier_engine = classifier_engine
        self.config = config