from abc import ABC, abstractmethod

class Parser_Base(ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def fill_dataloaders(self):
        raise NotImplementedError