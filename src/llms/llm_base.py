import time
from abc import ABC, abstractmethod
from omegaconf import DictConfig

import google.generativeai as genai

class LLM_Base(ABC):
    def __init__(self,
                 api_key,
                 config: DictConfig):
        super().__init__()
        self.api_key = api_key
        self.config = config

    def upload_video(self, 
                     video_path: str):
        raise NotImplementedError

    def inference(self, 
                  prompt: str, 
                  video_path: str):
        raise NotImplementedError