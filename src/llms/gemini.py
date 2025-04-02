import time
import hydra

import google.generativeai as genai

from src.llms.llm_base import LLM_Base

class Gemini_Agent(LLM_Base):
    def __init__(self, api_key, config):
        super().__init__(api_key, config)        
        self.llm = genai.GenerativeModel(**self.config)
    
    def inference(self, prompt):
        genai.configure(api_key=self.api_key)
        response = self.llm.generate_content([prompt],
                                             request_options={"timeout": 600})
        time.sleep(12)        
        print(response.text.strip())
        return response.text
