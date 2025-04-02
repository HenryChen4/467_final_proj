Installation instructions:


```
conda create --name henry_final_proj python=3.10
conda activate henry_final_proj
pip install -r requirements.txt
```


Command:
```
python -m src.classification llm=gemini strategy=prompting_fewshot prompt=prompt_1 data=sempi
```