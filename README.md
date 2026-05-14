# QUEST
Unofficial implementation of the "Anatomy of a Query: W5H Dimensions and FAR Patterns for Text-to-SQL Evaluation" [paper](https://arxiv.org/abs/2605.05525). The paper doesn't come with companion source code, so, after reading it, I tried, with the help of Claude, to implement the proposed framework in Python. Unlike the original research work, where Google's Gemini has been used as LLM, the goal here was to implement something that could run locally and offline: the supported models need to be hosted in a local [Ollama](https://ollama.com/) server. To date, only the W5H dimensional classifier has been implemented. I have also added a script to run a W5H benchmark towards the ```gretelai/synthetic_text_to_sql``` [dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql), available in the Hugging Face Hub.  
#### Code Execution
Prerequisites to execute the code in this repo are:  
* Python 3.12+
* Ollama server
* A GPU with at least 8 GB of VRAM
  
The reference model is ```qwen2.5:7b-instruct-q4_K_M```, available in the Ollama's model catalog. It needs to be pulled from there before running the code:  
```ollama pull qwen2.5:7b-instruct-q4_K_M```  
Then clone this repo, create a Python virtual environment, activate it and install the requirements listed in [requirements.txt](requirements.txt).  
To verify that everything is fine, you can execute:  
```
cd quest
python w5h_classifier.py
```  
It runs on some of the prompts mentioned in the paper.  The script includes the ```print_profile``` function, which, for each parsed prompt, formats the output as follows:  
```
========================================================================
  Query : Which patients over 65 were readmitted within 30 days due to surgical complications?
  Active: Active dimensions: [WHO, WHAT, WHEN, WHY]
========================================================================
  Dimension  Active     Conf  Evidence
------------------------------------------------------------------------
  WHO        ✓          1.00  patients over 65
  WHAT       ✓          1.00  readmitted
  WHERE      –          0.00  —
  WHEN       ✓          1.00  within 30 days
  WHY        ✓          1.00  due to surgical complications
  HOW        –          0.00  —
------------------------------------------------------------------------
  ⚠  Frontier dimension warnings:
    WHY: WHY is a frontier dimension. Causal/explanatory constraints often
        require inference that SQL semantics do not natively
        support (e.g. foreign-key chains or cross-schema
        linkages). Verify that this constraint can be grounded in
        a recorded field.
========================================================================
```  
To run the benchmark over one of the 100 domains in the ```gretelai/synthetic_text_to_sql``` dataset:  
```
cd benchmarks
python w5h_benchmark.py --domain pharmaceuticals --samples 5
```  
You must provide a valid domain name and the max number of samples to parse within the given domain.  
