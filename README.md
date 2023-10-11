# DIGGER

 
## Overview
In this work, we introduce a universal optimization framework, denoted as Digger, designed to test whether a sample has been learned by the target LLM. We conduct a thorough feature study to understand the characteristics of sample loss and the rate of change in sample loss as samples are learned by the LLM. Based on these characteristics, we formulate the difference in loss change as an indicator to distinguish between samples that have been learned by the LLM and those that have not.

## Dependencies:
We develop the codes on Windows operation system, and run the codes on Ubuntu 20.04. The codes depend on Python 3.10.9. Other packages (e.g., transformers) can be found in the `./requirements.txt`.

##  Usage
#### 1. Data processing
Put the book in txt format into the specified directory, cut out n paragraphs, the length of each paragraph is l.

      cd Datasets
      python build_samples_set.py
     
The sample set is then randomly sliced proportionally and book lists are constructed for each dataset.

      python build_dataset.py

#### 2. Fine-tune LLM & get loss


      cd Digger
      python finetune.py
      Parameter model_type: 
           benchmark:  for Reference LLM
           test: for vanilla-tuned
           union: for Reference-tuned
      python get_loss.py

All loss values are saved in batches as npy files and stored in the outputs directory
