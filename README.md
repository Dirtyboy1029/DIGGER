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

      python finetune.py
      Parameter model_type: 
           benchmark:  for Reference LLM
           test: for vanilla-tuned
           union: for Reference-tuned

      

#### 3. train correction model and correction

train：myexperiment/uncertainity_metrics_utils/ml_true_flase.py

      python ml_true_flase.py -experiment_type train -save_model y -data_type small_drebin -banlance n -train_data_size 1.0
      
      ###  experiment_type: Type of experiment,training correction model or resultant correction.
      ###  save_model: Whether to save trained correction models
      ###  data_type: Data types for training corrective models，(small_drebin,small_multi)
      ###  train_data_size: The effect of the scale of the training data on the corrective model,[1.0,0.8,0.4,0.2,0.1]
      ###  banlance: The effect of whether the training data is balanced or not on the corrective model

correction： myexperiment/uncertainity_metrics_utils/ml_true_flase.py

      python ml_true_flase.py -experiment_type test -data_type small_amd -banlance n -train_data_size 1.0 -test_model_type small_drebin

      ###  Show the results of training the correction model using the uncertainty metrics of the drebin dataset obtained from deepdrebin's model, using the unbalanced full set of data, and correct the AMD dataset.
