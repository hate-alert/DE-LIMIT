------------------------------------------
***Instructions for LASER+LR models***
------------------------------------------

1. **LASER+LR baseline**
	1. Generate the LASER embeddings for the datasets of the target language. Refer to the [LASER github repository](https://github.com/facebookresearch/LASER) for guidelines on how to install and generate the embeddings.
	2. The code expects the embeddings to be present in the directory `Dataset/embedding`, with the train, val and test files present respectively in the subdirectories of `train`,`val`,`test` in the `embeddings` folder. The name of the file is expected to be **{language}.csv**. E.g. English.csv, German.csv, etc
	3. For running the baseline experiment, make use of the `LASER+LR/Baselines.ipynb` notebook to run the Logistic regression model. You can choose the language, and sample size by passing them as paramters to the function. Refer to the notebook for further details and instructions. 

2. **LASER+LR all_but_one**
	1. Similar to the above case, generate the laser embeddings and place them in respective directories in the `Dataset/embedding` folder. 
	2. Use the files `LASER+LR/All_but_one.ipynb` and `LASER+LR/All.ipynb` to run the all_but_one experiments. The `All_but_one.ipynb` notebook contains code for running the all_but_one experiment using a sample of target language points (including zero-shot case). The `All.ipynb` file contains codes for running the experiment using all the datasets available, i.e. 100% of training dataset. 