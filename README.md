# DELIMIT--DeEpLearning models for MultIlingual haTespeech

------------------------------------------
***Folder Description***
------------------------------------------
~~~

./Dataset             --> Contains the dataset
./BERT_Classifier     --> Contains the codes for BERT classifiers performing binary classifier on the dataset
./CNN_GRU			  --> Contains the codes for CNN-GRU model		
./LASER+LR 	      --> Containes the codes for Logistic regression classifier used on top of LASER embeddings
./Translation         --> Code for translating the Non-English datasets to English.

~~~

## Requirements 

Make sure to use **Python3** when running the scripts. The package requirements can be obtained by running `pip install -r requirements.txt`.


------------------------------------------
***Instructions for training the classifier models***
------------------------------------------

1. **mBERT Baseline**
	1. Download the multilingual bert model and tokenizer from the [transformers repository](https://github.com/huggingface/transformers/) and store in the folder `BERT Classifier/multilingual_bert`.
	2. Set the `language` you wish to train on in the `params` dictionary of `BERT_training_inference.py`. 
	3. Load the datasets into the model using the data_loader function as shown in `BERT_training_inference.py`, using the parameters `files` to specify the dataset directory, `csv_file` set as `*_full.csv` in order to load the untranslated dataset.
	4. Load the pretrained bert model required, using the parameters `path_files`, `which_bert`
	5. Set the `how_train` parameter in `BERT_training_inference.py` to `baseline`, and set the parameters `sample_ratio`, `take_ratio`, and `samp_strategy` depending on the experiment setting. 
	6. Call the train_model function. It trains the bert model with the dataset given, for the specified number of epochs. Use the parameter `to_save` for saving the model at the epoch having best validation scores.

2. **mBERT All_but_one**
	1. Similar to the instructions above, set the required parameters for target language, bert model to be used and sample ratio of the target dataset.
	2. Set the `how_train` parameter to `all_but_one`. Now data_loader function will load the datasets all other language fully, and the dataset for the target language in the given sample ratio. 

3. **Translation + BERT Baseline**
	1. Set the language and other parameters similar to mBERT baseline case. 
	2. Set the `csv_file` parameter to `*_translated.csv`. Now data_loader function will load the csv files containing the texts translated to English.

4. **CNN+GRU Baseline**
	1. Download the MUSE embeddings from the [MUSE github repository](https://github.com/facebookresearch/MUSE) and store them in the folder `CNN_GRU/muse_embeddings`
	2. The files for the CNN-GRU model are located in the `CNN_GRU` folder. The main file is called `CNN_GRU.py`. 
	3. In the params dictionary in `CNN_GRU.py`, set the values of parameters like `language`, `epochs`, `sample_ratio`, etc depending on the experimental setup.
	
5. **LASER+LR baseline**
	1. Generate the LASER embeddings for the datasets of the target language. Refer to the [LASER github repository](https://github.com/facebookresearch/LASER) for guidelines on how to install and generate the embeddings.
	2. The code expects the embeddings to be present in the directory `Dataset/embedding`, with the train, val and test files present respectively in the subdirectories of `train`,`val`,`test` in the `embeddings` folder. The name of the file is expected to be **{language}.csv**. E.g. English.csv, German.csv, etc
	3. For running the baseline experiment, make use of the `LASER+LR/Baselines.ipynb` notebook to run the Logistic regression model. You can choose the language, and sample size by passing them as paramters to the function. Refer to the notebook for further details and instructions. 
6. **LASER+LR all_but_one**
	1. Similar to the above case, generate the laser embeddings and place them in respective directories in the `Dataset/embedding` folder. 
	2. Use the files `LASER+LR/All_but_one.ipynb` and `LASER+LR/All.ipynb` to run the all_but_one experiments. The `All_but_one.ipynb` notebook contains code for running the all_but_one experiment using a sample of target language points (including zero-shot case). The `All.ipynb` file contains codes for running the experiment using all the datasets available, i.e. 100% of training dataset. 
