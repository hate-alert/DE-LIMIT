# DELIMIT--DeEpLearning models for MultIlingual haTespeech

------------------------------------------
***Folder Description***
------------------------------------------
~~~

./Dataset             --> Contains the dataset
./BERT_Classifier  	  --> Contains the codes for BERT classifiers performing binary classifier on the dataset
./LASER+LR 			  --> Containes the codes for Logistic regression classifier used on top of LASER embeddings
./Translation         --> Code for translating the Non-English datasets to English.

~~~

## Requirements 

Make sure to use **Python3** when running the scripts. The package requirements can be obtained by running `pip install -r requirements.txt`.


------------------------------------------
***Instructions for training the classifier models***
------------------------------------------

1. **mBERT Baseline**
	1. Set the `language` you wish to train on in the `params` dictionary. 
	2. Load the datasets into the model using the data_loader function as shown in `BERT_training_inference.py`, using the parameters `files` to specify the dataset directory, `csv_file` set as `*_full.csv` in order to load the untranslated dataset.
	3. Load the pretrained bert model required, using the parameters `path_files`, `which_bert`
	4. Set the `how_train` parameter in `BERT_training_inference.py` to `baseline`, and set the parameters `sample_ratio`, `take_ratio`, and `samp_strategy` depending on the experiment setting. 
	5. Call the train_model function. It trains the bert model with the dataset given, for the specified number of epochs. Use the parameter `to_save` for saving the model at the epoch having best validation scores.

2. **mBERT All_but_one**
	1. Similar to the instructions above, set the required parameters for target language, bert model to be used and sample ratio of the target dataset.
	2. Set the `how_train` parameter to `all_but_one`. Now data_loader function will load the datasets all other language fully, and the dataset for the target language in the given sample ratio. 

3. **Translation + BERT Baseline**
	1. Set the language and other parameters similar to mBERT baseline case. 
	2. Set the `csv_file` parameter to `*_translated.csv`. Now data_loader function will load the csv files containing the texts translated to English.

4. **CNN+GRU Baseline**

5. **LASER+LR baseline**
6. **LASER+LR all_but_one