[![HitCount](http://hits.dwyl.com/punyajoy/DE-LIMIT.svg)](http://hits.dwyl.com/punyajoy/DE-LIMIT)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/punyajoy/DE-LIMIT/issues)
# Deeplearning models for multilingual hate speech

:portugal: :saudi_arabia: :poland: :indonesia: :it: Solving the problem of hate speech detection in **9 languages** across **16 datasets**.
:fr: :us: :es: :de:

------------------------------------------
***Folder Description*** :point_left:
------------------------------------------
~~~

./Dataset             --> Contains the dataset related files.
./BERT_Classifier     --> Contains the codes for BERT classifiers performing binary classifier on the dataset
./CNN_GRU			  --> Contains the codes for CNN-GRU model		
./LASER+LR 	      --> Containes the codes for Logistic regression classifier used on top of LASER embeddings

~~~

## Requirements 

Make sure to use **Python3** when running the scripts. The package requirements can be obtained by running `pip install -r requirements.txt`.


------------------------------------------
***Instructions for training the classifier models***
------------------------------------------

1. **mBERT Baseline**
	This setting consists of using multilingual bert model with the same language dataset for training and testing. Refer to `BERT Classifier` folder for the codes and usage instructions.

2. **mBERT All_but_one**
	This setting consists of using multilingual bert model with training dataset from multiple languages and validation and test from a single target language. Refer to `BERT Classifier` folder for the codes and usage instructions.

3. **Translation + BERT Baseline**
	This setting consists of translating the other language datasets to english and finetuning the bert-base model using this translated datasets. Refer to `BERT Classifier` folder for the codes and usage instructions.

4. **CNN+GRU Baseline**
	This setting consists of using MUSE word embeddings along with a CNN-GRU based model, and training and testing on the same language. Refer to `CNN_GRU` folder for the codes and usage instructions.
	
5. **LASER+LR baseline**
	This setting consists of training a logistic regression model on the LASER embeddings of the dataset. The training and testing dataset are from the same language. Refer to `LASER+LR` folder for the codes and usage instructions.
 
6. **LASER+LR all_but_one**
	This setting consists of training a logistic regression model on the LASER embeddings of the dataset. The dataset from other languages are also used to train the LR model. Refer to `LASER+LR` folder for the codes and usage instructions.
	
	
### Blogs and github repos which we used for reference :angel:
1. Muse embeddding are downloaded and extracted using the code from [MUSE github repository](https://github.com/facebookresearch/MUSE)
2. For finetuning BERT this [blog](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)  by Chris McCormick is used and we also referred [Transformers github repo](https://github.com/huggingface/transformers)
3. For CNN-GRU model we used the original [repo](https://github.com/ziqizhang/chase) for reference 
4. For generating the LASER embeddings of the dataset, we used the code from [LASER github repository]()

### Todos
- [ ] Upload the our models to [transformers community](https://huggingface.co/models) to make them public
- [ ] Add arxiv paper link.
- [ ] Create an interface for **social scientists** where they can use our models easily with their data
- [ ] Create a pull request to add the models to official [transformers repo](https://github.com/huggingface/transformers)


#####  :thumbsup: The repo is still in active developements. Feel free to create an issue !!  :thumbsup:
