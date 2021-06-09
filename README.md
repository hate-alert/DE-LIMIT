[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhate-alert%2FDE-LIMIT&count_bg=%2379C83D&title_bg=%23555555&icon=adblock.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/punyajoy/DE-LIMIT/issues)
# Deep Learning Models for Multilingual Hate Speech Detection
:portugal: :saudi_arabia: :poland: :indonesia: :it: Solving the problem of **hate speech detection** in **9 languages** across **16 datasets**.
:fr: :us: :es: :de:

### New update -- :tada: :tada: all our BERT models are available [here](https://huggingface.co/Hate-speech-CNERG). Be sure to check it out :tada: :tada:.

### Demo 

Please look [here](Example/N_Class_model.ipynb) to check model loading and inference.


***Please cite our paper in any published work that uses any of these resources.***

~~~bibtex
@inproceedings{aluru2021deep,
  title={A Deep Dive into Multilingual Hate Speech Classification},
  author={Aluru, Sai Saketh and Mathew, Binny and Saha, Punyajoy and Mukherjee, Animesh},
  booktitle={Machine Learning and Knowledge Discovery in Databases. Applied Data Science and Demo Track: European Conference, ECML PKDD 2020, Ghent, Belgium, September 14--18, 2020, Proceedings, Part V},
  pages={423--439},
  year={2021},
  organization={Springer International Publishing}
}
~~~

------------------------------------------
***Folder Description*** :point_left:
------------------------------------------
~~~

./Dataset             --> Contains the dataset related files.
./BERT_Classifier     --> Contains the codes for BERT classifiers performing binary classifier on the dataset
./CNN_GRU	      --> Contains the codes for CNN-GRU model		
./LASER+LR 	      --> Containes the codes for Logistic regression classifier used on top of LASER embeddings

~~~

## Requirements 

Make sure to use **Python3** when running the scripts. The package requirements can be obtained by running `pip install -r requirements.txt`.

------------------------------------------
***Dataset***
------------------------------------------
Check out the `Dataset ` folder to know more about how we curated the dataset for different languages.  :warning: There are few datasets which requires crawling them hence we can gurantee the retrieval of all the datapoints as tweets may get deleted.
:warning:

-----------------------------------------
***Models used for our this task***
------------------------------------------
We release the code for train/finetuning the following models along with their hyperparamters.

:1st_place_medal: `best for high resource language` , :medal_sports: `best for low resource language`

:airplane: `fastest to train`  , :small_airplane: `slowest to train`

1. **mBERT Baseline:**
	This setting consists of using multilingual bert model with the same language dataset for training and testing. Refer to `BERT Classifier` folder for the codes and usage instructions.

2. **mBERT All_but_one::1st_place_medal::small_airplane:** 
	This setting consists of using multilingual bert model with training dataset from multiple languages and validation and test from a single target language. Refer to `BERT Classifier` folder for the codes and usage instructions.

3. **Translation + BERT Baseline:**
	This setting consists of translating the other language datasets to english and finetuning the bert-base model using this translated datasets. Refer to `BERT Classifier` folder for the codes and usage instructions.

4. **CNN+GRU Baseline:**
	This setting consists of using MUSE word embeddings along with a CNN-GRU based model, and training and testing on the same language. Refer to `CNN_GRU` folder for the codes and usage instructions.
	
5. **LASER+LR baseline::airplane:**
	This setting consists of training a logistic regression model on the LASER embeddings of the dataset. The training and testing dataset are from the same language. Refer to `LASER+LR` folder for the codes and usage instructions.
 
6. **LASER+LR all_but_one::medal_sports:**
	This setting consists of training a logistic regression model on the LASER embeddings of the dataset. The dataset from other languages are also used to train the LR model. Refer to `LASER+LR` folder for the codes and usage instructions.


	
### Blogs and github repos which we used for reference :angel:
1. Muse embeddding are downloaded and extracted using the code from [MUSE github repository](https://github.com/facebookresearch/MUSE)
2. For finetuning BERT this [blog](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)  by Chris McCormick is used and we also referred [Transformers github repo](https://github.com/huggingface/transformers)
3. For CNN-GRU model we used the original [repo](https://github.com/ziqizhang/chase) for reference 
4. For generating the LASER embeddings of the dataset, we used the code from [LASER github repository](https://github.com/facebookresearch/LASER)

### For more details about our paper

Sai Saketh Aluru, Binny Mathew, Punyajoy Saha and Animesh Mukherjee. 2020. "[Deep Learning Models for Multilingual Hate Speech Detection](https://arxiv.org/abs/2004.06465)". ECML-PKDD



### Todos
- [x] Upload our models to [transformers community](https://huggingface.co/models) to make them public
- [x] Add arxiv paper link and description
- [ ] Create an interface for **social scientists** where they can use our models easily with their data
- [ ] Create a pull request to add the models to official [transformers repo](https://github.com/huggingface/transformers)


#####  :thumbsup: The repo is still in active developements. Feel free to create an [issue](https://github.com/punyajoy/DE-LIMIT/issues) !!  :thumbsup:
