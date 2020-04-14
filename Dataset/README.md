# Dataset Description
The datasets used are as follows:
1. Arabic:
	1a. [Mulki et al.](https://github.com/Hala-Mulki/L-HSAB-First-Arabic-Levantine-HateSpeech-Dataset)
	1b. [Ousidhoum et al.](https://github.com/HKUST-KnowComp/MLMA_hate_speech)
2. English:
	2a. [Davidson et al.](https://github.com/t-davidson/hate-speech-and-offensive-language)
	2b. [Gilbert et al.](https://github.com/aitor-garcia-p/hate-speech-dataset)
	2c. [Waseem et al.](https://github.com/zeerakw/hatespeech)
	2d. [Basile et al.](https://github.com/msang/hateval)
	2e. [Ousidhoum et al.](https://github.com/HKUST-KnowComp/MLMA_hate_speech)
	2f. [Founta et al.](https://github.com/ENCASEH2020/hatespeech-twitter)
3. German:
	3a. [Ross et al.](https://github.com/UCSM-DUE/IWG_hatespeech_public)
	3b. [Bretschneider et al.](http://www.ub-web.de/research/)
4. Indonesian:
	4a. [Ibrohim et al.](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection)
	4b. [Alfina et al.](https://github.com/ialfina/id-hatespeech-detection)
5. Italian: 
	5a. [Sanguinetti et al.](https://github.com/msang/hate-speech-corpus)
	5b. [Bosco et al.](https://github.com/msang/haspeede2018)
6. Polish: 
	6a. [Ptaszynski et al.](http://poleval.pl/tasks/task6)
7. Portugese:
	7a. [Fortuna et al.](https://github.com/paulafortuna/Portuguese-Hate-Speech-Dataset)
8. Spanish:
	8a. [Basile et al.](https://github.com/msang/hateval)
	8b. [Pereira et al.](https://zenodo.org/record/2592149)
9. French:
	9a. [Ousidhoum et al.](https://github.com/HKUST-KnowComp/MLMA_hate_speech)

In cases where the actual text is not given by the source and only tweet ids and labels are given, use any twitter scraping tools to extract the texts.
In the above datasets, some of them contain multiple labels for the text such as hate-speech, abusive, offensive, etc. In such cases, only the text with either hate-speech and normal labels are used and others are discarded. 


## Instructions for getting the datasets
1. Download the datasets from the above sources and place it in the subfolder `Dataset/full_data`
2. Use the `Translation.ipynb ` to translate the datasets into english
3. Use the ids given in `ID Mapping` folder for splitting the datasets into train, val and test splits. Use the file `Stratified Split.ipynb` for doing the splits. 