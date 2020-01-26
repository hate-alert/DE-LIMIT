from BERT_training_inference import *



params={
	'logging':'locals',
	'language':'English',
	'is_train':False,
	'is_model':False,
	'learning_rate':5e-5,
	'epsilon':1e-8,
	'path_files':'models_saved/multilingual_bert_English_baseline_100/',
	'sample_ratio':100,
	'how_train':'baseline',
	'epochs':5,
	'batch_size':16,
	'to_save':False
}

if __name__=='__main__':
	for lang in ['English','Polish','Portugese','German','Indonesian','Italian','Arabic']:
			params['language']=lang
			Eval_phase(params)

