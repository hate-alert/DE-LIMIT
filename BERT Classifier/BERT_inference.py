# Necessary imports
import transformers 
import torch
import neptune

from api_config import project_name,proxies,api_token
import glob 
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
import random
from transformers import BertTokenizer
from bert_codes.feature_generation import combine_features,return_dataloader
from bert_codes.data_extractor import data_collector
from bert_codes.own_bert_models import *
from bert_codes.utils import *
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
import os

# If gpu is available
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Initialize neptune for logging
neptune.init(project_name,api_token=api_token,proxies=proxies)
neptune.set_project(project_name)


# The function for evaluating
# Params - see below for description
# which_files - what files to test on - {'train','val','test'}
# model - the model to use if passed. If model==None, the model is loaded based on the params passed.
def Eval_phase(params,which_files='test',model=None):

	# For english, there is no translation, hence use full dataset.
	if(params['language']=='English'):
		params['csv_file']='*_full.csv'
	
	# Load the files to test on
	if(which_files=='train'):
		path=params['files']+'/train/'+params['csv_file']
		test_files=glob.glob(path)
	if(which_files=='val'):
		path=params['files']+'/val/'+params['csv_file']
		test_files=glob.glob(path)
	if(which_files=='test'):
		path=params['files']+'/test/'+params['csv_file']
		test_files=glob.glob(path)
	
	'''Testing phase of the model'''
	print('Loading BERT tokenizer...')
	# Load bert tokenizer
	tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)

	# If model is passed, then use the given model. Else load the model from the saved location
	# Put the model in evaluation mode--the dropout layers behave differently
	# during evaluation.
	if(params['is_model']==True):
		print("model previously passed")
		model.eval()
	else:
		model=select_model(params['what_bert'],params['path_files'],params['weights'])
		model.cuda()
		model.eval()

	# Load the dataset
	df_test=data_collector(test_files,params,False)
	if(params['csv_file']=='*_translated.csv'):
		sentences_test = df_test.translated.values
	elif(params['csv_file']=='*_full.csv'):
		sentences_test = df_test.text.values
		

	labels_test = df_test.label.values
	# Encode the dataset using the tokenizer
	input_test_ids,att_masks_test=combine_features(sentences_test,tokenizer,params['max_length'])
	test_dataloader=return_dataloader(input_test_ids,labels_test,att_masks_test,batch_size=params['batch_size'],is_train=False)
	print("Running eval on ",which_files,"...")
	t0 = time.time()

	# Tracking variables 
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	true_labels=[]
	pred_labels=[]
	for batch in test_dataloader:
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		# Unpack the inputs from our dataloader
		b_input_ids, b_input_mask, b_labels = batch
		# Telling the model not to compute or store gradients, saving memory and
		# speeding up validation
		with torch.no_grad():        
			outputs = model(b_input_ids, 
							token_type_ids=None, 
							attention_mask=b_input_mask)

		logits = outputs[0]
		# Move logits and labels to CPU
		logits = logits.detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()
		# Calculate the accuracy for this batch of test sentences.
		tmp_eval_accuracy = flat_accuracy(logits, label_ids)
		# Accumulate the total accuracy.
		eval_accuracy += tmp_eval_accuracy
		
		pred_labels+=list(np.argmax(logits, axis=1).flatten())
		true_labels+=list(label_ids.flatten())

		# Track the number of batches
		nb_eval_steps += 1

	# Get the accuracy and macro f1 scores
	testf1=f1_score(true_labels, pred_labels, average='macro')
	testacc=accuracy_score(true_labels,pred_labels)

	# Log the metrics obtained
	if(params['logging']!='neptune' or params['is_model'] == True):
		# Report the final accuracy for this validation run.
		print(" Accuracy: {0:.2f}".format(testacc))
		print(" Fscore: {0:.2f}".format(testf1))
		print(" Test took: {:}".format(format_time(time.time() - t0)))
	else:
		bert_model = params['path_files'][:-1]
		language  = params['language']
		name_one=bert_model+"_"+language
		neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
		neptune.append_tag(bert_model)
		neptune.append_tag(language)
		neptune.append_tag('test')
		neptune.log_metric('test_f1score',testf1)
		neptune.log_metric('test_accuracy',testacc)
		neptune.stop()
	
	return testf1,testacc

# Params used here 
params={
	'logging':'locals',
	'language':'German',
	'is_train':False,
	'is_model':False,
	'learning_rate':2e-5,
	'epsilon':1e-8,
	'path_files':'models_saved/multilingual_bert_English_baseline_100/',
	'sample_ratio':0.1,
	'how_train':'baseline',
	'epochs':5,
	'batch_size':16,
	'to_save':False,
	'weights':[1.0,1.0],
	'what_bert':'weighted',
	'save_only_bert':True
}




if __name__=='__main__':
	for lang in ['English','Polish','Portugese','German','Indonesian','Italian','Arabic']:
			params['language']=lang
			Eval_phase(params,'test')

