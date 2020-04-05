import transformers 
import torch
import neptune
from knockknock import slack_sender
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
from BERT_inference import *
import os

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


neptune.init(project_name,api_token=api_token,proxies=proxies)
neptune.set_project(project_name)

print("current gpu device", torch.cuda.current_device())
torch.cuda.set_device(0)
print("current gpu device",torch.cuda.current_device())
	   





webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="#model_messages")
def train_model(params,best_val_fscore):
	
	if(params['language']=='English'):
		params['csv_file']='*_full.csv'
	
	train_path=params['files']+'/train/'+params['csv_file']
	val_path=params['files']+'/val/'+params['csv_file']


	train_files=glob.glob(train_path)
	val_files=glob.glob(val_path)
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
	df_train=data_collector(train_files,params,True)
	df_val=data_collector(val_files,params,False)
	
	if(params['csv_file']=='*_full.csv'):
		sentences_train = df_train.text.values
		sentences_val = df_val.text.values
	elif(params['csv_file']=='*_translated.csv'):
		sentences_train = df_train.translated.values
		sentences_val = df_val.translated.values
	
	labels_train = df_train.label.values
	labels_val = df_val.label.values
	label_counts=df_train['label'].value_counts()
	print(label_counts)
	label_weights = [ (len(df_train))/label_counts[0],len(df_train)/label_counts[1] ]
	print(label_weights)
	
	model=select_model(params['what_bert'],params['path_files'],params['weights'])
	# Tell pytorch to run this model on the GPU.
	model.cuda()
	input_train_ids,att_masks_train=combine_features(sentences_train,tokenizer,params['max_length'])


	input_val_ids,att_masks_val=combine_features(sentences_val,tokenizer,params['max_length'])
	train_dataloader = return_dataloader(input_train_ids,labels_train,att_masks_train,batch_size=params['batch_size'],is_train=params['is_train'])
	validation_dataloader=return_dataloader(input_val_ids,labels_val,att_masks_val,batch_size=params['batch_size'],is_train=False)
	
	optimizer = AdamW(model.parameters(),
				  lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
				  eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
				)
	
	# Number of training epochs (authors recommend between 2 and 4)
	# Total number of training steps is number of batches * number of epochs.
	total_steps = len(train_dataloader) * params['epochs']

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, 
												num_warmup_steps = int(total_steps/10), # Default value in run_glue.py
												num_training_steps = total_steps)

	# Set the seed value all over the place to make this reproducible.
	fix_the_random(seed_val = params['random_seed'])
	# Store the average loss after each epoch so we can plot them.
	loss_values = []

	bert_model = params['path_files']
	language  = params['language']
	name_one=bert_model+"_"+language
	if(params['logging']=='neptune'):
		neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
		neptune.append_tag(bert_model)
		neptune.append_tag(language)
		
	# For each epo=ch...
	best_val_fscore=best_val_fscore

	for epoch_i in range(0, params['epochs']):
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
		print('Training...')

		# Measure how long the training epoch takes.
		t0 = time.time()

		# Reset the total loss for this epoch.
		total_loss = 0
		model.train()

		# For each batch of training data...
		for step, batch in tqdm(enumerate(train_dataloader)):

			# Progress update every 40 batches.
			if step % 40 == 0 and not step == 0:
				# Calculate elapsed time in minutes.
				elapsed = format_time(time.time() - t0)
			# `batch` contains three pytorch tensors:
			#   [0]: input ids 
			#   [1]: attention masks
			#   [2]: labels 
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)
			# (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
			model.zero_grad()        

			outputs = model(b_input_ids, 
						token_type_ids=None, 
						attention_mask=b_input_mask, 
						labels=b_labels)

			# The call to `model` always returns a tuple, so we need to pull the 
			# loss value out of the tuple.
			loss = outputs[0]
			if(params['logging']=='neptune'):
				neptune.log_metric('batch_loss',loss)
			# Accumulate the training loss over all of the batches so that we can
			# calculate the average loss at the end. `loss` is a Tensor containing a
			# single value; the `.item()` function just returns the Python value 
			# from the tensor.
			total_loss += loss.item()

			# Perform a backward pass to calculate the gradients.
			loss.backward()

			# Clip the norm of the gradients to 1.0.
			# This is to help prevent the "exploding gradients" problem.
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			# Update parameters and take a step using the computed gradient.
			# The optimizer dictates the "update rule"--how the parameters are
			# modified based on their gradients, the learning rate, etc.
			optimizer.step()
			# Update the learning rate.
			scheduler.step()
		# Calculate the average loss over the training data.
		avg_train_loss = total_loss / len(train_dataloader)
		if(params['logging']=='neptune'):
			neptune.log_metric('avg_train_loss',avg_train_loss)
		

		# Store the loss value for plotting the learning curve.
		loss_values.append(avg_train_loss)
		val_fscore,val_accuracy=Eval_phase(params,'val',model)		
		test_fscore,test_accuracy=Eval_phase(params,'test',model)

		#Report the final accuracy for this validation run.
		if(params['logging']=='neptune'):	
			neptune.log_metric('val_fscore',val_fscore)
			neptune.log_metric('val_acc',val_accuracy)
			# neptune.log_metric('test_fscore',test_fscore)
			# neptune.log_metric('test_accuracy',test_accuracy)
		if(val_fscore > best_val_fscore):
			print(val_fscore,best_val_fscore)
			best_val_fscore=val_fscore

			save_model(model,tokenizer,params)
#   		


	  
#     print("Training complete!")
	if(params['save_only_bert']==False):
		fscore,accuracy=Eval_phase(params,'test',model)

		
	if(params['logging']=='neptune'):
		neptune.log_metric('test_fscore',fscore)
		neptune.log_metric('test_accuracy',accuracy)
		
		# neptune.log_metric('val_acc',accuracy_score(true_labels,pred_labels))
		neptune.stop()
	del model
	torch.cuda.empty_cache()
	return fscore,best_val_fscore



# 'logging':where logging {'local','neptune'}
# 'language': language {'English','Polish','Portugese','German','Indonesian','Italian','Arabic'}
# 'is_train': whether train dataset 
# 'is_model':is model 
# 'learning_rate':Adam parameter lr
# 'epsilon': Adam parameter episilon
# 'path_files':bert path from the bert model should be loaded,
# 'sample_ratio':ratio of the training data to take
# 'how_train':how the bert is trained possible option {'all','baseline','all_but_one'}
# 'epochs': number of epochs to train bert
# 'batch_size': batch size
# 'to_save': whether to save the model or not
# 'weights': weights for binary classifier
# 'what_bert': type of bert possible option {'normal','weighted'}
# 'save_only_bert': if only bert (without classifier) should be used 
# 'files': parent folder,
# 'csv_file':pattern of the file,
# 'samp_strategy':'how to select the samples if less data used for training,





params={
	'logging':'neptune',
	'language':'English',
	'is_train':True,
	'is_model':True,
	'learning_rate':2e-5,
	'files':'only_hate',
	'csv_file':'*_translated.csv',
	'samp_strategy':'stratified',
	'epsilon':1e-8,
	'path_files':'multilingual_bert',
	'take_ratio':False,
	'sample_ratio':16,
	'how_train':'baseline',
	'epochs':5,
	'batch_size':16,
	'to_save':True,
	'weights':[1.0,1.0],
	'what_bert':'normal',
	'save_only_bert':False,
	'max_length':128,
	'columns_to_consider':['directness','target','group'],
	'random_seed':42

}



if __name__=='__main__':

	lang_map={'Arabic':'ar','French':'fr','Portugese':'pt','Spanish':'es','English':'en','Indonesian':'id','Italian':'it','German':'de','Polish':'pl'}
	# torch.cuda.set_device(0)

	lang_list=list(lang_map.keys())
	for lang in lang_list[0:3]:
		params['language']=lang
		for sample_ratio,take_ratio in [(16,False),(32,False),(64,False),(128,False),(256,False)]:
			count=0
			params['take_ratio']=take_ratio
			params['sample_ratio']=sample_ratio
			best_val_fscore=0
			for lr in [2e-5,3e-5,5e-5]:
				params['learning_rate']=lr
				for ss in ['stratified','equal']:
					params['samp_strategy']=ss
					for seed in [2018,2019,2020,2021,2022]:
						params['random_seed']=seed
						count+=1
						_,best_val_fscore=train_model(params,best_val_fscore)


		best_val_fscore=00
		for lr in [2e-5,3e-5,5e-5]:
			params['learning_rate']=lr
			params['samp_strategy']='stratified'
			for seed in [2018,2019,2020,2021,2022][:1]:
				params['random_seed']=seed
				_,best_val_fscore=train_model(params,best_val_fscore)
		print('============================')
		print('Model for Language',lang,'is trained')
		print('============================')
		