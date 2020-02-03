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

# # If there's a GPU available...
# if torch.cuda.is_available():    
#     # Tell PyTorch to use the GPU.    
#     device = torch.device("cuda")
#     print('There are %d GPU(s) available.' % torch.cuda.device_count())
#     print('We will use the GPU:', torch.cuda.get_device_name(0))
# # If not...
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")


# neptune.init(project_name,api_token=api_token,proxies=proxies)
# neptune.set_project(project_name)

# print("current gpu device", torch.cuda.current_device())
# torch.cuda.set_device(0)
# print("current gpu device",torch.cuda.current_device())
	   





webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="#model_messages")
def train_model(params):
	train_files=glob.glob('full_data/Train/*.csv')
	val_files=glob.glob('full_data/Val/*.csv')

	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
	df_train=data_collector(train_files,params['language'],is_train=True,sample_ratio=params['sample_ratio'],type_train=params['how_train'])
	df_val=data_collector(val_files,params['language'],is_train=False,sample_ratio=100,type_train=params['how_train'])
	
	
	sentences_train = df_train.text.values
	labels_train = df_train.label.values
	sentences_val = df_val.text.values
	labels_val = df_val.label.values
	#print(df_train['label'].value_counts())
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
												num_warmup_steps = 0, # Default value in run_glue.py
												num_training_steps = total_steps)

	# Set the seed value all over the place to make this reproducible.
	fix_the_random(seed_val = 42)
	# Store the average loss after each epoch so we can plot them.
	loss_values = []

	bert_model = params['path_files'][:-1]
	langauge  = params['language']
	name_one=bert_model+"_"+langauge
	if(params['logging']=='neptune'):
		neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
		neptune.append_tag(bert_model)
		neptune.append_tag(langauge)
		
	# For each epoch...
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
		fscore,accuracy=Eval_phase(params,'val',model)		
		
		#Report the final accuracy for this validation run.
		if(params['logging']=='neptune'):	
			neptune.log_metric('val_fscore',fscore)
			neptune.log_metric('val_acc',accuracy)
	


	if(params['to_save']==True):
		
		# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
		if(params['how_train']!='all'):
			output_dir = 'models_saved/'+params['path_files'][:-1]+'_'+params['language']+'_'+params['how_train']+'_'+str(params['sample_ratio'])
		else:
			output_dir = 'models_saved/'+params['path_files'][:-1]+'_'+params['how_train']+'_'+str(params['sample_ratio'])
		

		if(params['save_only_bert']):
			model=model.bert
			output_dir=output_dir+'_only_bert/'
		else:
			output_dir=output_dir+'/'
		print(output_dir)
		# Create output directory if needed
		if not os.path.exists(output_dir):
		    os.makedirs(output_dir)

		print("Saving model to %s" % output_dir)

		# Save a trained model, configuration and tokenizer using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		
		model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
		model_to_save.save_pretrained(output_dir)
		tokenizer.save_pretrained(output_dir)

#     print("")
#     print("Training complete!")
	if(params['save_only_bert']==False):
		fscore=Eval_phase(params,'test',model)

		
	if(params['logging']=='neptune'):
		neptune.log_metric('test_fscore',fscore)
		# neptune.log_metric('val_acc',accuracy_score(true_labels,pred_labels))
		neptune.stop()
	return fscore






webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="#model_messages")
def train_multitask_model(params):
	#train_files=glob.glob('hate_speech_mlma/*.csv')
	train_files=['To_Punyajoy_MTL.csv']
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
	df_train=data_collector(train_files,'English',is_train=True,sample_ratio=100,type_train=params['how_train'])
	#df_train=df_train.drop(['sentiment','annotator_sentiment'],axis=1)
	df_train=MultiColumnLabelEncoder(columns = params['columns_to_consider']).fit_transform(df_train)

	list_unique=[]

	for column in params['columns_to_consider']:
	    list_unique.append((df_train[column].nunique()))
	print(list_unique)

	labels_train=df_train[params['columns_to_consider']].values
	sentences_train=df_train.text.values

	model=select_model(params['what_bert'],params['path_files'],params['weights'],list_unique)
	# Tell pytorch to run this model on the GPU.
	model.cuda()


	input_train_ids,att_masks_train=combine_features(sentences_train,tokenizer,params['max_length'])
	train_dataloader = return_dataloader(input_train_ids,labels_train,att_masks_train,batch_size=params['batch_size'],is_train=params['is_train'])




	optimizer = AdamW(model.parameters(),
				  lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
				  eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
				)
	
	# Number of training epochs (authors recommend between 2 and 4)
	# Total number of training steps is number of batches * number of epochs.
	total_steps = len(train_dataloader) * params['epochs']

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, 
												num_warmup_steps = 0, # Default value in run_glue.py
												num_training_steps = total_steps)

	# Set the seed value all over the place to make this reproducible.
	fix_the_random(seed_val = 42)
	# Store the average loss after each epoch so we can plot them.
	loss_values = []

	bert_model = params['path_files'][:-1]
	langauge  = params['language']
	name_one=bert_model+"_"+langauge
	if(params['logging']=='neptune'):
		neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
		neptune.append_tag(bert_model)
		neptune.append_tag(langauge)
		
	# For each epoch...
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
			# Accumulate the modeltraining loss over all of the batches so that we can
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
		else:
			print('avg_train_loss',avg_train_loss)

		loss_values.append(avg_train_loss)
		

	if(params['to_save']==True):
		
		# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
		if(params['how_train']!='all'):
			output_dir = 'models_saved/'+params['path_files'][:-1]+'_'+params['language']+'_'+params['how_train']+'_'+str(params['sample_ratio'])
		else:
			output_dir = 'models_saved/'+params['path_files'][:-1]+'_'+params['how_train']+'_'+str(params['sample_ratio'])
		

		if(params['save_only_bert']):
			model=model.bert
			output_dir=output_dir+'_only_bert/'
		else:
			output_dir=output_dir+'/'
		print(output_dir)
		# Create output directory if needed
		if not os.path.exists(output_dir):
		    os.makedirs(output_dir)

		print("Saving model to %s" % output_dir)

		# Save a trained model, configuration and tokenizer using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		
		model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
		model_to_save.save_pretrained(output_dir)
		tokenizer.save_pretrained(output_dir)

#     print("")
#     print("Training complete!")
		
	if(params['logging']=='neptune'):
		# neptune.log_metric('val_acc',accuracy_score(true_labels,pred_labels))
		neptune.stop()














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





# params={
# 	'logging':'local',
# 	'language':'Arabic',
# 	'is_train':True,
# 	'is_model':True,
# 	'learning_rate':2e-5,
# 	'epsilon':1e-8,
# 	'path_files':'models_saved/multilingual_bert_English_all_multitask_0.1_only_bert/',
# 	'sample_ratio':100,
# 	'how_train':'baseline',
# 	'epochs':5,
# 	'batch_size':8,
# 	'to_save':True,
# 	'weights':[1.0,1.0],
# 	'what_bert':'weighted',
# 	'save_only_bert':False,
# 	'columns_to_consider':['directness','target','group']
# }
#### multitask 
params={
	'logging':'local',
	'language':'English',
	'is_train':True,
	'is_model':True,
	'learning_rate':2e-5,
	'epsilon':1e-8,
	'path_files':'multilingual_bert/',
	'sample_ratio':100,
	'how_train':'all_multitask_own',
	'epochs':5,
	'batch_size':64,
	'to_save':True,
	'weights':[1.0,1.0],
	'what_bert':'multitask',
	'save_only_bert':True,
	'max_length':128,
	'columns_to_consider':['label','is_about_class','is_about_disability','is_about_ethnicity','is_about_gender','is_about_nationality','is_about_religion','is_about_sexual_orientation']
}



if __name__=='__main__':
	train_multitask_model(params)
	#train_model(params)
