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
torch.cuda.set_device(1)
print("current gpu device",torch.cuda.current_device())
	   



webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="#model_messages")
def Eval_phase(params,which_files='test',model=None):
	if(which_files=='test'):
		test_files=glob.glob('full_data/Test/*.csv')
	if(which_files=='train'):
		test_files=glob.glob('full_data/Train/*.csv')
	if(which_files=='val'):
		test_files=glob.glob('full_data/Val/*.csv')
	
	'''Testing phase of the model'''
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)

	if(params['is_model']==True):
		model.eval()
	else:
			
		model = SC_weighted_BERT.from_pretrained(
			params['path_files'], # Use the 12-layer BERT model, with an uncased vocab.
			num_labels = 2, # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.   
			output_attentions = False, # Whether the model returns attentions weights.
			output_hidden_states = False, # Whether the model returns all hidden-states.
			weights=[0.5,0.5]
		)
		model.cuda()
		model.eval()

		
	df_test=data_collector(test_files,params['language'],is_train=False,sample_ratio=params['sample_ratio'],type_train=params['how_train'])
	sentences_test = df_test.text.values
	labels_test = df_test.label.values
	input_test_ids,att_masks_test=combine_features(sentences_test,tokenizer)
	test_dataloader=return_dataloader(input_test_ids,labels_test,att_masks_test,batch_size=params['batch_size'],is_train=False)
	print("Running Test...")
	t0 = time.time()

	# Put the model in evaluation mode--the dropout layers behave differently
	# during evaluation.
	# Tracking variables 
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	true_labels=[]
	pred_labels=[]
	# Evaluate data for one epoch
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

	testf1=f1_score(true_labels, pred_labels, average='macro')
	testacc=accuracy_score(true_labels,pred_labels)

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


webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="#model_messages")
def train_model(params):
	train_files=glob.glob('full_data/Train/*.csv')
	val_files=glob.glob('full_data/Val/*.csv')

	# Load the BERT tokenizer.
	# Load BertForSequenceClassification, the pretrained BERT model with a single 
	# linear classification layer on top. 
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
	# model = BertForSequenceClassification.from_pretrained(
	# 	params['path_files'], # Use the 12-layer BERT model, with an uncased vocab.
	# 	num_labels = 2, # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.   
	# 	output_attentions = False, # Whether the model returns attentions weights.
	# 	output_hidden_states = False, # Whether the model returns all hidden-states.
	# )
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
	model = SC_weighted_BERT.from_pretrained(
		params['path_files'], # Use the 12-layer BERT model, with an uncased vocab.
		num_labels = 2, # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.   
		output_attentions = False, # Whether the model returns attentions weights.
		output_hidden_states = False, # Whether the model returns all hidden-states.
		weights=params['weights']
	)
	# model = BertForSequenceClassification.from_pretrained(
	# 	params['path_files'], # Use the 12-layer BERT model, with an uncased vocab.
	# 	num_labels = 2, # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.   
	# 	output_attentions = False, # Whether the model returns attentions weights.
	# 	output_hidden_states = False, # Whether the model returns all hidden-states.
	# )
	
	# Tell pytorch to run this model on the GPU.
	model.cuda()
	input_train_ids,att_masks_train=combine_features(sentences_train,tokenizer)


	input_val_ids,att_masks_val=combine_features(sentences_val,tokenizer)
	train_dataloader = return_dataloader(input_train_ids,labels_train,att_masks_train,batch_size=params['batch_size'],is_train=params['is_train'])
	validation_dataloader=return_dataloader(input_val_ids,labels_val,att_masks_val,batch_size=params['batch_size'],is_train=False)
	
	optimizer = AdamW(model.parameters(),
				  lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
				  eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
				)
	
	# Number of training epochs (authors recommend between 2 and 4)
	epochs = 5

	# Total number of training steps is number of batches * number of epochs.
	total_steps = len(train_dataloader) * epochs

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
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		print('Training...')

		# Measure how long the training epoch takes.
		t0 = time.time()

		# Reset the total loss for this epoch.
		total_loss = 0

		# Put the model into training mode. Don't be mislead--the call to 
		# `train` just changes the *mode*, it doesn't *perform* the training.
		# `dropout` and `batchnorm` layers behave differently during training
		# vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
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
		fscore=Eval_phase(params,'val',model)		
		t0 = time.time()

		# Put the model in evaluation mode--the dropout layers behave differently
		# during evaluation.
		model.eval()

		# Tracking variables 
		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0
		true_labels=[]
		pred_labels=[]
		# Evaluate data for one epoch
		for batch in validation_dataloader:

			# Add batch to GPU
			batch = tuple(t.to(device) for t in batch)

			# Unpack the inputs from our dataloader
			b_input_ids, b_input_mask, b_labels = batch

			# Telling the model not to compute or store gradients, saving memory and
			# speeding up validation
			with torch.no_grad():        
				# Forward pass, calculate logit predictions.
				# This will return the logits rather than the loss because we have
				# not provided labels.
				# token_type_ids is the same as the "segment ids", which 
				# differentiates sentence 1 and 2 in 2-sentence tasks.
				# The documentation for this `model` function is here: 
				# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
				outputs = model(b_input_ids, 
								token_type_ids=None, 
								attention_mask=b_input_mask)

			# Get the "logits" output by the model. The "logits" are the output
			# values prior to applying an activation function like the softmax.
			logits = outputs[0]

			# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()

			# Calculate the accuracy for this batch of test sentences.
			tmp_eval_accuracy = flat_accuracy(logits, label_ids)
			pred_labels+=list(np.argmax(logits, axis=1).flatten())
			true_labels+=list(label_ids.flatten())
			# Accumulate the total accuracy.
			eval_accuracy += tmp_eval_accuracy

			# Track the number of batches
			nb_eval_steps += 1
		
		#Report the final accuracy for this validation run.
		if(params['logging']!='neptune'):
			print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
			print("  Fscore: {0:.2f}".format(f1_score(true_labels, pred_labels, average='macro')))
			print("  Validation took: {:}".format(format_time(time.time() - t0)))
		else:
			neptune.log_metric('val_fscore',f1_score(true_labels, pred_labels, average='macro'))
			neptune.log_metric('val_acc',accuracy_score(true_labels,pred_labels))
	if(params['to_save']==True):

		# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
		if(params['how_train']!='all'):
			output_dir = 'models_saved/'+params['path_files'][:-1]+'_'+params['language']+'_'+params['how_train']+'_'+str(params['sample_ratio'])+'/'
		else:
			output_dir = 'models_saved/'+params['path_files'][:-1]+'_'+params['how_train']+'_'+str(params['sample_ratio'])+'/'
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
	fscore=Eval_phase(params,model)
	if(params['logging']=='neptune'):
		neptune.log_metric('test_fscore',fscore)
		# neptune.log_metric('val_acc',accuracy_score(true_labels,pred_labels))
		neptune.stop()
	return fscore





params={
	'logging':'local',
	'language':'English',
	'is_train':True,
	'is_model':True,
	'learning_rate':2e-5,
	'epsilon':1e-8,
	'path_files':'multilingual_bert/',
	'sample_ratio':100,
	'how_train':'all',
	'epochs':5,
	'batch_size':8,
	'to_save':True,
	'weights':[1.0,1.0]
}

if __name__=='__main__':
	train_model(params)

