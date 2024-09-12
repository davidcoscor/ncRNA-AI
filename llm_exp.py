"""
title
"""

import pandas as pd
import ast
import subprocess

from src.llm_re import *
from utils.prompts import *


def parse_response(text:str):
	'''Expected format: `{"relation":int, "explanation":str}`'''
	try:
		# Get only the content inside {}
		json_content = text.split('{')[1].split('}')[0]
		# Replace bad characters
		json_string = json_content.replace("”",'"').replace("“",'"')
		json_string = json_string.replace('\n','').replace('\\','')
		# Get the part of the string that refers to each field
		fields = json_string.split(',',1)
		rel_string = fields[0]
		rel_val = rel_string.split(':')[1].replace('"','').replace("'","")
		exp_string = fields[1]
		exp_val = exp_string.split(':')[1].replace('"','').replace("'","")
		# Guarantee that "relation" and "explanation" fields are not mismatched
		if isinstance(ast.literal_eval(f'{rel_val}'), int):
			# if the value is int, then it refers to "relation"
			rel_string = f'"answer":{rel_val}'
			exp_string = f'"explanation":"{exp_val}"'
		else: # they're mismatched
			rel_string = f'"answer":"{exp_val}"'
			exp_string = f'"explanation":{rel_val}'
		# Merge everything
		json_string = '{'+rel_string+','+exp_string+'}'
		return ast.literal_eval(json_string)
	except:
		return text


def load_dataset(file, N:int=False, uncertains:bool=False, random:bool=False):
	data = pd.read_csv(file, sep='\t')
	if random: 
		data = data.sample(N, random_state=1)
		
	queries, true_labels, metadatas = [], [], []
	for _,row in data.iterrows():
		_id, sentence, e1, e1_id, e2, e2_id, true_label = row
		
		if not uncertains:
			true_label = int(true_label)
		else:
			true_label = 0 if true_label == 'U' else 1
		
		true_labels.append(true_label)
		queries.append({"sentence":sentence, "e1":e1, "e2":e2})
		metadatas.append((sentence, e1, e1_id, e2, e2_id))

	if N:
		return queries[:N], true_labels[:N], metadatas[:N]
	else:
		return queries, true_labels, metadatas


def control():
	### LOAD DATASET ###
	queries, true_labels, metadatas = load_dataset(
		file='./data/test.csv',
		N=False,
	)


	### EXAMPLES ###
	raw_examples = load_examples('./data/llm_examples.json')
	example_template = '###Example: Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}". ###Expected Response: {response}'
	examples = format_examples(raw_examples, example_template)


	### QUERY TEMPLATES ###
	query_template = 'Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}".'


	### INIT LLMS ###
	params = {
		'model': ['llama3','llama3:70b','phi3','phi3:14b','gemma','gemma2','gemma2:27b','mixtral','mixtral:8x22b'],
		'base_prompt': [prompt_2],
		'examples': [(n,examples) for n in range(2,11,2)],
		'query_template': [query_template],
		'response_parser': [parse_response],
		'options': [None],
		'timeout': [600]
	}
	llms = bulk_create(params)

	for k in range(5):
		### RUN ###
		all_preds = queue_evaluation(
			queries=queries, 
			true_labels=true_labels, 
			log_file=f'./outputs/chained/control_{k}.json', 
			classifiers=llms, 
			verbose=True
		)
		### SAVE PREDS ###
		for i,llm in enumerate(llms):
			llm_info = llm.__dict__
			model = llm_info['model']
			n_shots = llm_info['n_shots']
			save_preds(
				file=f'./outputs/chained/control_preds/{model}_{n_shots}_{k}.csv',
				classifier=llm,
				preds=all_preds[i],
				true_labels=true_labels,
				metadatas=metadatas
			)
		subprocess.run(['./push.sh'], capture_output=True)


def chained():
	### LOAD DATASET ###
	queries, true_labels, metadatas = load_dataset(
		file='./data/test.csv',
		N=False,
	)


	### EXAMPLES ###
	raw_examples = load_examples('./data/llm_examples.json')
	example_template = '###Example: Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}". ###Expected Response: {response}'
	examples = format_examples(raw_examples, example_template)

	raw_examples_u = load_examples('./data/llm_examples_u.json')
	example_template_u = '###Example: Do you think the following sentence conveys information about the relation between "{e1}" and "{e2}" in a clear way? Sentence: "{sentence}"\n ###Expected Response: {response}'
	examples_u = format_examples(raw_examples_u, example_template_u)
	

	### QUERY TEMPLATES ###
	query_template = 'Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}".'
	query_template_u = 'Do you think the following sentence conveys information about the relation between "{e1}" and "{e2}" in a clear way? Sentence: "{sentence}"'


	### INIT LLMS ###
	models = ['llama3','llama3:70b','phi3','phi3:14b','gemma','gemma2','gemma2:27b','mixtral','mixtral:8x22b']
	
	params_u = {
		'model': models,
		'base_prompt': [prompt_u],
		'examples': [(n, examples_u) for n in range(2,11,2)],
		'query_template': [query_template_u],
		'response_parser': [parse_response],
		'options': [None],
		'timeout': [600]
	}
	llms_u = bulk_create(params_u)
	
	params = {
		'model': models,
		'base_prompt': [prompt_2],
		'examples': [(n, examples) for n in range(2,11,2)],
		'query_template': [query_template],
		'response_parser': [parse_response],
		'options': [None],
		'timeout': [600]
	}
	llms = bulk_create(params)

	for k in range(5):
		### RUN 1 - FILTER UNCERTAIN ###
		all_preds_u = queue_evaluation(
			queries=queries, 
			true_labels=true_labels, 
			log_file=f'./outputs/chained/unc_{k}.json', 
			classifiers=llms_u, 
			verbose=True
		)
		
		# Filter out sentences predicted uncertain
		filtered_queries = [[] for _ in range(len(all_preds_u))]
		filtered_true_labels = [[] for _ in range(len(all_preds_u))]
		filtered_metadatas = [[] for _ in range(len(all_preds_u))]
		for i,llm_preds in enumerate(all_preds_u):
			assert len(llm_preds) == len(queries), 'Some preds are missing'
			for j,pred in enumerate(llm_preds):
				if isinstance(pred,dict) and pred['answer'] == 1: # answer valid and sentence not uncertain
					filtered_queries[i].append(queries[j])
					filtered_true_labels[i].append(true_labels[j])
					filtered_metadatas[i].append(metadatas[j])

		### RUN 2 - RE ###
		all_preds = queue_evaluation(
			queries=filtered_queries, 
			true_labels=filtered_true_labels, 
			log_file=f'./outputs/chained/chained_{k}.json', 
			classifiers=llms, 
			verbose=True
		)

		### SAVE PREDS ###
		for i,llm in enumerate(llms):
			llm_info = llm.__dict__
			model = llm_info['model']
			n_shots = llm_info['n_shots']
			save_preds(
				file=f'./outputs/chained/chained_preds/{model}_{n_shots}_{k}.csv',
				classifier=llm,
				preds=all_preds[i],
				true_labels=filtered_true_labels[i],
				metadatas=filtered_metadatas[i]
			)
		subprocess.run(['./push.sh'], capture_output=True)


def validation():
	### LOAD DATASET ###
	queries, true_labels, metadatas = load_dataset(
		file='./data/validation.csv',
		N=False,
	)


	### EXAMPLES ###
	raw_examples = load_examples('./data/llm_examples.json')
	example_template = '###Example: Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}". ###Expected Response: {response}'
	examples = format_examples(raw_examples, example_template)

	raw_examples_u = load_examples('./data/llm_examples_u.json')
	example_template_u = '###Example: Do you think the following sentence conveys information about the relation between "{e1}" and "{e2}" in a clear way? Sentence: "{sentence}"\n ###Expected Response: {response}'
	examples_u = format_examples(raw_examples_u, example_template_u)
	

	### QUERY TEMPLATES ###
	query_template = 'Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}".'
	query_template_u = 'Do you think the following sentence conveys information about the relation between "{e1}" and "{e2}" in a clear way? Sentence: "{sentence}"'


	### INIT LLMS ###
	models = ['mixtral']
	
	params_u = {
		'model': models,
		'base_prompt': [prompt_u],
		'examples': [(10, examples_u)],
		'query_template': [query_template_u],
		'response_parser': [parse_response],
		'options': [None],
		'timeout': [600]
	}
	llms_u = bulk_create(params_u)
	
	params = {
		'model': models,
		'base_prompt': [prompt_2],
		'examples': [(10, examples)],
		'query_template': [query_template],
		'response_parser': [parse_response],
		'options': [None],
		'timeout': [600]
	}
	llms = bulk_create(params)

	for k in range(5):
		### RUN 1 - FILTER UNCERTAIN ###
		all_preds_u = queue_evaluation(
			queries=queries, 
			true_labels=true_labels, 
			log_file=f'./outputs/chained/val_unc_{k}.json', 
			classifiers=llms_u, 
			verbose=True
		)
		
		# Filter out sentences predicted uncertain
		filtered_queries = [[] for _ in range(len(all_preds_u))]
		filtered_true_labels = [[] for _ in range(len(all_preds_u))]
		filtered_metadatas = [[] for _ in range(len(all_preds_u))]
		for i,llm_preds in enumerate(all_preds_u):
			assert len(llm_preds) == len(queries), 'Some preds are missing'
			for j,pred in enumerate(llm_preds):
				if isinstance(pred,dict) and pred['answer'] == 1: # answer valid and sentence not uncertain
					filtered_queries[i].append(queries[j])
					filtered_true_labels[i].append(true_labels[j])
					filtered_metadatas[i].append(metadatas[j])

		### RUN 2 - RE ###
		all_preds = queue_evaluation(
			queries=filtered_queries, 
			true_labels=filtered_true_labels, 
			log_file=f'./outputs/chained/val_chained_{k}.json', 
			classifiers=llms, 
			verbose=True
		)

		### SAVE PREDS ###
		for i,llm in enumerate(llms):
			llm_info = llm.__dict__
			model = llm_info['model']
			n_shots = llm_info['n_shots']
			save_preds(
				file=f'./outputs/chained/val_chained_preds/{model}_{n_shots}_{k}.csv',
				classifier=llm,
				preds=all_preds[i],
				true_labels=filtered_true_labels[i],
				metadatas=filtered_metadatas[i]
			)
		subprocess.run(['./push.sh'], capture_output=True)


def main():
	# control()
	# chained()
	validation()

if __name__ == '__main__':
	main()