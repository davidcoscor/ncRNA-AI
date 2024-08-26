"""
Austism Spectrum Disorder Case Study supporting code 
David da Costa Correia @ FCUL & INSA
"""

### Settings ###
ARTICLES_FILE = './data/asd_articles.json'
OUTPUT_FOLDER = './outputs/asd/'
N_WORKERS = 12
VERBOSE = True
################

import os
import json
import obonet
import ast
import pandas as pd
import networkx as nx
import multiprocessing as mp
from Bio import Entrez
from functools import partial

import merpy

from src.articles_download import download_articles
from ncoRP_creation import create_merpy_lexicon, get_sentences, get_entities
from src.llm_re import LLMClassifier, load_examples, format_examples, save_preds


def sentence_el_worker(article:dict, rna_lexicon:str, phen_lexicon:str):
	anns = []
	pmcid = article['IDs']['pmc']
	for paragraph in article['paragraphs']:
		# Extract sentences
		sentences = get_sentences(paragraph, 500, 50)
		if sentences is None:
			continue

		for sentence in sentences:
			# Get entities in sentence
			sent_ents = get_entities(sentence, rna_lexicon, phen_lexicon)
			if sent_ents is None:
				continue
			rna_ents, phen_ents = sent_ents
			
			for rna in rna_ents:
				for phen in phen_ents:
					if (rna[0] <= phen[1]) and (rna[0] >= phen[0]): # ents overlap?
						continue
					if rna[0] < phen[0]:
						e1, e2 = rna, phen  
					else:
						e1, e2 = phen, rna
					
					ann = {
						'e1':{'type':e1[4], 'text':e1[2], 'ID':e1[3], 'start':e1[0], 'end':e1[1]},
						'e2':{'type':e2[4], 'text':e2[2], 'ID':e2[3], 'start':e2[0], 'end':e2[1]},
						'sentence': sentence, 'pmcid': pmcid
					}
					anns.append(ann)
	return anns


def llm_response_parser(text:str):
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


def main():
	if VERBOSE: print(f'\nOutput folder created at {OUTPUT_FOLDER}.', flush=True)
	os.makedirs(OUTPUT_FOLDER, exist_ok=True)

	### DOWNLOAD ARTICLES ###
	if not os.path.exists(ARTICLES_FILE):
		Entrez.email = 'fc54329@alunos.fc.ul.pt'
		search_handle = Entrez.esearch(
			db='pmc',
			term='("Austim Spectrum Disorder" OR "ASD") AND "oa full text xml"[Filter]',
			retmax=50000
		)
		search_results = Entrez.read(search_handle)
		pmc_ids = search_results['IdList']
		search_handle.close()

		download_articles(
			ids=pmc_ids,
			dest_file=ARTICLES_FILE,
			batch_size=2500,
			pmids=False,
			verbose=VERBOSE
		)
	
	with open(ARTICLES_FILE, 'r') as f:
		articles = json.load(f)

	if VERBOSE: 
		n_articles = len(articles)
		print(f'\nLoaded {n_articles} articles', flush=True)


	### CREATE ASD PHENOTYPE LEXICON ###
	if VERBOSE: print('\nLoading lexicons', flush=True)
	phen_lexicon = 'asd-phenotypes'
	if phen_lexicon not in merpy.get_lexicons()[1]: 
		ont = obonet.read_obo('./data/hp.obo')
		# Get only the descendants of Autistic Behaviour (HP:0000729)
		ont_phens = nx.ancestors(ont, 'HP:0000729')
		ont_phens.add('HP:0000729') # inclusive
		phen_maps = {}
		for _id in ont_phens:
			node = ont.nodes[_id]
			phen_name = node.get('name')
			if phen_name is not None:
				phen_name = phen_name
			all_names = [phen_name]
			syns = node.get('synonym')
			if syns is not None:
				syns = [syn.split('"')[1] for syn in syns]
				all_names.extend(syns)
			for name in all_names:
				phen_maps[name.lower()] = _id
		create_merpy_lexicon(phen_lexicon, list(phen_maps.keys()), phen_maps)

	rna_lexicon = 'ncrnas-rnac'
	if rna_lexicon not in merpy.get_lexicons()[1]: 
		aliases_df = pd.read_csv('./data/hgnc_rna_aliases.txt', sep='\t')
		rna_aliases = {} # {rna:[aliases]}
		for _,row in aliases_df.iterrows():
			rna, prev_names, alias_names = row
			row_aliases = []
			if not pd.isna(prev_names):
				row_aliases.extend(prev_names.split(', '))
			if not pd.isna(alias_names):
				row_aliases.extend(alias_names.split(', '))
			if row_aliases:
				rna_aliases[rna.lower()] = row_aliases
		# Pre-compute RNA IDs
		rna_ids_df = pd.read_csv('./data/RNACentral_mapping_human.csv', sep='\t', header=None)
		rna_maps = {} # {rna:id}
		for _,row in rna_ids_df.iterrows():
			rnac_id, ext_id, name = row
			if not isinstance(name, float): # not NaN?
				rna_maps[name.lower()] = rnac_id
			if not isinstance(ext_id, float): # not NaN?
				rna_maps[ext_id.lower()] = rnac_id
		del rna_ids_df
		# Expand ID dict with aliases
		for rna,aliases in rna_aliases.items():
			rna_id = rna_maps.get(rna)
			if rna_id is not None:
				for alias in aliases:
					alias = alias.lower()
					if alias not in rna_maps.keys():
						rna_maps[alias] = rna_id
		ignore = set(['ncRNA', 'non-coding RNA', 'lncRNA', 'small nuclear RNA', 'TR', 'E', 'REST', 'LUST', 'RAIN', 'PLANE'])
		for name in ignore:
			rna_maps.pop(name.lower(), None)
		create_merpy_lexicon(rna_lexicon, list(rna_maps.keys()), rna_maps)


	### GET SENTENCES WITH MENTIONS ###
	if VERBOSE: print(f'\nStarting sentence EL with {N_WORKERS} workers', flush=True)
	worker_fn = partial(
		sentence_el_worker, 
		rna_lexicon=rna_lexicon,
		phen_lexicon=phen_lexicon,
	)

	results = []
	if N_WORKERS > 1:
		with mp.Pool(processes=N_WORKERS) as pool:
			for i,result in enumerate(pool.imap_unordered(worker_fn, articles), 1):
				results.append(result)
				if VERBOSE: print('Article %5i/%5i' % (i, n_articles), end='\n', flush=True)
	else:
		for i,article in enumerate(articles, 1):
			result = worker_fn(article)
			results.append(result)
			if VERBOSE: print('Article %5i/%5i' % (i, n_articles), end='\n', flush=True)

	del articles
	annotations = []
	for result in results:
		annotations.extend(result)
	
	if VERBOSE: print(f'\nDone. A total of {len(annotations)} annotations found.', flush=True)


	### DEPLOY LLMS ###
	queries = [
		{'sentence':ann['sentence'], 'e1':ann['e1']['text'], 'e2':ann['e2']['text']} 
		for ann in annotations
	]

	metadatas = [
		(ann['sentence'], ann['e1']['text'], ann['e1']['ID'], ann['e2']['text'], ann['e2']['ID']) 
		for ann in annotations
	]

	# Filter out uncertain sentences
	if VERBOSE: print(f'\nQuerying LLM to filter uncertains', flush=True)
	model = 'phi3'

	prompt_u = (
		'Do you think the following sentence conveys information about the relation between "{e1}" and "{e2}" in a clear way? Sentence: "{sentence}"\n'
		'You must also explain the reasoning behind your answer\n'
		'Your response should be a JSON object with two fields: "answer", "explanation".\n'
	)
	query_template_u = 'Do you think the following sentence conveys information about the relation between "{e1}" and "{e2}" in a clear way? Sentence: "{sentence}"'

	raw_examples_u = load_examples('./data/llms/llm_examples_u.json')
	example_template_u = '###Example: Do you think the following sentence conveys information about the relation between "{e1}" and "{e2}" in a clear way? Sentence: "{sentence}"\n ###Expected Response: {response}'
	examples_u = format_examples(raw_examples_u, example_template_u)

	llm_u = LLMClassifier(
		model=model,
		base_prompt=prompt_u,
		examples=(10,examples_u),
		query_template=query_template_u,
		response_parser=llm_response_parser,
		options=None,
		timeout=600
	)

	preds_u = llm_u.evaluate(queries=queries,verbose=VERBOSE)

	save_preds(
		file=OUTPUT_FOLDER+'llm_preds_u.csv',
		llm_re=llm_u,
		preds=preds_u,
		metadatas=metadatas,
	)

	filtered_queries = []
	filtered_metadatas = []
	filtered_annotations = []
	for i,pred in enumerate(preds_u):
		if isinstance(pred,dict) and pred['answer'] == 1:
			filtered_queries.append(queries[i])
			filtered_metadatas.append(metadatas[i])
			filtered_annotations.append(annotations[i])
	
	if VERBOSE: print(f'\n{len(queries)-len(filtered_queries)} sentences filtered out as uncertain. Proceeding with {len(filtered_queries)}', flush=True)
	
	# Predict labels
	if VERBOSE: print(f'\nQuerying LLM to predict labels', flush=True)
	prompt = (
		'Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}".\n'
		'You must provide an explanation for your answer.\n'
		'Your response should be a JSON object with two fields: "relation" and "explanation".\n'
	)
	query_template = 'Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}".'

	raw_examples = load_examples('./data/llms/llm_examples.json')
	example_template = '###Example: Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}". ###Expected Response: {response}'
	examples = format_examples(raw_examples, example_template)

	llm = LLMClassifier(
		model=model,
		base_prompt=prompt,
		examples=(10,examples),
		query_template=query_template,
		response_parser=llm_response_parser,
		options=None,
		timeout=600
	)

	preds = llm.evaluate(queries=filtered_queries, verbose=VERBOSE)


	### OUTPUT ###
	save_preds(
		file=OUTPUT_FOLDER+'llm_preds.csv',
		llm_re=llm,
		preds=preds,
		metadatas=filtered_metadatas,
	)

	rows = {
		"e1_type":[], "e1_text":[], "e1_ID":[], "e1_start":[], "e1_end":[],
		"e2_type":[], "e2_text":[], "e2_ID":[], "e2_start":[], "e2_end":[],
		"sentence":[], "pmcid":[], "label":[]
	}
	for i,pred in enumerate(preds):
		if not isinstance(pred, dict): # invalid pred
			continue
		
		for ent in ['e1','e2']:
			for info in ['type','text','ID','start','end']:
				rows[f"{ent}_{info}"].append(filtered_annotations[i][ent][info])

		rows["pmcid"].append(filtered_annotations[i]["pmcid"])
		rows["sentence"].append(filtered_annotations[i]["sentence"])
		rows["label"].append(pred['answer'])

	output_df = pd.DataFrame(rows)
	output_df.to_csv(OUTPUT_FOLDER+'asd_rels.csv', sep='\t', index=None, na_rep='None')	


if __name__ == '__main__':
	main()