''' 
ncoRP Corpus Creation Pipeline
David da Costa Correia @ FCUL & INSA
 
0. Notes:
	1. Pre-requisites
		1. Download data (download_data.sh)
	2. Total runtime of the pipeline is ~1 day with 8 workers (depending on the data)
1. What this script does
	1. Call dataset_creation.py to create the Relation Dataset 
		NOTE: To change the databases used to create the dataset, changes need to be made in dataset_creation.py
		1. Process and merge databases
		2. Propagate relations to RNA aliases
		3. Add IDs to RNAs
		4. Add IDs to phenotypes
		5. Propagate relations to phenotype ancestors
	2. Call articles_download.py to download the articles
 	3. Create ncRNA & phenotype lexicons
	4. Build corpus
		(For each article)
		1. Process the text to get isolated sentences
		2. Use merpy to identify ncRNAs and phenotypes in the sentences
		3. Identify existing relations
		4. Create output JSON files
'''

### SETTINGS ##########################################################

# Paths
DATA_FOLDER = './data/'
OUTPUT_FOLDER = './outputs/corpus/'
ARTICLES_FILE = DATA_FOLDER+'articles.json'
LOG_FILE = OUTPUT_FOLDER+'/corpus.log'
DATASET_FILE = './outputs/dataset/rel_dataset.csv'

# Relation Dataset columns
RNA_NAME_COLUMN = 'ncRNA'
RNA_ID_COLUMN = 'RNAC ID' # If no ID available, leave None
PHEN_NAME_COLUMN = 'Disease'
PHEN_ID_COLUMN = 'HPO ID' # If no ID available, leave None
DATABASE_COLUMN = 'DB'
ARTICLE_ID_COLUMN = 'PMID'
# NOTE: if the relation dataset doesn't yet exist, it will be created with the column names here defined

# Others
VERBOSE = True
ZIP_OUTPUT = True
N_WORKERS = 8

DATASET_CREATION_ARGS = {
	'OUTPUT_FILE':DATASET_FILE,
	'RNA_NAME_COLUMN':RNA_NAME_COLUMN,
	'PHEN_NAME_COLUMN':PHEN_NAME_COLUMN,
	'DATABASE_COLUMN':DATABASE_COLUMN,
	'ARTICLE_ID_COLUMN':ARTICLE_ID_COLUMN,
	'RNA_ID_COLUMN':RNA_ID_COLUMN,
	'PHEN_ID_COLUMN':PHEN_ID_COLUMN,
	# NER settings
	'DISTANCE_METRIC':'l2',
	'MAX_DISTANCE':0.5, # In general, decrease for precision, increase for coverage
	'BATCH_SIZE':1000,
	'DO_ANCESTOR_PROPAGATION':True, # Relations of phenotype will be propagated to all ancestors of phenotype
	# Others
	'VERBOSE':VERBOSE,
	'DROP_NANS':True # At the end, drop rows with NaNs? (e.g. rows whose IDs could not be found)
}

########################################################################

import pandas as pd
import json
import os
import shutil
from functools import partial
from time import time
import multiprocessing as mp
import zipfile
import obonet
import networkx as nx

import merpy

import dataset_creation
import articles_download


RNA_COLUMN = RNA_NAME_COLUMN if RNA_ID_COLUMN is None else RNA_ID_COLUMN
PHEN_COLUMN = PHEN_NAME_COLUMN if PHEN_ID_COLUMN is None else PHEN_ID_COLUMN	
ONLY_IDS = all((RNA_ID_COLUMN is not None, PHEN_ID_COLUMN is not None))


def create_merpy_lexicon(name:str, terms:list, mappings:dict=None):
	"""Creates a merpy lexicon given a list of `terms`, and optionally, ID `mappings`

	:param name: the name of the lexicon.
	:type name: str
	:param terms: list of terms to add to the lexicon.
	:type terms: list
	:param mappings: `{term:ID}` dict, defaults to None
	:type mappings: dict, optional
	"""
	merpy.create_lexicon(set(terms), name)
	if mappings is not None:
		merpy.create_mappings(mappings, name)
	merpy.process_lexicon(name)


def clean_sentence(sentence:str) -> str:
	"""Processes a sentence.

	:param sentence: The sentence to process.
	:type sentence: str

	:return: The processed sentence.
	:rtype: str
	"""
	sentence = sentence.replace('\n','')
	sentence = sentence.replace('&','and')
	sentence = sentence+'.'
	return sentence


def get_sentences(paragraph:str, min_paragraph_size:int, min_sentence_size:int):
	"""Separates a block of text i.e. a paragraph in sentences. 
	Checks if the paragraph is bigger than a `min_paragraph_size`, 
	and returns only sentences bigger than `min_sentence_size`.

	:param paragraph: The paragraph to split into sentences.
	:type paragraph: str
	:param min_paragraph_size: The minimum size threshold of a paragraph. 
	If the paragraph is smaller than this threshold, will return None.
	:type min_paragraph_size: int
	:param min_sentence_size: The minimum size threshold of a sentence.
	:type min_sentence_size: int
	:return: The list of sentences extracted from the paragraph. If no sentences were extracted, returns None
	:rtype: list, None
	"""
	if len(paragraph) < min_paragraph_size:
		return None
	
	raw_sentences = paragraph.split('. ')
	sentences = []
	for sentence in raw_sentences:
		if len(sentence) >= min_sentence_size:
			sentence = clean_sentence(sentence)
			sentences.append(sentence)
		else:
			continue
	
	if sentences:
		return sentences


def process_entities(entities:list, type:str):
	"""Processes merpy entity list. 
	NOTE: This function should be adjusted for different data/needs.

	:param entities: The list of merpy entities. A merpy entity is itself a list: `[start_pos, end_pos, text, ID]`
	:type entities: list
	:param type: Enables different additional processing depending on custom entity types.
	:type type: str
	:return: List of processed merpy entities
	:rtype: list
	"""
	used_ents = set()
	clean_ents = []
	for ent in entities:
		# Check if entity has ID
		if len(ent) < 4: # If no ID
			ent.append('None')

		# Make the positions int
		ent[0] = int(ent[0])
		ent[1] = int(ent[1])

		# Make the text lowercase
		if not ONLY_IDS: ent[2] = ent[2].lower()

		# Type specific processing
		if type == 'phen':
			# Format ID
			# ent[3] = ent[3].split('/')[-1].replace('_',':')
			ent.append('Phenotype') 
		else: # if 'rna'
			ent.append('ncRNA') 

		# Remove duplicates
		if ent[2] not in used_ents: 
			clean_ents.append(ent)
			used_ents.add(ent[2])

	return clean_ents


def get_entities(sentence:str, rna_lexicon:str, phen_lexicon:str) -> tuple[list, list]:
	"""Deploys `merpy.get_entities()` for both lexicons to find entities in a sentence.
	Calls `process_entities()` to process the entities found.

	:param sentence: The sentence where entities will be found.
	:type sentence: str
	:param rna_lexicon: The first merpy lexicon name.
	:type rna_lexicon: str
	:param phen_lexicon: The second merpy lexicon name.
	:type phen_lexicon: str
	:return: A tuple of lists, containing the entities found from both lexicons, respectively.
	:rtype: tuple
	"""
	# Search for ncRNAs
	rna_ents = [ent for ent in merpy.get_entities(sentence, rna_lexicon) if ent not in [[''],['-1','-1']]]
	if not rna_ents: 
		return None
	
	# Search for phenotypes
	phen_ents = [ent for ent in merpy.get_entities(sentence, phen_lexicon) if ent not in [[''],['-1','-1']]]
	if not phen_ents: 
		return None
	
	# Process entities. An entity is a list: [start, end, text, ID, type]
	rna_ents = process_entities(rna_ents, 'rna')
	phen_ents = process_entities(phen_ents, 'phen')
	return rna_ents, phen_ents


def check_relation(rna:str, phen:str, relation_df:pd.DataFrame) -> tuple[int, list]:
	"""Find rows in a df that contain both `rna` and `phen`, i.e. a relation between `rna` and `phen`.
	The dataframe index should be pd.MultiIndex

	:param df: The dataframe to search for relations.
	:type df: pd.Dataframe
	:param rna: The first entity in the candidate relation
	:type rna: str
	:param phen: The second entity in the candidate relation
	:type phen: str
	:return: A tuple (int, list): the int can be 0 (no relation) or 1 (relation); 
	the list contains the databases that contain the relation i.e. the evidence for the relation.
	:rtype: tuple
	"""
	# Query the df for rows where rna and phen both appear
	try:
		evidence = relation_df.loc[(rna, phen)][DATABASE_COLUMN]
		if isinstance(evidence, str):
			evidence = [evidence]
		else:
			evidence = list(set(evidence))
		return 1, evidence
	except KeyError:
		return 0, []


def get_relations(rna_ents:list, phen_ents:list, relation_df:pd.DataFrame):
	"""Calls `check_relation()` for all pairs of elements from `rna_ents` and `phen_ents`.
	Returns a list of relations between. Each relation: 
	:param rna_ents: The first list of merpy entities
	:type rna_ents: list
	:param phen_ents: The second list of merpy entities
	:type phen_ents: list
	:param relation_df: The dataframe containing the relations.
	:type relation_df: pd.Dataframe
	:return: The list of relations. 
	Each relation is represented by a dict: `{'e1':{'type', 'text', 'ID', 'start', 'end'}, 'e2':{...}, 'relation', 'evidence'}`.
	:rtype: list
	"""
	relations = []
	for rna in rna_ents:
		for phen in phen_ents:
			# Check if entities overlap
			if (rna[0] <= phen[1]) and (rna[0] >= phen[0]):
				continue

			# Check what entity comes first
			if rna[0] < phen[0]:
				e1, e2 = rna, phen
			else:
				e1, e2 = phen, rna
		
			# Check relation between entities
			label, evidence = check_relation(
				rna=rna[3], phen=phen[3],
				relation_df=relation_df,
			)

			# Prepare output
			relation = {
				'e1':{'type':e1[4], 'text':e1[2], 'ID':e1[3], 'start':e1[0], 'end':e1[1]},
				'e2':{'type':e2[4], 'text':e2[2], 'ID':e2[3], 'start':e2[0], 'end':e2[1]},
				'relation': label,
				'evidence': evidence
			}
			relations.append(relation)
	
	if relations:
		return relations


def save_corpus_csv(corpus_dir:os.PathLike):
	files = [os.path.join(root,file) for root,_,fils in os.walk(corpus_dir) for file in fils]
	rows = {
		"e1_type":[], "e1_text":[], "e1_ID":[], "e1_start":[], "e1_end":[],
		"e2_type":[], "e2_text":[], "e2_ID":[], "e2_start":[], "e2_end":[],
		"sentence":[], "pmid":[], "label":[], "evidence":[]
		}
	for file in files:
		if file.endswith('.log'):
			continue
		
		try:
			with open(file, 'r') as f:
				article = json.load(f)
		except Exception:
			continue
		
		pmid = int(os.path.splitext(file)[0].split('/')[-1])
		for sent_dict in article:
			sent = sent_dict['sentence']
			for rel in sent_dict['relations']:	
				
				for ent in ['e1','e2']:
					for info in ['type','text','ID','start','end']:
						rows[f'{ent}_{info}'].append(rel[ent][info])
				
				rows['pmid'].append(pmid)
				rows['sentence'].append(sent)
				rows['label'].append(rel['relation'])
				rows['evidence'].append(rel['evidence'])

	corpus_df = pd.DataFrame(rows)
	f_name = os.path.join(corpus_dir,os.path.basename(corpus_dir)+'.csv')
	corpus_df.to_csv(f_name, sep='\t', index=None, na_rep='None')


def save_corpus_zip(corpus_dir:os.PathLike):
	zip_file = corpus_dir.strip('/')+'.zip'
	with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as f:
		for root, _, files in os.walk(corpus_dir):
			for file in files:
				file_path = os.path.join(root, file)
				f.write(file_path, os.path.relpath(file_path, corpus_dir))


def worker(article:dict, rna_lexicon:str, phen_lexicon:str, relation_df:pd.DataFrame):
	"""The function called by each sub-process. Extracts all relations in an article.

	:param article: A dict representing an article: `{'IDs':{'pmid':str,...}, 'is_full_text':bool, 'paragraphs':list}`.
	:type article: dict
	:param rna_lexicon: The first merpy lexicon name.
	:type rna_lexicon: str
	:param phen_lexicon: The second merpy lexicon name.
	:type phen_lexicon: str
	:param relation_df: The dataframe containing the relations.
	:type relation_df: pd.DataFrame
	:return: A tuple containing the number of relations in an article and the number of positives.
	:rtype: tuple
	"""
	pmid = int(article['IDs']['pmid'])
	paragraphs = article['paragraphs']
	
	try:
		art_rel_df = relation_df.loc[pmid]
	except KeyError:
		art_rel_df = pd.DataFrame()

	anns = []
	n_rels = 0
	n_pos = 0  
	for paragraph in paragraphs:
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

			# Check relations between entities
			sent_rels = get_relations(rna_ents, phen_ents, art_rel_df)
			if sent_rels is None:
				continue
			sent_ann = {'sentence':sentence, 'relations':sent_rels}
			anns.append(sent_ann)

			# Update counts
			for rel in sent_rels:
				n_rels += 1
				n_pos += rel['relation']
	
	# Output
	if anns:
		# Write output file
		if n_pos > 0:
			dest_file = os.path.join(OUTPUT_FOLDER+'positives',f'{pmid}.json')
		else:
			dest_file = os.path.join(OUTPUT_FOLDER+'negatives',f'{pmid}.json')
		with open(dest_file, 'w') as f:
			json.dump(anns, f, indent=1)

	return (n_rels, n_pos)


def main():
	t1 = time()
	### CREATE RELATION DATASET (if needed) ###
	if VERBOSE: print('Initializing Corpus Creation')
	if not os.path.exists(DATASET_FILE):
		if VERBOSE: print('\nRelation Dataset not found. Calling dataset_creation.py')
		os.environ["TOKENIZERS_PARALLELISM"] = "false" # necessary due to some conflict due to multiprocessing
		dataset_creation.main(args=DATASET_CREATION_ARGS)
	
	# Load Database
	if ONLY_IDS:
		rel_df = pd.read_csv(DATASET_FILE, sep='\t').dropna()
	else:
		rel_df = pd.read_csv(DATASET_FILE, sep='\t').fillna('None')
	

	### CREATE LEXICONS (if needed) ###
	if VERBOSE: print('\nCreating lexicons', flush=True)
	# Create ncRNA lexicon
	rna_lexicon = 'ncrnas-rnac'
	if rna_lexicon not in merpy.get_lexicons()[1]: 
		aliases_df = pd.read_csv(DATA_FOLDER+'hgnc_rna_aliases.txt', sep='\t')
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
		rna_ids_df = pd.read_csv(DATA_FOLDER+'RNACentral_mapping_human.csv', sep='\t', header=None)
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
		ignore = set(['ncRNA', 'non-coding RNA', 'lncRNA', 'small nuclear RNA', 'TR', 'E', 'REST', 'LUST', 'RAIN'])
		for name in ignore:
			rna_maps.pop(name.lower(), None)
		create_merpy_lexicon(rna_lexicon, list(rna_maps.keys()), rna_maps)

	# Create phenotype lexicon
	phen_lexicon = 'phenotypes'
	if phen_lexicon not in merpy.get_lexicons()[1]: 
		ont = obonet.read_obo(DATA_FOLDER+'hp.obo')
		# Get only the descendants of Phenotypic Abnormality (HP:0000118)
		ont_phens = nx.ancestors(ont, 'HP:0000118') 
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


	### CREATE INDEX ###
	rel_df = rel_df[[RNA_COLUMN, PHEN_COLUMN, ARTICLE_ID_COLUMN, DATABASE_COLUMN]]
	colnames = [ARTICLE_ID_COLUMN, RNA_COLUMN, PHEN_COLUMN]
	index = pd.MultiIndex.from_frame(rel_df[colnames])
	rel_df.set_index(index, inplace=True)
	rel_df.sort_index(inplace=True)


	### DOWNLOAD ARTICLES (if needed) ###
	if not os.path.exists(ARTICLES_FILE):
		if VERBOSE: print(f'\nArticles file ({ARTICLES_FILE}) not found. Calling articles_download.py', flush=True)
		pmids = list(set(rel_df['PMID']))
		articles_download.download_articles(
			ids=pmids,
			dest_file=ARTICLES_FILE,
			batch_size=2500,
			pmids=True,
			verbose=VERBOSE
		)
	# Load articles
	if VERBOSE: print('\nLoading articles', flush=True)
	with open(ARTICLES_FILE, 'r') as f:
		articles = json.load(f)
	n_articles = len(articles)

	if VERBOSE:
		n_full_text = 0
		for article in articles:
			if article['is_full_text']: 
				n_full_text += 1
		print(f'Total number of articles: {n_articles} ({n_full_text} Full-Text | {n_articles-n_full_text} Abstracts)\n')


	### CREATE OUTPUT FOLDER TREE ###
	if VERBOSE: print(f'\nOutput folder created at {OUTPUT_FOLDER}.', flush=True)
	try:
		shutil.rmtree(OUTPUT_FOLDER)
	except FileNotFoundError: pass	
	os.makedirs(OUTPUT_FOLDER+'/positives')
	os.makedirs(OUTPUT_FOLDER+'/negatives')


	### DEPLOY WORKERS ###
	worker_fn = partial(
		worker, 
		rna_lexicon=rna_lexicon,
		phen_lexicon=phen_lexicon,
		relation_df=rel_df,
	)

	if VERBOSE:
		print('Starting corpus creation')
		print(f'Launching {N_WORKERS} processes.\n')
	
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


	### FINALIZE ###
	n_inputs = len(results)
	results = [res for res in results if res[0] > 0]
	n_rels = sum([res[0] for res in results])
	n_pos = sum([res[1] for res in results])
	n_neg = n_rels-n_pos

	t2 = time()

	with open(LOG_FILE, 'w') as f:
		f.write(f'Total runtime: {t2-t1:.2f}s\n')
		f.write(f'Total articles annotated: {len(results)}/{n_inputs}\n')
		f.write(f'Total number of relations: {n_rels}\n')
		f.write(f'Negatives - {n_neg} | Positives - {n_pos}\n')
	if VERBOSE:
		print(f'\nDone. Total runtime: {t2-t1:.2f}s')
		print(f'Total articles annotated: {len(results)}/{n_inputs}')
		print(f'Total number of relations: {n_rels}')
		print(f'Negatives - {n_neg} | Positives - {n_pos}')


	### OUTPUT ###
	save_corpus_csv(OUTPUT_FOLDER)
	if ZIP_OUTPUT:
		save_corpus_zip(OUTPUT_FOLDER)
		
		
if __name__ == '__main__':
	main()