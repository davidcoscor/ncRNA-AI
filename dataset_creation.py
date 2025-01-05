''' 
Relation Dataset Creation script
David da Costa Correia @ FCUL & INSA
 
0. Notes
	1. This script is called from corpus_creation.py, but to change the databases 
	used to create the dataset, changes need to be made in this script:
		1. To add a new database, you need to define: i) input files, ii) processing functions.
		Follow examples given.
1. What this script does:
	1. Process the databases and merge
	2. Add RNA IDs and aliases
	3. Perform NER of phenotypes
	4. Propagate relations to phenotype ancestors
	5. Write output CSV file
'''

### SETTINGS ##########################################################
ARGS = {
	'OUTPUT_FILE': './outputs/dataset/rel_dataset.csv',
	# Database columns
	'RNA_NAME_COLUMN': 'ncRNA',
	'PHEN_NAME_COLUMN': 'Disease',
	'DATABASE_COLUMN': 'DB',
	'ARTICLE_ID_COLUMN': 'PMID',
	# Optional columns
	'RNA_ID_COLUMN': 'RNAC ID',
	'PHEN_ID_COLUMN': 'HPO ID',
	# NER settings
	'DISTANCE_METRIC': 'l2',
    'MAX_DISTANCE': 0.5,
    'BATCH_SIZE': 1000,
    'DO_ANCESTOR_PROPAGATION': True,
	# Others
	'VERBOSE': True,
	'DROP_NANS': True # At the end, drop rows with NaNs? (e.g. rows whose IDs could not be found)
}
DEBUG = False
PRODUCE_NER_ANALYSIS = False
NER_ANALYSIS_SIZE = 200
#######################################################################

import pandas as pd
import obonet
import networkx as nx
import os
import warnings

from src.relDS import RelationDataset
from src.FAISS_EL import *


### PROCESSING FUNCTIONS ###
def process_ncr(input_files):
	df = pd.read_csv(input_files[0], sep='\t', encoding_errors='ignore')
	df = pd.concat(objs=[df['ncRNA'], df['Disease'], df['PMID']], axis=1)
	df['ncRNA'] = df['ncRNA'].str.lower()
	df['Disease'] = df['Disease'].str.lower()
	df['RNAC ID'] = None
	df['HPO ID'] = None
	return df[['RNAC ID', 'ncRNA', 'HPO ID', 'Disease', 'PMID']]


def process_rsc(input_files):
	with open(input_files[0], 'r') as f:
		lines = f.readlines()
	rsc_info = pd.read_csv(input_files[1], sep='\t')
	rsc_pmids = list(rsc_info['PMID']) # PMIDs 
	rsc_sents = [(rsc_pmids[i],sent.split('\n')) for i,sent in enumerate(''.join(lines).split('\n\n')[:-1])] # Associate PMIDs with respective sentencess
	rsc_rows = []
	for pmid,sent in rsc_sents:
		# Extract diseases and rnas from sentences
		rnas = []
		diss = []
		for word in sent:
			text,tag = tuple(word.split(' '))
			if 'b-rna' in tag:
				rnas.append(text)
			elif 'i-rna' in tag:
				rnas[-1] += ' '+text
			elif 'b-disease' in tag:
				diss.append(text)
			elif 'i-disease' in tag:
				diss[-1] += ' '+text
		# Remove rnas or diseases smaller than 2 chars
		rnas = [rna for rna in rnas if len(rna)>=2]
		diss = [dis for dis in diss if len(dis)>=2]
		# Combine all rnas with all diseases
		for rna in rnas:
			for dis in diss:
				rsc_rows.append({'ncRNA':rna, 'Disease':dis, 'PMID':pmid})
	df = pd.DataFrame(rsc_rows)
	df['RNAC ID'] = None
	df['HPO ID'] = None
	df.drop_duplicates(ignore_index=True, inplace=True)
	return df[['RNAC ID', 'ncRNA', 'HPO ID', 'Disease', 'PMID']]


def process_hmdd(input_files):
	df = pd.read_csv(input_files[0], sep='\t')
	df = df[df['miRNA'].str.contains('hsa')]
	df = pd.concat(objs=[df['miRNA'], df['disease'], df['PMID']], axis=1)
	df['miRNA'] = df['miRNA'].str.lower()
	df['disease'] = df['disease'].str.lower()
	df['RNAC ID'] = None
	df['HPO ID'] = None
	return df[['RNAC ID', 'miRNA', 'HPO ID', 'disease', 'PMID']]


def process_lnc(input_files):
	df = pd.read_csv(input_files[0], sep='\t')
	df = df[df['Species'].str.contains('Homo sapiens')]
	df = pd.concat(objs=[df['ncRNA Symbol'], df['Disease Name'], df['PubMed ID']], axis=1)
	df['ncRNA Symbol'] = df['ncRNA Symbol'].str.lower()
	df['Disease Name'] = df['Disease Name'].str.lower()
	df['RNAC ID'] = None
	df['HPO ID'] = None
	return df[['RNAC ID', 'ncRNA Symbol', 'HPO ID', 'Disease Name', 'PubMed ID']]


def process_rnad(input_files):
	df = pd.read_excel(input_files[0])
	df = df[df['specise'].str.contains('Homo sapiens')]
	df = df[~df['RNA Type'].isin(['mRNA','unknown','mtRNA','pseudo'])]
	hpo = obonet.read_obo(input_files[1])
	# Get the alternative IDs for an HPO ID
	alt_ids = {}
	for _id in hpo.nodes:
		alts = hpo.nodes[_id].get('xref')
		if alts is None:
			continue
		for alt in alts:
			alt = alt.split(':')[1] # i.e. from UMLS:C0006826 remove the tag "UMLS"
			alt_ids[alt] = _id
	# Convert alternative IDs to HPO IDs
	hpo_ids = []
	for _,row in df.iterrows():
		_,_,_,_,_,doid,mesh,kegg,_,_ = row
		for alt_id in [doid,mesh,kegg]:
			hpo_id = alt_ids.get(alt_id)
			if hpo_id is not None:
				break
		hpo_ids.append(hpo_id)
	df['RNAC ID'] = None
	df['HPO ID'] = hpo_ids
	df = df[['RNAC ID', 'RNA Symbol', 'HPO ID', 'Disease Name', 'PMID']]
	return df


def main(args=ARGS):
	### READ PARAMETERS ###
	OUTPUT_FILE = args['OUTPUT_FILE']
	RNA_ID_COLUMN = args['RNA_ID_COLUMN']
	RNA_NAME_COLUMN = args['RNA_NAME_COLUMN']
	PHEN_ID_COLUMN = args['PHEN_ID_COLUMN']
	PHEN_NAME_COLUMN = args['PHEN_NAME_COLUMN']
	DATABASE_COLUMN = args['DATABASE_COLUMN']
	ARTICLE_ID_COLUMN = args['ARTICLE_ID_COLUMN']
	DISTANCE_METRIC = args['DISTANCE_METRIC']
	MAX_DISTANCE = args['MAX_DISTANCE']
	BATCH_SIZE = args['BATCH_SIZE']
	DO_ANCESTOR_PROPAGATION = args['DO_ANCESTOR_PROPAGATION']
	VERBOSE = args['VERBOSE']
	DROP_NANS = args['DROP_NANS']
	

	### DATASET CREATION ###
	if VERBOSE: print('Initializing Relation Dataset Creation', flush=True)
	if VERBOSE: print('\nProcessing databases', flush=True)

	# Create RelationDataset instance
	colnames = [RNA_ID_COLUMN, RNA_NAME_COLUMN, PHEN_ID_COLUMN, PHEN_NAME_COLUMN, ARTICLE_ID_COLUMN]
	dataset = RelationDataset(colnames=colnames)

	# Add databases
	# ncrPheno
	inputs_ncr = ['./data/Evidence_information.txt']
	dataset.add_database(inputs_ncr, process_ncr, 'ncrPheno')
	# RIscoper
	inputs_rsc = ['./data/All-RNA-Disease_NER.txt', './data/All-RNA-Disease_Sentence.txt']
	dataset.add_database(inputs_rsc, process_rsc, 'RIscoper')
	# HMDD
	inputs_hmdd = ['./data/hmdd_data_v4.txt']
	dataset.add_database(inputs_hmdd, process_hmdd, 'HMDD')
	# lncRNA Disease
	inputs_lnc = ['./data/lncDD_causal_data.tsv']
	dataset.add_database(inputs_lnc, process_lnc, 'lncRNA-Disease')
	# RNADisease
	inputs_rnad = ['./data/RNADiseasev4.0_RNA-disease_experiment_all.xlsx', './data/hp.obo']
	dataset.add_database(inputs_rnad, process_rnad, 'RNADisease')
	

	### MERGING ###
	if VERBOSE: print('Merging databases', flush=True)
	dataset.merge_databases(db_name_col=DATABASE_COLUMN)
	dataset_df = dataset.get_df()
	if DEBUG: print("# DEBUG # Lenght of Relation Dataset: ", len(dataset_df))
	

	### NER of phenotypes ###
	if VERBOSE: print("\nStarting NER of phenotypes", flush=True)
	warnings.filterwarnings("ignore")
	model = SentenceTransformer('all-MiniLM-L6-v2')
	collection = create_embedding_collection(
		path='./data/embeddings.csv',
		model=model,
		ontology_file='./data/hp.obo',
		verbose=True
	)
	no_ids = dataset_df[dataset_df[PHEN_ID_COLUMN].isna()] # Rows that don't have IDs yet
	ids = map_ids(
		df=no_ids, 
		column=PHEN_NAME_COLUMN,
		collection=collection,
		model=model,
		distance_metric=DISTANCE_METRIC,
		max_distance=MAX_DISTANCE,
		batch_size=BATCH_SIZE,
		return_queries=PRODUCE_NER_ANALYSIS,
		verbose=VERBOSE
	)

	if PRODUCE_NER_ANALYSIS:
		analysis = pd.DataFrame()
		hpo_ids = []
		diseases = []
		for _id in ids:
			hpo_id,disease = _id
			hpo_ids.append(hpo_id)
			diseases.append(disease)
		analysis['HPO ID'] = hpo_ids
		analysis['Disease'] = diseases
		analysis = analysis.drop_duplicates(subset='Disease')
		ont = obonet.read_obo('./data/hp.obo')
		term_names = {node:ont.nodes[node]['name'] for node in ont.nodes}
		def get_name(x):
			return term_names.get(x, x)
		analysis['Term Name'] = analysis['HPO ID'].apply(get_name)
		negatives = analysis[analysis['HPO ID'].isna()].sample(int(NER_ANALYSIS_SIZE/2), random_state=1)
		try:
			positives = analysis[~analysis['HPO ID'].isna()].sample(int(NER_ANALYSIS_SIZE/2), random_state=1)
			analysis = pd.concat([negatives, positives], ignore_index=True)
			analysis.to_csv(f"./ner_analysis_{DISTANCE_METRIC}_{str(MAX_DISTANCE).replace('.','_')}.csv", sep='\t', index=False, na_rep='None')
		except:
			print('Not enough attributed IDs, analysis not possible.')
		print('PRODUCE_NER_ANALYSIS given as True. Done and exiting.')
		exit()

	dataset_df.loc[no_ids.index,PHEN_ID_COLUMN] = ids
	dataset_df.drop_duplicates(inplace=True)
	if DEBUG:
		debug_copy = dataset_df.copy(deep=True)
		debug_copy.dropna(subset=PHEN_ID_COLUMN,inplace=True)
		print('unique rnas:', len(debug_copy[RNA_NAME_COLUMN].unique()))
		print("# DEBUG # Lenght of Relation Dataset: ", len(debug_copy))
		del debug_copy

	### ADD RNA ALIASES AND IDS ###
	if VERBOSE: print('\nAdding RNA aliases and IDs', flush=True)
	# Pre-compute RNA aliases
	if VERBOSE: print('\tLoading RNA aliases', flush=True)
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
	if VERBOSE: print('\tLoading RNA IDs', flush=True)
	rna_ids_df = pd.read_csv('./data/RNACentral_mapping_human.csv', sep='\t', header=None)
	rna_ids = {} # {rna:id}
	for _,row in rna_ids_df.iterrows():
		rnac_id, ext_id, name = row
		if not isinstance(name, float): # not NaN?
			rna_ids[name.lower()] = rnac_id
		if not isinstance(ext_id, float): # not NaN?
			rna_ids[ext_id.lower()] = rnac_id
	del rna_ids_df
	# Expand ID dict with aliases
	for rna,aliases in rna_aliases.items():
		rna_id = rna_ids.get(rna)
		if rna_id is not None:
			for alias in aliases:
				rna_ids[alias] = rna_id

	# Expand dataset with aliases
	if VERBOSE: print('\tPropagating relations to RNA aliases', flush=True)
	new_rows = []
	for _,row in dataset_df.iterrows():
		aliases = rna_aliases.get(row[RNA_NAME_COLUMN])
		if aliases is None:
			continue
		for alias in aliases:
			new_row = row.copy()
			new_row[RNA_NAME_COLUMN] = alias.lower()
			new_rows.append(new_row)
	new_rows = pd.DataFrame(new_rows)
	dataset_df = pd.concat([dataset_df, new_rows], ignore_index=True)
	dataset_df.drop_duplicates(inplace=True)
	if DEBUG: print("# DEBUG # Lenght of Relation Dataset: ", len(dataset_df))
		
	# Link RNA IDs
	if VERBOSE: print('\tLinking RNAs to IDs', flush=True)
	dataset_df[RNA_ID_COLUMN] = pd.Series(dtype=str)
	unique_rnas = dataset_df[RNA_NAME_COLUMN].unique().tolist()
	unique_rna_queries = {
		rna:[rna, rna.replace('hsa-',''), rna.replace('hsa_','')]
		for rna in unique_rnas
	}

	matches = {}
	for rna,queries in unique_rna_queries.items():
		# Search for the first query that returns a RNAC ID
		for query in queries:
			rna_id = rna_ids.get(query)
			if rna_id is not None:
				matches[rna] = rna_id
				break
	dataset_df[RNA_ID_COLUMN] = dataset_df[RNA_NAME_COLUMN].apply(lambda rna: matches.get(rna))
	dataset_df.drop_duplicates(inplace=True)
	del unique_rnas, unique_rna_queries

	if DEBUG:
		debug_copy = dataset_df.copy(deep=True)
		debug_copy.dropna(inplace=True, subset=RNA_ID_COLUMN)
		print("# DEBUG # Lenght of Relation Dataset: ", len(debug_copy))
		print('unique rnas:', len(debug_copy[RNA_NAME_COLUMN].unique()))
		del debug_copy

	### PROPAGATE RELATIONS TO PHENOTYPE ANCESTORS ###
	if DO_ANCESTOR_PROPAGATION:
		# Precompute phenotype ancestors
		if VERBOSE: print('\nPropagating relations to phenotype ancestors', flush=True)
		if VERBOSE: print('\tLoading phenotype ancestors', flush=True)
		ontology = obonet.read_obo('./data/hp.obo')
		term_ancestors = {}
		term_names = {}
		for _id in ontology.nodes:
			term_names[_id] = ontology.nodes[_id]['name'] # Save the name of the term
			# try:
			ancestors = nx.descendants(ontology, _id) # descendants instead of ancestors due to the direction of the DAG
			# except nx.exception.NetworkXError:
			# 	continue
			# Exclude too-general terms to avoid redundancy
			ancestors.discard('HP:0000001') # 'All'
			ancestors.discard('HP:0000118') # 'Phenotypic abnormality'
			term_ancestors[_id] = ancestors
		del ontology
		
		if VERBOSE: print('\tPropagating relations', flush=True) 
		CHUNK_SIZE = 100000 # necessary due to RAM limitations
		chunks = []
		for start in range(0, len(dataset_df), CHUNK_SIZE):
			end = min(start + CHUNK_SIZE, len(dataset_df))
			chunk = dataset_df.iloc[start:end]
			chunk_new_rows = []
			for _,row in chunk.iterrows():
				# Get the ancestors of the term in the row
				ancestors = term_ancestors.get(row[PHEN_ID_COLUMN])
				if ancestors is None:
					continue
				# For each ancestor, propagate the relation
				for ancestor_id in ancestors:
					new_row = row.copy()
					new_row[PHEN_ID_COLUMN] = ancestor_id
					new_row[PHEN_NAME_COLUMN] = term_names.get(ancestor_id)
					new_row[DATABASE_COLUMN] = 'Ancestor Propagation'
					chunk_new_rows.append(new_row)
			chunks.append(pd.DataFrame(chunk_new_rows))
		
		new_rows_df = pd.concat(chunks, ignore_index=True)
		dataset_df = pd.concat([dataset_df, new_rows_df], ignore_index=True)
		dataset_df.drop_duplicates(inplace=True)
		if DEBUG:
			debug_copy = dataset_df.copy(deep=True)
			debug_copy.dropna(inplace=True)
			print("# DEBUG # Lenght of Relation Dataset: ", len(debug_copy))
			del debug_copy


	### OUTPUT ###
	if VERBOSE: print('\nFinishing')
	if DROP_NANS:
		dataset_df.dropna(inplace=True)
	dtypes = {
		RNA_ID_COLUMN:str, 
		RNA_NAME_COLUMN:str, 
		PHEN_ID_COLUMN:str, 
		PHEN_NAME_COLUMN:str, 
		ARTICLE_ID_COLUMN:str, 
		DATABASE_COLUMN:str
	}
	dataset_df = dataset_df.astype(dtypes)
	os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
	if DEBUG: OUTPUT_FILE = os.path.splitext(OUTPUT_FILE)[0]+'_debug.csv'
	dataset_df.to_csv(OUTPUT_FILE, sep='\t', index=False, na_rep='None')
	if VERBOSE: print(f'Relation Dataset creation done. Saved to {OUTPUT_FILE}', flush=True)


if __name__ == '__main__':
	main()
