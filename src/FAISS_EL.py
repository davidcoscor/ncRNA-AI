''' 
FAISS EL Tool
David da Costa Correia @ FCUL & INSA
 
- Usage example:

>>> from FAISSNER import *
>>> model = SentenceTransformer('all-MiniLM-L6-v2')
>>> collection = create_embedding_collection(
		path='embeddings.csv',
		model=model,
		ontology_file='hp.obo',
		verbose=True
	)
>>> my_data = pd.read_csv('./my_data.csv')
>>> ids = map_ids(
		df=my_data, 
		column='Disease',
		collection=collection,
		model=model,
		distance_metric='l2', 
		batch_size=1000,
		max_distance=0.5,
		verbose=True
	)
>>> my_data['HPO ID'] = ids
'''

import faiss
from sentence_transformers import SentenceTransformer
import os
import obonet
from time import time
import pandas as pd


EmbeddingCollection = tuple[pd.DataFrame, list]


def create_embedding_collection(path:os.PathLike, model:SentenceTransformer, ontology_file:os.PathLike, 
								normalize_embeddings:bool=False, verbose:bool=False) -> EmbeddingCollection:
	"""Creates an embedding collection given an OBO `ontology_file`. If the collection already exists, 
	loads it instead. A collection is saved in a CSV file with D+1 columns where D is the size of the embeddings.
	The last column contains the IDs attributed to each embedding.

	:param path: the path to save/load the collection CSV file
	:type path: str
	:param model: a SentenceTransformer object loaded with an embedding model
	:type model: SentenceTransformer
	:param ontology_file: the ontology OBO files
	:type ontology_file: str
	:param verbose: verbose?
	:type verbose: bool
	:param normalize_embeddings: normalize the embeddings?
	:type normalize_embeddings: bool
	:return: tuple representing the collection `(embeddings:pd.DataFrame, ids:list)`
	:rtype: tuple
	"""
	if os.path.exists(path): # load
		collection = pd.read_csv(path)
		ids = list(collection['ID'])
		collection.drop(['ID'], axis=1, inplace=True)
		embeddings = collection
		if verbose: print('Embedding collection loaded')
	
	else: # create
		if verbose: print('Embedding collection not found. Creating...')
		graph = obonet.read_obo(ontology_file)
		nodes = [node for node in graph.nodes(data=True)]
		terms = []
		ids = []
		for node in nodes:
			_id, meta = node
			phen_name = meta.get('name')
			if phen_name is not None:
				terms.append(phen_name)
				ids.append(_id)
			syns = meta.get('synonym')
			if syns is not None:
				for syn in syns:
					terms.append(syn.split('"')[1])
					ids.append(_id)

		embeddings = pd.DataFrame(model.encode(terms, normalize_embeddings=normalize_embeddings))
		collection = embeddings.copy(deep=True)
		collection['ID'] = ids
		collection.to_csv(path, index=None)
		if verbose: print('Embedding collection created')
	
	return embeddings, ids

	
def map_ids(df:pd.DataFrame, column:str, collection:EmbeddingCollection, model:SentenceTransformer, 
			distance_metric:str, max_distance:float, batch_size:int, 
			return_queries:bool=False, verbose:bool=False):
	"""Maps a `df` `column`'s terms to predicted IDs.

	:param df: dataframe containing the column of interest.
	:type df: pd.DataFrame
	:param column: the `df` column containing the terms.
	:type column: str
	:param collection: the embedding collection, obtained with `create_embedding_collection()`.
	:type collection: tuple
	:param model: a SentenceTransformer object loaded with an embedding model.
	:type model: SentenceTransformer
	:param distance_metric: the distance metric to calculate distance between embeddings. Possible values are `["l2","ip"]`.
	:type distance_metric: str
	:param max_distance: the max distance threshold for an ID to be attributed to a term.
	:type max_distance: float
	:param batch_size: the amount of terms to query at the same time.
	:type batch_size: int
	:param return_queries: if True the return will be list[(id,query)] instead of list[id]
	:type return_queries: bool
	:param verbose: verbose?
	:type verbose: bool
	:return: list of the IDs corresponding to entities from the `df`'s `column` in the same order.
	:rtype: list
	"""
	df_column = df[column].tolist()
	queries = list(set(df_column))
	N = len(queries)

	t1 = time()
	if verbose: print(f'\tQuerying {N} unique entities in batches of {batch_size}.')
	
	embeddings, index_ids = collection
	if distance_metric == 'l2':
		distance_index = faiss.IndexFlatL2(embeddings.shape[1])
	elif distance_metric == 'ip':
		distance_index = faiss.IndexFlatIP(embeddings.shape[1])
	else:
		raise ValueError(f'Invalid distance metric: {distance_metric}. Possible values are ["l2","ip"].')
	distance_index.add(embeddings)
	
	entities = {}
	for start in range(0, N, batch_size):
		end = min(N, start+batch_size)
		batch_queries = queries[start:end]
		
		if verbose: print('\tBatch %5i:%5i' % (start,end), end='\r', flush=True)
		enc_query = model.encode(batch_queries)
		distances,indexes = distance_index.search(enc_query, 1)

		for i,query in enumerate(batch_queries):
			entities[query] = (indexes[i][0], distances[i][0], query)

	t2 = time()
	if verbose: print(f'\n\tQuery time: {t2-t1:.2f} s')

	if verbose: print('\tFiltering')
	ids = []
	found = set()
	for i,term in enumerate(df_column):
		index,dis,query = entities[term]
		dis = round(dis, 6) # remove artifact decimal places
		_id = None
		if dis <= max_distance:
			_id = index_ids[index]
			found.add(term)
		if return_queries:
			ids.append((_id,query))
		else: 
			ids.append(_id)

	if verbose: print(f'\tTotal IDs attributed: {len(found)}/{len(queries)}')
	return ids