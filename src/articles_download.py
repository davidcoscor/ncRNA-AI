'''
Articles Download script
David da Costa Correia @ FCUL & INSA

- Functionality:
	1. Load PMIDs from the Relation Dataset
	2. For each batch of PMIDs (using NCBI's E-utils):
		1. Convert PMIDs to PMCIDs
		2. Download the articles from PMC in XML format
		3. Parse the article text from the XML
		4. Write to output file
		- If an error is raised, it's probably due to connection/request issues. The error is handled by repeating the batch

- Usage Example:
>>> from articles_download import *

>>> pmids = ['123456789','987654321']
>>> download_articles(
		pmids = pmids,
		dest_file = 'articles.json',
		batch_size = 10,
		max_errors = 5,
		verbose = True
	)
'''

from Bio import Entrez
import xml.etree.ElementTree as ET
import json


def parse_text(parent:ET.Element, child:str) -> list:
	"""
	:param parent: parent element
	:type parent: ET.Element
	:param child: child tag. Example: './/body//p'
	:type child: str
	:return: List that contains all the text inside all child tag elements or XPath,
	that are children of a parent ET.Element object.
	:rtype: list
	"""
	return [''.join(c.itertext()) for c in parent.findall(child)]


def parse_articles(xml:str) -> list[dict]:
	"""
	:param xml: article formatted xml string 
	:type xml: str
	:return: List of dicts {'ID':str, 'paragraphs':list, 'is_full_text:bool'}
	for each <article> tag element in a formated xml string.
	:rtype: dict
	"""
	root = ET.fromstring(xml)
	articles = []
	for article_elem in root:
		article = {}
		article['IDs'] = {}
		for id_type in ['pmid','pmc']:
			found_id = article_elem.findtext(f".//article-id[@pub-id-type='{id_type}']")
			if found_id:
				article['IDs'][id_type] = found_id
			
		article['paragraphs'] = []
		abstract = parse_text(article_elem, './/abstract//p')
		article['paragraphs'].extend(abstract)

		article['is_full_text'] = False
		full_text = parse_text(article_elem, './/body//p')
		if full_text:
			article['is_full_text'] = True
			article['paragraphs'].extend(full_text)
		
		articles.append(article)
	
	return articles


def download_articles(ids:list, dest_file:str, batch_size:int, pmids:bool=False, max_errors:int=10, verbose:bool=True):
	"""Converts `pmids` to PMC IDs, searches, downloads and parses the articles.
	The process is done in batches of `batch_size` size, if an error is raised during the processing
	of a batch, the batch is retried `max_errors` number of times.
	The output is written as a list of dictionaries [{pmid:int, is_full_text:bool, paragraphs:list}] 
	to a JSON `dest_file` - each dictionary represents an article.

	:param ids: list of PMC IDs of the articles to download. The IDs can be PMIDs if `pmids` is passed as True.
	:type ids: list
	:param dest_file: path of the output file.
	:type dest_file: str
	:param batch_size: the number of articles to process in a batch. 
	In general, use bigger batches if the connection is stable.
	:type batch_size: int, optional
	:param pmids: The ids given in `ids` are PMIDs?
	:type pmids: bool
	:param max_errors: the number of times a batch is retried when an error is raised, defaults to 10
	:type max_errors: int, optional
	:param verbose: verbose?, defaults to True
	:type verbose: bool, optional
	:raises RuntimeError: when a batch fails more than `max_errors` times
	"""
	# Guarantee dest_file is empty
	with open(dest_file, 'w') as f: 
		f.write('[\n') # To make a single top-level JSON list

	# Init
	count = len(ids)
	if verbose: 
		total_found = 0
		total_full = 0
		print(f'Trying to download {count} articles.\n')
	batch_size = min(batch_size, count)

	for start in range(0, count, batch_size):
		while True: # To retry the iteration if any error is raised
			n_errors = 0
			try:
				end = min(count, start+batch_size)
				if verbose: print("Downloading %5i to %5i" % (start + 1, end), end=' ')

				if pmids: # Search for PMCIDs that link to the PMIDs
					if verbose: print('| Searching for PMCIDs', end=' ')
					link_handle = Entrez.elink(
						db="pmc",
						dbfrom="pubmed", 
						id=ids[start:end],
						retmode='xml',
						linkname='pubmed_pmc'
					)
					links = Entrez.read(link_handle) 
					link_handle.close()

					# Format the PMCIDs
					pmc_ids = []
					for link in links:
						try:
							pmc_id = link['LinkSetDb'][0]['Link'][0]['Id']
							pmc_ids.append(pmc_id)
						except IndexError:
							continue
					if len(pmc_ids) == 0: # If none are found, skip batch
						if verbose: print(f'| Found 0')
						break
				else:
					pmc_ids = ids[start:end]

				# Search for the papers in XML format
				if verbose: print('| Searching for papers', end=' ')
				fetch_handle = Entrez.efetch(
					db='pmc',
					id=pmc_ids,
					rettype='full',
					retmode='xml'
				)

				xml = fetch_handle.read()
				fetch_handle.close()

				# Parse the XML for the abstract and full-text if available
				articles = parse_articles(xml)
				if verbose:
					found = len(articles)
					print(f'| Found {found}', flush=True)
					total_found += found
					total_full += sum([int(a['is_full_text']) for a in articles])
				break

			except Exception as e:
				if n_errors < max_errors:
					n_errors += 1
					if verbose: print(f'\n"{e}" occured. Retrying last batch.')
					continue
				else:
					raise RuntimeError('The maximum number of errors was reached.')
					
		# Write to output file
		with open(dest_file, 'a') as f:
			try:
				for article in articles:
					if f.tell() > 2:
						f.write(',\n')
					json.dump(article, f, indent=2, sort_keys=True)
			except UnboundLocalError:
				continue
				
	# Finish
	with open(dest_file, 'a') as f:
		f.write('\n]') # Close the top-level JSON list
	if verbose: print(f'\nDone! Found a total of {total_found} articles ({total_full} Full-text).')