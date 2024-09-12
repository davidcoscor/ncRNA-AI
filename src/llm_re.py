"""
Ollama LLM Binary Classification Tool
David da Costa Correia @ FCUL & INSA
"""

import pandas as pd
import os
import json
import itertools
import random
from time import time

import ollama

class LLMClassifier:
	"""
	LLMClassifier wraps `ollama.generate()` to enable handling and evaluation of binary classification-like answers.
	Supports few-shot prompting and custom response parsing. NOTE: Voting not completely implemented.
	"""
	def __init__(
		self, model:str, base_prompt:str, examples:tuple[int,list|set]=None,
		query_template:str=None, response_parser:callable=None,
		options:dict=None, timeout=None
		):
		"""
		:param model: The name of the model, as it would be passed to ollama
		:type model: str
		:param base_prompt: The base prompt. Can be a formatted str if queries are dict, for more information see 
		`LLMClassifier.predict()`.
		:type base_prompt: str
		:param examples: For few-shot prompting. `(n_shots:int, examples:[list,set])`, `list` for the first `n_shots` ordered examples, 
		`set` for `n_shots` random examples.
		:type examples: tuple, optional
		:param query_template: formatted str, required when queries are dict, for more information see `LLMClassifier.predict()`.
		:type query_template: str, optional
		:param response_parser: Function applied to `ollama.generate()['response']`, must return dict if the parsing was successful and str otherwise.
		:type response_parser: callable, optional
		:param options: For additional options supported by ollama. See https://pypi.org/project/ollama-python/#:~:text=Valid%20Options/Parameters
		:type options: dict, optional
		:param timeout: The max response time for a query, in seconds.
		:type timeout: int, optional
		"""
		self.client = ollama.Client(timeout=timeout)
		self.base_prompt =  base_prompt
		self.model = model
		self.query_template = query_template
		self.response_parser = response_parser
		self.options = options
		
		if examples:
			self.n_shots, self.examples = examples
			self.examples_text = self._process_examples()
		else:
			self.n_shots = 0
			self.examples = None
			self.examples_text = ''
		
		self.voting = isinstance(model, list)
		if self.voting:
			assert len(model)%2 != 0, "Number of voters must be odd to avoid draws." # TEMP until weighted votes

	def _process_examples(self) -> str:
		"""
		Called by the constructor to process the examples into a text block to be used in the prompt.
		"""
		if isinstance(self.examples, set):
			# If the examples iter was passed as a set, sample self.n_shots random examples
			self.examples = random.sample(list(self.examples), self.n_shots)
		else:
			# If the examples iter is a list, use self.n_shots first examples instead
			self.examples = self.examples[:self.n_shots]
		
		example_text = "\n".join(self.examples)
		return example_text

	def _query_llm(self, model:str, prompt:str) -> dict|str:
		"""
		Called by `LLMClassifier.predict()`. 
		Calls `ollama.generate()` and returns the response. Parses the response if self.response_parser was passed.
		"""
		response = self.client.generate(model=model, prompt=prompt, options=self.options)["response"]
		if self.response_parser:
			return self.response_parser(response)
		else:
			return response
	
	def _log(self, log_file, results):
		"""
		Called by `LLMClassifer.evaluate()` to process `results` and write them to `log_file`.
		For custom processing of results and logging, implement custom `_log()` function. 
		"""
		# Compute metrics
		N, tp, fp, fn, inv, time = results
		prc = tp/(tp+len(fp)) if tp+len(fp) != 0 else 0
		rcl = tp/(tp+len(fn)) if tp+len(fn) != 0 else 0
		f1 =  2*(prc*rcl)/(prc+rcl) if prc+rcl != 0 else 0

		# Process mislabels
		mislabels = {}
		for name,insts in zip(['fp','fn','invalid'],[fp,fn,inv]):
			if not insts:
				continue
			insts = [{'query':inst[0],'model_response':inst[1]} for inst in insts]
			mislabel = {'N':len(insts), 'instances':insts}
			mislabels[name] = mislabel

		if len(mislabels) == 0:
			mislabels = None
		
		# Prepare entry
		new_entry = {
			'model': self.model,
			'n_shots': self.n_shots,
			'options': self.options,
			'prompt': self.base_prompt,
			'examples': self.examples,
			'n_queries': N,
			'runtime': {'total': time, 'single': time/N if N>0 else time},
			'invalid%': (len(inv)/N)*100 if N>0 else 0,
			'metrics': {'precision': prc, 'recall': rcl, 'f1': f1},
			'mislabels': mislabels
		}

		# Write to log file
		log = []
		if os.path.exists(log_file):
			with open(log_file, 'r') as f:
				log = json.load(f)
		log.append(new_entry)

		os.makedirs(os.path.dirname(log_file), exist_ok=True)
		with open(log_file, 'w') as f:
			json.dump(log, f, indent=1)

	def predict(self, query:dict|str, verbose:bool=False) -> dict|str:
		"""Calls `LLMClassifier._query_llm()` to make a prediction for a given `query`.

		:param query: The instance to be predicted. It will be appended in the end of the prompt.
		If `LLMClassifier.query_template` was passed, query must be dict. This is useful to apply the same template to multiple instances.
		:type query: dict | str
		:param verbose: verbose?, defaults to False.
		:type verbose: bool, optional
		:return: The LLM prediction. It will be dict if the response was successfuly parsed, else it will be str.
		:rtype: dict|str
		"""
		if self.query_template:
			assert isinstance(query, dict), "query_template provided. query must be dict with keys of query_template"
			prompt = self.base_prompt.format_map(query) + self.examples_text + self.query_template.format_map(query)
		else:
			prompt = self.base_prompt + self.examples_text + query
		
		if verbose: print(f'Query:\n{prompt}\n')
		
		if self.voting:
			votes = []
			for model in self.model:
				try:
					response = self._query_llm(model, prompt)["relation"]
					votes.append(response)
				except: pass
			votes = [vote for vote in votes if isinstance(vote,int)]
			pred = round(sum(votes)/len(votes)) if votes else None

		else:
			pred = self._query_llm(self.model, prompt)
		
		if verbose: print(f'Answer: \n{pred}\n\n')
		return pred

	def evaluate(self, queries:list[dict]|list[str], true_labels:list[int]=None, log_file:os.PathLike=None, verbose:bool=False) -> list:
		"""Calls predict on all `queries` and evaluates results if `true_labels` are passed, appeding to `log_file`.

		:param queries: The list of all queries to be predicted.
		:type queries: list[dict] | list[str]
		:param true_labels: The list of true labels of each query, true_labels[i] must refer to queries[i],
		passed the predictions are evaluated.
		:type true_labels: list[int], optional
		:param log_file: The JSON file to log the evaluation results, required if true_labels was passed.
		:type log_file: os.PathLike, optional
		:param verbose: verbose?, defaults to False.
		:type verbose: bool, optional
		:return: A list of all model predictions.
		:rtype: list
		"""
		if true_labels:
			assert len(queries) == len(true_labels), "Mismatch in texts and true_labels sizes."
		if verbose: print(f'Model: {self.model}')
		if verbose: print(f'N-shots: {self.n_shots}')
		# Predict
		N = len(queries)
		preds = []
		t1 = time()
		for i,query in enumerate(queries,1):
			if verbose: print('\tQuery %4i/%4i' % (i,N), end='\r', flush=True)
			try:
				preds.append(self.predict(query))
			except Exception as e:
				preds.append(str(e.__class__)+str(e))
		
		t2 = time()
		if verbose: print('\nDone\n')
		
		# Evaluate
		if true_labels:
			assert log_file, "log_file is required, because true_labels was passed."
			tp, fp, fn = 0, [], []
			invalid = []
			for i,pred in enumerate(preds):
				if isinstance(pred, dict):
					pred_label = pred["answer"]
					if   true_labels[i] == 1 and pred_label == 1: tp += 1
					elif true_labels[i] == 0 and pred_label == 1: fp.append((queries[i],preds[i]))
					elif true_labels[i] == 1 and pred_label == 0: fn.append((queries[i],preds[i]))
				else: 
					invalid.append((queries[i],preds[i]))
		
			self._log(log_file, results=(N, tp, fp, fn, invalid, t2-t1))
		
		return preds


def rank_models(k, log_file:str, rank_file:str=None):
	'''Ranks the best `k` models in `log_file` and returns them, optionally writes to `rank_file`'''
	with open(log_file, 'r') as f:
		entries = json.load(f)
	
	rated_entries = []
	for entry in entries:
		n_preds = entry['n_preds']
		f1 = entry['metrics']['f1']
		rated_entries.append((n_preds,f1,entry))
	rated_entries.sort(key=lambda i: i[1], reverse=True) # sort by f1
	rated_entries.sort(key=lambda i: i[0], reverse=True) # sort by n of valid preds
	best_entries = [entry[2] for entry in rated_entries[:k]]

	if rank_file:
		with open(rank_file, 'w') as f:
			json.dump(best_entries, f, indent=1)

	return best_entries


def queue_evaluation(queries:list[dict]|list[str]|list[list], true_labels:list[int], classifiers:list[LLMClassifier], log_file:str=False, verbose=False):
	'''Calls `LLMClassifier.evaluate(queries, true_labels, log_file, verbose)` for each LLMClassifier instance in `classifiers`
	
	:param queries: The list of queries. Use list[list] to call different queries on each model queries[i] will be called on classifiers[i].
	:type queries: list
	'''
	diff_queries = isinstance(queries[0], list)
	all_preds = []
	for i,llm_re in enumerate(classifiers):
		if verbose: print('\nEvaluation %2i/%2i' % (i+1, len(classifiers)), flush=True)
		llm_queries = queries[i] if diff_queries else queries
		llm_true_labels = true_labels[i] if diff_queries else true_labels
		preds = llm_re.evaluate(
			queries=llm_queries,
			true_labels=llm_true_labels, 
			log_file=log_file,
			verbose=verbose
		)
		all_preds.append(preds)

	return all_preds
	

def bulk_create(params:dict) -> list[LLMClassifier]:
	'''Creates LLM_RE instances with all the possible combinations of `params` values by passing keys as kwargs
	
	:param params: {param_name:[values]}
	:type params: dict
	:return: A list containing all the LLMClassifier instances.
	:rtype: list
	'''
	keys, values = zip(*params.items())
	combos = list(itertools.product(*values))		
	llms = [dict(zip(keys, combo)) for combo in combos]
	llms = [LLMClassifier(**llm_params) for llm_params in llms]
	return llms


def load_examples(file:os.PathLike) -> list:
	"""Returns examples loaded from a JSON file."""
	examples = []
	if os.path.exists(file):
		with open(file, 'r') as f:
			examples = json.load(f)
		return examples	


def format_examples(examples:list, template:str) -> list:
	return [template.format_map(ex) for ex in examples]


def generate_examples(examples_file:os.PathLike, balance:bool, queries:list, preds:list, true_labels:list=False):
	"""Generates examples from `queries` and LLM `preds` and saves them into `examples_file`.

	:param examples_file: The JSON file to store the examples.
	:type examples_file: str
	:param balance: Balance the examples to include equal number of positives and negatives?
	:type balance: bool
	:param queries: The queries.
	:type queries: list
	:param preds: The LLM predictions.
	:type preds: list
	:param true_labels: list of the true_labels, if given only correct preds will be saved as examples.
	:type true_labels: list, optional
	"""
	raw_examples = load_examples(examples_file)
	ex_sents = set([ex['sentence'] for ex in raw_examples])

	for j,pred in enumerate(preds):

		sent = queries[j]["sentence"]
		if isinstance(pred, str): # invalid llm response
			continue
		if sent in ex_sents: # duplicate example
			continue
		if true_labels and (true_labels[j] != pred["answer"]): # incorrect pred
			continue
			
		new_ex = {
			"sentence": sent,
			"e1": queries[j]["e1"],
			"e2": queries[j]["e2"],
			"response": pred
		}
		raw_examples.append(new_ex)

	if balance:
		labels = [ex["response"]["answer"] for ex in raw_examples]
		pos = labels.count(1)
		neg = labels.count(0)
		if pos != neg:
			most = 1 if pos>neg else 0
			diff = abs(pos-neg)
			for _ in range(diff):
				for i, label in enumerate(reversed(labels),1):
					if label == most:
						raw_examples.pop(-i)
						labels.pop(-i)
						break

	with open(examples_file, 'w') as f:
		json.dump(raw_examples, f, indent=1)


def save_preds(file:os.PathLike, classifier:LLMClassifier, preds:list, metadatas:list, true_labels:list=None):
	"""Saves the `preds` of a `classifier` to `file`."""
	llm_info = classifier.__dict__

	output = {'sentence':[], 'e1':[], 'e1_ID':[], 'e2':[], 'e2_ID':[], 'pred':[]}
	for i,pred in enumerate(preds):
		sent, e1, e1_id, e2, e2_id = metadatas[i]
		output['sentence'].append(sent)
		output['e1'].append(e1)
		output['e1_ID'].append(e1_id)
		output['e2'].append(e2)
		output['e2_ID'].append(e2_id)
		if isinstance(pred, dict):
			output['pred'].append(pred['answer'])
		else:
			output['pred'].append(pred)
	
	if true_labels: 
		output['true_label'] = true_labels
	
	os.makedirs(os.path.dirname(file), exist_ok=True)
	output = pd.DataFrame(output)
	output.to_csv(file, sep='\t', index=None)
	with open(file, 'a') as f:
		f.write(f'\n! LLM parameters: {llm_info}')