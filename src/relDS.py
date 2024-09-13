''' 
RelationDataset Class Implementation
David da Costa Correia @ FCUL & INSA
'''

import pandas as pd
from os import PathLike

class RelationDataset():
	"""
	A RelationDataset enables uniform concatenation of relational databases.
	Databases are added through `RelationDataset.add_database()`, and can be merged using `RelationDataset.merge_databases()`.
	"""
	def __init__(self, colnames:list[str]):
		"""
		:param colnames: the ordered names of the columns.
		:type colnames: list
		"""
		self.colnames = colnames
		self.dbs = {}
		self.df = pd.DataFrame()

	def add_database(self, input_files:list[PathLike], processing_fn:callable, name:str) -> None:
		"""Add a new database to the RelationDataset, given its `input_files`, `processing_fn` and `name`.

		:param input_files: a list of paths to the database files.
		:type input_files: list
		:param processing_fn: the function that should process the input files and 
		return pd.DataFrame with the columns defined in `RelationDataset.colnames`.
		The columns must come in the desired order.
		:type processing_fn: callable
		:param name: The name of the database
		:type name: str
		:raises ValueError: when `processing_fn(input_files)` raises an exception; 
		OR `if len(df.columns) != len(self.colnames)`.
		"""
		try: # Try to process database
			df = processing_fn(input_files)
		except Exception as e:
			raise ValueError(f"Error processing {name} database due to Exception: {e}. Check given processing_fn.")
		
		if len(df.columns) != len(self.colnames): # incompatible columns?
			raise ValueError(f"The {name} database DataFrame's number of columns ({len(df.columns)}) is incompatible with RelationDataset ({len(self.colnames)}).")
		df.columns = self.colnames

		self.dbs[name] = df

	def merge_databases(self, db_name_col:str=None) -> None:
		"""Merges all the databases added to RelationDataset (using `RelationDataset.add_database()`).
		To obtain the merged RelationDataset dataframe, `RelationDataset.get_df()` must be called.

		:param db_name_col: the name of the column that will contain the name of the database of origin
		of each row, if this new column is not desired, leave None.
		:type db_name_col: str, optional
		"""
		# Merge and add DB column
		merged_df = pd.concat(objs=list(self.dbs.values()), keys=list(self.dbs.keys()))
		if db_name_col is not None:
			merged_df[db_name_col] = [db for db,_ in merged_df.index]
		merged_df.reset_index(drop=True, inplace=True)

		# Post-process			
		merged_df.drop_duplicates(inplace=True)
		self.df = merged_df

	def get_df(self) -> pd.DataFrame:
		"""
		:raises UserWarning: if the RelationDataset dataframe is empty.
		:return: the RelationDataset DataFrame.
		:rtype: pd.DataFrame
		"""
		if self.df.empty:
			raise UserWarning("RelationDataset DataFrame is empty. You might want to call RelationDataset.merge_databases()")
		return self.df