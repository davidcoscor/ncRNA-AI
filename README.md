## ncRNA-AI
Source code supporting David da Costa Correia's MSc thesis project "Predicting non-coding RNA function using artificial intelligence".

Supervised by Hugo Martiniano, PhD and Francisco Couto, PhD.

Executed at FCUL & INSA, Portugal in 2023-2024.

### File information

```
ncRNA-AI
├── src                       | Contains the developed modules that support the pipelines
│   ├── articles_download.py  | Implements a simple framework to download articles using NCBI's E-utils
│   ├── FAISS_EL.py           | FAISS and SentenceTransformers Entity Linking Tool
│   ├── relDS.py              | Implements RelationDataset class
│   └── llm_re.py             | Implements an Ollama-based LLM binary classification framework
├── utils                     | Contains other utility python scripts
├── misc                      | Contains supporting jupyter notebooks for data/output analysis
├── data                      | Contains the data (raw and processed) used by the pipelines
├── outputs                   | Contains the final outputs from the pipelines
├── ncoRP_creation.py         | ncoRP corpus creation pipeline
├── dataset_creation.py       | Relation Dataset creation pipeline
├── llm_exp.py                | LLM methodology implementation
├── asd_cs.py                 | Austim Spectrum Disorder Case Study pipeline
├── download_data.sh          | Script to download all the necessary raw data
├── env.yaml                  | Dependencies
...
```
