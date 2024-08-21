#!/bin/bash

# Data Download Script
# David da Costa Correia @ FCUL & INSA
# Last updated on 15/05/2024

original_dir=$PWD
download_dir='./data/'

rm -r $download_dir
mkdir -p $download_dir
cd $download_dir
echo "Downloading data to $PWD"

# RNA IDs
wget -O ./RNACentral_mapping.tsv.gz "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/id_mapping/id_mapping.tsv.gz"
zcat ./RNACentral_mapping.tsv.gz | awk -F'\t' '$4 == 9606 {print $1, $3, $6}' OFS='\t' > ./RNACentral_mapping_human.csv

# RNA Aliases
wget -O ./hgnc_rna_aliases.txt "https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=gd_prev_sym&col=gd_aliases&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&where=gd_locus_group%20=%20%27non-coding%20RNA%27&submit=submit"

# HPO Ontologies
wget "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2024-04-26/hp.obo" 

# Databases
# 	ncrPheno
wget "http://liwzlab.ifr.fidt.top:61010/ncrpheno/static/ncRPheDB_Data/Evidence_information.txt" 
# 	RIscoper
wget -O ./All-RNA-Disease_NER.txt "http://www.rnainter.org/riscoper/static/media/allRDI.73edab62d856407ab62f.txt"
wget -O ./All-RNA-Disease_Sentence.txt "http://www.rnainter.org/riscoper/static/media/All%20RNA-Disease_supplementarylower_datasource.cc26aff845c3e634e937.txt"
#	HMDD
wget -O ./hmdd_data_v4.txt "https://www.cuilab.cn/static/hmdd3/data/alldata_v4.txt" --no-check-certificate
#	lncRNA-Disease
wget -O ./lncDD_causal_data.tsv "http://www.rnanut.net/lncrnadisease/static/download/website_causal_data.tsv"
#   RNADisease
wget "http://www.rnadisease.org/static/download/RNADiseasev4.0_RNA-disease_experiment_all.zip"
unzip RNADiseasev4.0_RNA-disease_experiment_all.zip
rm RNADiseasev4.0_RNA-disease_experiment_all.zip

cd $original_dir