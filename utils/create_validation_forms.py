''' 
Validation Forms Creation Script
David da Costa Correia @ FCUL & INSA
'''

### SETTINGS ##########################################################
CORPUS_FOLDER = '../outputs/corpus.zip' # Or zip - zip is better performance
# Sampling
N_SAMPLES = 100
SAMPLE_SIZE = 40
OVERLAP_SIZE = 20
# Forms 
FORMS_FILE = '../outputs/validation/form_responses.json'
#######################################################################

import copy
import zipfile
import os
import json
import random
from datetime import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

PUBSUB_ID = 'projects/stately-transit-421114/topics/submitted_forms'

FORM_TEMPLATE = {"info": {"title": "Corpus Validation"}}

QUESTION_OPTIONS_TEMPLATE = ['Correct','Incorrect','Uncertain']

DESCRIPTION_TEMPLATE = (
    "IMPORTANT NOTE: This form is unique and was assigned to you. Please don't share its link with anyone else.\n\n"
    "Once again, I am very thankful for your assistance. As a reminder, your task will involve validating 40 ncRNA-phenotype relations. This takes about 20 minutes to complete.\n\n"
    "If, for any reason you need to submit more than one response, only your final response will be taken in consideration.\n\n"
    "If you wish to proceed, please continue to the next page."
)

INSTRUCTIONS_TEMPLATE = """The detailed instructions for this task are presented below. Please read them attentively.

1. Each relation follows the following format:

Sentence: "This is an example: ncRNA1 is related to Phenotype1" <-- The sentence containing the relation to be validated

Entity 1: ncRNA1 (ncRNA | URL: <RNACentral URL>)

Entity 2: Phenotype1 (Phenotype | URL: <Human Phenotype Ontology URL>)

Predicted Relation*: Yes <-- The predicted relation between the two entities (Entity 1 and Entity 2)

Evaluation: (select below) <-- Your evaluation (see point 2)

*Predicted Relation can take two values: "Yes" or "No":
    - "Yes": A relation (positive/activation or negative/inhibition) between the two entities was predicted.
    - "No": No relation between the two entities was predicted.

2. To validate each relation you must assign one of three possible evaluations: Correct, Incorrect or Uncertain

- Correct: when the predicted relation between the two entities is correct.
    Example 1: "ncRNA1 is related to Phenotype1." Predicted Relation: Yes
    Example 2: "ncRNA1 leads to inhibited Phenotype1." Predicted Relation: Yes
    Example 3:
        Sentence: "These implied that loss of BLACAT1 restrained the progression of non-small cell lung cancer.."
        Entity 1: BLACAT1 (ncRNA | URL: https://rnacentral.org/rna/URS000234849C/9606)
        Entity 2: non-small cell lung cancer (Phenotype | URL: https://hpo.jax.org/app/browse/term/HP:0030358)
        Predicted Relation: Yes

- Incorrect: when the predicted relation between the two entities is incorrect.
    Example 1: "ncRNA1 effects on Phenotype1 were studied." Predicted Relation: Yes
    Example 2:
        Sentence: "Thus, miR-93 may play an important part in other IR-related diseases, such as obesity and T2DM."
        Entity 1: miR-93 (ncRNA | URL: https://rnacentral.org/rna/URS00000DDD35/9606)
        Entity 2: obesity (Phenotype | URL: https://hpo.jax.org/app/browse/term/HP:0001513)
        Predicted Relation: No

- Uncertain: should be used when it is hard to comprehend the meaning of a sentence, or other reasons that could make a relation impossible to label as Correct or Incorrect.
    Example 1:
        Sentence: "It is worth noting that EGFR-AS1 is predominantly localized in the cytoplasm."
        Entity 1: EGFR-AS1 (ncRNA | URL: https://rnacentral.org/rna/URS000075EB15/9606)
        Entity 2: localized (Phenotype | URL: https://hpo.jax.org/app/browse/term/HP:0012838)
        Predicted Relation: No
    Note: the identified phenotype "localized" does not have any clinical significance, as so, the relation is Uncertain.
    Example 2:
        Sentence: "Altogether, SAMMSON interacts with EZH2 in liver TICs.."
        Entity 1: SAMMSON (ncRNA | URL: https://rnacentral.org/rna/URS00026A1F9C/9606)
        Entity 2: TICs (Phenotype | URL: https://hpo.jax.org/app/browse/term/HP:0100033)
        Predicted Relation: Yes
    Note: Despite "Tics" being a phenotype with clinical significance, by reading the sentence (and using the URL) we can understand that the "TICs" mentioned in the sentence were wrongly linked to the phenotype "Tics". As so, the relation is Uncertain.
    """

FIRST_5_TEMPLATE = """These are the correct evaluations for the first 5 relations. Please review your answers to confirm you understood the instructions, but do not change your original answers.

Relation 1

Sentence: "Among them, DKK1 (dickkopf WNT signaling pathway inhibitor 1) is found to be regulated by NBAT1 in a PRC2 dependent manner, and is responsible for NBAT1's effects in inhibiting migration and invasion of breast cancer cells."

Entity 1: NBAT1 (ncRNA | URL: https://rnacentral.org/rna/URS0002207646/9606)

Entity 2: cancer (Phenotype | URL: https://hpo.jax.org/app/browse/term/HP:0002664)

Predicted Relation: Yes

Evaluation: Correct (It's clear that NBAT1 inhibits migration and invasion of cancer cells, so there is a relation)


Relation 2

Sentence: "Moreover, we discovered that HULC promotes breast cancer cells’ proliferation, migration and invasion by sponging miR-6754-5p."

Entity 1: HULC (ncRNA | URL: https://rnacentral.org/rna/URS00026A1D8A/9606)

Entity 2: cancer (Phenotype | URL: https://hpo.jax.org/app/browse/term/HP:0002664)

Predicted Relation: Yes

Evaluation: Correct (HULC promotes cancer proliferation, so there is a relation)


Relation 3

Sentence: "This assertion is based on several observations in HULC and/ot MALAT1 overexpresed liver cancer stem cells: (1) Upregulated lncRNA MALAT1/HULC were positively associated with the TRF2 expression in human liver cancer tissues."

Entity 1: MALAT1 (ncRNA | URL: https://rnacentral.org/rna/URS00025F0023/9606)

Entity 2: cancer (Phenotype | URL: https://hpo.jax.org/app/browse/term/HP:0002664)

Predicted Relation: Yes

Evaluation: Correct (MALAT1 is associated with TRF2 expression in cancer cells, so there is a relation)


Relation 4

Sentence: "Expression and clinical prognostic role of CARD8-AS1 in lung adenocarcinoma."

Entity 1: CARD8-AS1 (ncRNA | URL: https://rnacentral.org/rna/URS000251E64A/9606)

Entity 2: lung adenocarcinoma (Phenotype | URL: https://hpo.jax.org/app/browse/term/HP:0030078)

Predicted Relation: Yes

Evaluation: Uncertain (The sentence seems to be incomplete, making it hard to judge its meaning)


Relation 5

Sentence: "The association between lncRNA SNHG4 expression and clinicopathologic characteristics and prognosis in patients with osteosarcoma was analysed by TCGA RNA‐sequencing data."

Entity 1: SNHG4 (ncRNA | URL: https://rnacentral.org/rna/URS0002617A23/9606)

Entity 2: osteosarcoma (Phenotype | URL: https://hpo.jax.org/app/browse/term/HP:0002669)

Predicted Relation: Yes

Evaluation: Incorrect (The expression and characteristics of SNHG4 in osteosarcoma patients was merely analysed, not necessarily indicating a relation)


Thank you, you will not be asked to review any other answer. Please proceed with the validation.
"""

def form_description_update_request(description:str) -> dict:
    ":description: tuple of paragraphs of the lines of the description"
    request = {"updateFormInfo": {"info":{"description":description},
                                  "updateMask":"description"}}
    return request


def form_add_question_request(loc, question_i, text, options) -> dict:
    ":question_i: number of the question"
    ":text: text of the question"
    ":options: answer options"
    question = {"question":{"questionId":str(question_i),
                            "required":True,
                            "choiceQuestion":{"type":"RADIO",
                                              "options":[{"value":option} for option in options]}}}
    request = {"createItem":{"item":{"title":f"Relation {question_i}",
                                     "description":text,
                                     "questionItem":question},
                             "location":{"index":loc}}}
    return request


def form_add_pagebreak_request(i, title, text):
    request = {"createItem":{"item":{"title":title,
                                     "description":text,
                                     "pageBreakItem":{}},
                             "location":{"index":i}}}
    return request


def form_watch_request(pubsub_id):
    request = {"watch":{"target":{"topic":{"topicName":pubsub_id}},
                        "eventType":"RESPONSES"}}
    return request


def load_corpus(corpus_dir:os.PathLike) -> dict:
    zipped = corpus_dir.endswith('.zip')
    if zipped:
        zip_dir = zipfile.ZipFile(corpus_dir)
        files = zip_dir.namelist()
    else:
        files = []
        for root,_,fils in os.walk(corpus_dir):
            for file in fils:
                files.append(os.path.join(root,file))

    corpus = {}
    errors = []
    for file in files:
        if file.endswith('.log'):
            continue
        
        pmid = int(os.path.splitext(file)[0].split('/')[-1])
        corpus[pmid] = None
        
        try:
            if zipped: 
                f = zip_dir.open(file)
            else:
                f = open(file, 'r')
            corpus[pmid] = json.load(f)
            f.close()
        except Exception as e:
            errors.append(pmid)
    
    if zipped: zip_dir.close()
    
    return corpus, errors


def get_relations(corpus):
    pos = []
    neg = []
    for pmid,sents in corpus.items():
        if sents is None:
            continue
        for sent in sents:
            sentence = sent["sentence"]
            rels = sent["relations"]
            for rel in rels:
                if rel["relation"] == 0:
                    neg.append((sentence, rel))
                else:
                    pos.append((sentence, rel))
    return pos, neg


def sample_relations(relations:tuple, n_samples:int, size:int, overlap:int, pos_ratio:float, seed=None):
    # PROBLEM: Only works with even numbers
    random.seed(seed)

    pos_rels, neg_rels = relations[0].copy(), relations[1].copy()
    
    pos_overlap_sample = random.sample(pos_rels, round(overlap*pos_ratio))
    neg_overlap_sample = random.sample(neg_rels, round(overlap*(1-pos_ratio)))

    for rel in pos_overlap_sample:
        pos_rels.remove(rel)
    for rel in neg_overlap_sample:
        neg_rels.remove(rel)

    overlap_sample = pos_overlap_sample+neg_overlap_sample
   
    samples = [[] for _ in range(n_samples)]
    for sample in samples:
        sample += overlap_sample
        
        pos_sample = random.sample(pos_rels, round((size-overlap)*pos_ratio))
        sample += pos_sample
        for rel in pos_sample:
            pos_rels.remove(rel)
        
        neg_sample = random.sample(neg_rels, round((size-overlap)*(1-pos_ratio)))
        sample += neg_sample
        for rel in neg_sample:
            neg_rels.remove(rel)

    return samples


def get_entity_url(entity_id:str):
    url = 'Not available'
    if entity_id.startswith('HP'):
        url = f'https://hpo.jax.org/app/browse/term/{entity_id}'  
    elif entity_id.startswith('URS'):
        url = f'https://rnacentral.org/rna/{entity_id}/9606'        
    return url


def write_relation(relation) -> str:
    sentence, rel = relation
    label = 'Yes' if bool(rel['relation']) else 'No'
    text = (
        f'Sentence: "{sentence}"\n\n'
        f"Entity 1: {rel['e1']['text']} ({rel['e1']['type']} | URL: {get_entity_url(rel['e1']['ID'])})\n\n"
        f"Entity 2: {rel['e2']['text']} ({rel['e2']['type']} | URL: {get_entity_url(rel['e2']['ID'])})\n\n"
        f"Predicted Relation: {label}\n\n"
        f'Evaluation: (select below)'
    )
    return text


def write_forms(samples, forms_service, forms_file):
    forms = {} # {form_id:{'relations':list,'responses':list}}
    # first_5 = [] # DEBUG
    for i,sample in enumerate(samples,1):
        # Create form from template
        form = copy.deepcopy(FORM_TEMPLATE)
        # form['info']['title'] = f'Corpus Validation {i}' # DEBUG
        form['info']['documentTitle'] = f"corpus_validation_{i}"
        created_form = forms_service.forms().create(body=form).execute()
        form_id = created_form['formId']
        forms[form_id] = {'relations':None, 'responses':None}

        # Write form description, sections and questions requests
        requests = []
        forms[form_id]['relations'] = sample
        requests.append(form_description_update_request(DESCRIPTION_TEMPLATE))
        # First section
        requests.append(form_add_pagebreak_request(0, "Corpus Validation", INSTRUCTIONS_TEMPLATE))
        for j,rel in enumerate(sample[0:5], 1):
            text = write_relation(rel)
            # first_5.append(text) # DEBUG
            requests.append(form_add_question_request(j, j, text, QUESTION_OPTIONS_TEMPLATE))
        # Review first 5
        # requests.append(form_add_pagebreak_request(6, "Please review your first 5 answers", "\n".join(first_5))) # DEBUG
        requests.append(form_add_pagebreak_request(6, "Please review your first 5 answers", FIRST_5_TEMPLATE))
        requests.append(form_add_pagebreak_request(7, "Corpus Validation", INSTRUCTIONS_TEMPLATE))
        # Second section
        for j,rel in enumerate(sample[5:], 8):
            text = write_relation(rel)
            requests.append(form_add_question_request(j, j-2, text, QUESTION_OPTIONS_TEMPLATE))

        # Execute requests
        update_requests = {"requests":requests}
        forms_service.forms().batchUpdate(formId=form_id, body=update_requests).execute()
    
    # Write output file
    with open(forms_file, "w") as f:
        json.dump(forms, f, indent=1)
        
    return forms
    

# def set_watches(forms_service, forms_ids_file):
#     with open(forms_ids_file, 'r') as f:
#         form_ids = f.readlines()
    
#     watches = []
#     for form_id in form_ids:
#         form_id = form_id.strip('\n')
#         form_watches = forms_service.forms().watches().list(formId=form_id).execute()
#         form_watches = form_watches.get("watches")
#         if form_watches: # if watches exists
#             watch = form_watches[0]
#             if watch["state"] == 'SUSPENDED': # if the watch is suspended, renew it
#                 forms_service.forms.watches().renew(formId=form_id, watchId=watch["watchId"]).execute()
#         else: # if the watch doesn't exist, create it
#             watch_request = form_watch_request(PUBSUB_ID)
#             watch = forms_service.forms().watches().create(formId=form_id, body=watch_request).execute()
#         watches.append(watch)

#     return watches


def get_responses(forms_service, forms_file):
    """Fetch responses for the forms in forms_file, returns (number of responses, number of new responses)"""
    with open(forms_file, 'r') as f:
        forms = json.load(f)
    
    n_responses = 0
    new_responses = 0
    for form_id,form in forms.items():
        if form['responses'] is not None: 
            n_responses += 1
            new_responses -= 1

        form_responses = forms_service.forms().responses().list(formId=form_id).execute()
        form_responses = form_responses.get("responses")
        if form_responses is None: 
            continue # if no responses, go to next form
        
        # Sort responses by the time they were created
        form_responses = sorted(form_responses, key=lambda x: datetime.strptime(x['createTime'], '%Y-%m-%dT%H:%M:%S.%fZ'))

        raw_answers = form_responses[-1]["answers"] # Get the last submitted response  
        answers = [[] for _ in range(len(raw_answers))]
        for i in range(1,len(answers)+1):
            answers[i-1] = raw_answers[f"{i:0{8}d}"]["textAnswers"]["answers"][0]["value"]
            
        forms[form_id]['responses'] = answers
        new_responses += 1

    with open(forms_file, 'w') as f:
        json.dump(forms, f, indent=1)

    return n_responses+new_responses, new_responses


def main():

    SCOPES = ["https://www.googleapis.com/auth/drive"]

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # if no credentials, prompt log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
            # save the credentials
            with open("token.json", "w") as token:
                token.write(creds.to_json())

    forms_service = build("forms", "v1", credentials=creds)

    if not os.path.exists(FORMS_FILE):
        corpus, errors = load_corpus(CORPUS_FOLDER)
        relations = get_relations(corpus)
        samples = sample_relations(relations, N_SAMPLES, SAMPLE_SIZE, OVERLAP_SIZE, 0.5, 1)
        forms = write_forms(samples, forms_service, FORMS_FILE)
    else:
        print(f"Forms already created, saved in: {FORMS_FILE}.")

    n_responses, n_new_responses = get_responses(forms_service, FORMS_FILE)
    print(f"Responses updated. Total responses: {n_responses} ({n_new_responses} new)")
    # watches = set_watches(forms_service, FORM_IDS_FILE)


if __name__ == '__main__':
    main()