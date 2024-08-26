prompt_1 = (
	"I will give you a sentence and a pair of entities (Entity 1 and Entity 2).\n" 
	"- You must identify if there is a direct relation between Entity 1 and Entity 2.\n"
	"- Use ONLY the context of the sentence\n"
	"- Do not use ANY external information you could have access to\n"
	"- Answer ONLY with 1 or 0, no further text\n"
	"- Provide a brief explanation to justify your answer\n"
	'- Provide your answer as a JSON object with two fields: "relation" and "explanation"\n'
	)

prompt_2 = (
    'Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}".\n'
    'You must provide an explanation for your answer.\n'
    'Your response should be a JSON object with two fields: "relation" and "explanation".\n'
)

prompt_3 = (
    'To validate my RE corpus I need to identify if there is a relation between pre-determined entities in a sentence. '
    'Identify if there is an explicit relation between "{e1}" and "{e2}" in the following sentence: "{sentence}". '
    'Your must provide a step-by-step explanation for your answer. '
    'Your response should consist of only a JSON object with two fields: "relation" and "explanation".\n'
)

prompt_4 = (
    '###Instruction: Based on the following sentence answer the question in the specified format:\n'
    '###Sentence: "{sentence}"\n'
    '###Question: First, does a relation between "{e1}" and "{e2}" exist in the sentence? Second, explain why the relation between "{e1}" and "{e2}" exists or does not exist.\n'
    '###Format: JSON object with 2 fields: "relation" and "explanation". Example: {{"relation":int, "explanation":str}}S'
)

prompt_4_1 = (
    '###Instruction: Based on the following sentence and entities, answer the question in the specified format:\n'
    '###Sentence: "{sentence}"\n'
    '###Entities: "{e1}", "{e2}"\n'
    '###Question: First, does a relation between the entities exist in the sentence? Second, explain why the relation between the entities exists or does not exist.\n'
    '###Format: JSON object with 2 fields: "relation" and "explanation". Example: {{"relation":int, "explanation":str}}S'
)

prompt_str = (
    'Identify if there is an explicit causal relation between "{e1}" and "{e2}" in the following sentence: "{sentence}". '
    'You must also explain why the relation exists or does not exist and classify how sure you are in your prediction from 0 to 1. '
    'Your response should be a JSON object with three fields: "relation", "explanation" and "strength".\n'
)

prompt_str_2 = (
    'Identify if there is an explicit causal relation between "{e1}" and "{e2}" in the following sentence: "{sentence}". '
    'Your response should be a JSON object with three fields: "relation", "explanation" and "strength":\n'
    '-"relation": Description: 1 if there is a relation, 0 if there is not a relation; Type: int\n'
    '-"explanation": Description: explanation on why there is or there is not a relation; Type: str\n'
    '-"strength": Description: the strenght of the prediction from 0 to 1; Type: float'    
)

prompt_u = (
    'Do you think the following sentence conveys information about the relation between "{e1}" and "{e2}" in a clear way? Sentence: "{sentence}"\n'
    'You must also explain the reasoning behind your answer\n'
    'Your response should be a JSON object with two fields: "answer", "explanation".\n'
)

prompt_u_2 = (
    'Analyse the following sentence: "{sentence}". Determine if the sentence is useful for understanding the relation between the two entities: "{e1}" and "{e2}".\n'
    'You must explain the reasoning behind your answer, considering aspects such as context, clarity and relevance of the information provided.\n'
    'Your response should be a JSON object with two fields: "answer" and "explanation".\n'
)