import torch
import pandas as pd

import json
import os 
import time
d = os.getcwd()
# from transformers import BertForQuestionAnswering
# from transformers import BertTokenizer
# model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# # Tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
name = "mrm8488/bert-small-finetuned-squadv2"

tokenizer = AutoTokenizer.from_pretrained(name,)

model = AutoModelForQuestionAnswering.from_pretrained(name)

def matchParaAnswer(keyword):
	topic = ['contact','application timeline','application process','application fees','selection process',
			 'required document', 'kaist']
	with open(d + '/app/topKeywords.json') as f:
		topKeywords = json.load(f)
	value = 0
	index = -1
	# print(topKeywords)
	for i in range(len(topKeywords)):
		if keyword in topKeywords[i]:
			value = max(topKeywords[i][keyword],value)
			index = i
	# print out topic 
	# print(topic[index])
	return index

def getAnswer(question, paragraph):
	encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)
	inputs = encoding['input_ids']  #Token embeddings
	sentence_embedding = encoding['token_type_ids']  #Segment embeddings
	tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

	start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
	start_index = torch.argmax(start_scores)
	end_index = torch.argmax(end_scores)
	answer = ' '.join(tokens[start_index:end_index+1])
	corrected_answer = ''

	for word in answer.split():
		if word[0:2] == '##':
			corrected_answer += word[2:]
		else:
			corrected_answer += ' ' + word
	corrected_answer = corrected_answer.replace(" , ", ",")
	corrected_answer = corrected_answer.replace(" . ", ".")
	corrected_answer = corrected_answer.replace(" @ ", "@")
	corrected_answer = corrected_answer.replace(" - ", "-")
	return corrected_answer

def Answer(question, keyword):
	# find the paragraph contain the answer
	keyword = keyword.lower()
	indexPara = matchParaAnswer(keyword)
	if indexPara == -1:
		return "Cannot find the answer! Please contact us"
	else:
		# load data
		data = pd.read_csv(d + "/app/data.csv")
		# retrieve the paragraph contains answer
		paragraph = data.iloc[indexPara]['DETAIL']
		# paragraph[:100] used when we do not have a great computer
		answer = getAnswer(question, paragraph[:100])
		# answer = getAnswer(question, paragraph model, tokenizer)
		return answer
