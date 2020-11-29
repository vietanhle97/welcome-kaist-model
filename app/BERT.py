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
info = "https://admission.kaist.ac.kr/intl-undergraduate/wp-content/uploads/sites/4/2020/11/Admission-Guide-for-international-applicants-2021_updated1119.pdf"
email = "creative.adm@kaist.ac.kr"

links = {"application fees": {"Application Fees":"https://admission.kaist.ac.kr/intl-undergraduate/faq/?category1=Application%20Fee&mod=list&pageid=1"},
"required document": {"Recommendation Letter":"https://admission.kaist.ac.kr/intl-undergraduate/faq/?category1=Recommendation%20Letter&mod=list&pageid=1",
"School Profile":"https://admission.kaist.ac.kr/intl-undergraduate/faq/?category1=School%20Profile&mod=list&pageid=1",
"Transcript":"https://admission.kaist.ac.kr/intl-undergraduate/faq/?category1=Transcript&mod=list&pageid=1",
"Standardize Official Test": "https://admission.kaist.ac.kr/intl-undergraduate/faq/?category1=Standardized%20Official%20Test&mod=list&pageid=1",
"English Proficiency Test Report":"https://admission.kaist.ac.kr/intl-undergraduate/faq/?category1=English%20Proficiency%20Test%20Report&mod=list&pageid=1"},
"application timeline": {"Application Timeline" :"https://admission.kaist.ac.kr/intl-undergraduate/eligibility/"},
"application process": {"Application Process": "https://admission.kaist.ac.kr/intl-undergraduate/wp-content/uploads/sites/4/2020/11/Admission-Guide-for-international-applicants-2021_updated1119.pdf"},
"selection process": {"Selection Process": "https://admission.kaist.ac.kr/intl-undergraduate/wp-content/uploads/sites/4/2020/11/Admission-Guide-for-international-applicants-2021_updated1119.pdf"},
"contact": {"Contact": "https://www.kaist.ac.kr/en/html/footer/0802.html"},
"kaist": {"About KAIST": "https://kaist.ac.kr/en/"}}

words = {'fees': 'application fees', 'admission deadline': 'application timeline', 'admission due': 'application timeline', 'deadline admission': 'application timeline', 'due admission': 'application timeline', 'letter': 'required document', 'sat1': 'required document', 'score': 'required document', 'transcript': 'required document', 'credit card': 'application fees', 'sat2': 'required document', 'ap': 'required document', 'ib': 'required document', 'gce': 'required document', 'act': 'required document', 'toefl': 'required document', 'ielts': 'required document', 'toeic': 'required document', 'phone number': 'contact', 'email': 'contact', 'select': 'selection process'}

topic = ['contact','application timeline','application process','application fees','selection process',
			 'required document', 'kaist']

def matchParaAnswer(keyword):
	if len(keyword) == 0:
		return -1
	
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
	if index == -1:
		for k in words:
			if keyword in k or k in keyword:
				return topic.index(words[k])
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
		return {"answer": "Cannot find the answer! Please contact " + email, "topic":"", "info":{"Info": info}}
	else:
		# load data
		data = pd.read_csv(d + "/app/data.csv")
		# retrieve the paragraph contains answer
		paragraph = data.iloc[indexPara]['DETAIL']
		# paragraph[:100] used when we do not have a great computer
		answer = getAnswer(question, paragraph[:100])
		# answer = getAnswer(question, paragraph model, tokenizer)
		if len(answer) == 0 or 'CLS' in answer:
			return {"answer": "Cannot find the answer! Please contact " + email, "topic":topic[indexPara], "info":links[topic[indexPara]]}
		return {"answer":answer, "topic": topic[indexPara], "info":links[topic[indexPara]]}

