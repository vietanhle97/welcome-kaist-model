from flask import Flask, request, jsonify
import json
from app.BERT import Answer

app = Flask(__name__)

@app.route('/answer', methods=['POST'])
def response():
    if request.method == 'POST':
        question = request.json['question']
        keyword = request.json['keyword']
        try:
        	data = Answer(question, keyword)
        	response = {'response': data['answer'], 'topic':data['topic'], 'info':data['info']}
        	return jsonify(response)
        except:
        	return jsonify({'error': 'Internal System Error.'})


