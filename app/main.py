from flask import Flask, request, jsonify
import json
from app.BERT import Answer

app = Flask(__name__)

@app.route('/answer', methods=['POST'])
def response():
    if request.method == 'POST':
        question = request.json['question']
        keyword = request.json['keyword']
        Answer(question, keyword)
        try:
            data = {'response': Answer(question, keyword)}
            return jsonify(data)
        except:
            return jsonify({'error': 'Internal System Error.'})


