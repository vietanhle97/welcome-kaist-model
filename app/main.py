from flask import Flask, request, jsonify
from app.BERT import Answer

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def response():
    if request.method == 'POST':
        question = request.form['question']
        keyword = request.form['keyword']
        # Answer(question, keyword)
        try:
            data = {'response': Answer(question, keyword)}
            return jsonify(data)
        except:
            return jsonify({'error': 'Internal System Error.'})


