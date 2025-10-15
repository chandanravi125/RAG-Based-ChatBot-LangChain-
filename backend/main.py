from flask import Flask, request, jsonify
from Rag_agent import create_rag_agent
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

try:
    rag_agent = create_rag_agent()
except Exception as e:
    logging.error(f"Failed to initialize RAG agent: {e}")
    rag_agent = None

@app.route('/ask', methods=['POST'])
def ask():
    """ Flask endpoint for user queries """
    if rag_agent is None:
        return jsonify({'error': 'RAG agent not initialized properly'}), 500

    try:
        query = request.json.get('query')
        if not query or not query.strip():
            return jsonify({'error': 'Query is required'}), 400

        logging.info(f"Received query: {query}")
        response = rag_agent.answer(query)
        logging.info(f"Response: {response}")

        return jsonify({'response': response})

    except Exception as e:
        logging.error(f"Error while answering: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == "__main__":
    app.run(debug=True)
