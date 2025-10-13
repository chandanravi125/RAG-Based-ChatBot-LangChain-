# Error handling: where the create_rag_agent function fails or the rag_agent.answer method raises an exception.
# Input validation: to ensure Query is not empty and meets certain criteria.
# Response formatting: 
# logging: to track requests and responses, which can be helpful for debugging and monitoring
from flask import Flask, request, jsonify
from Rag_agent import RAGAgent
import logging
import os

app = Flask(__name__)

# Initialize logging: 
logging.basicConfig(level=logging.INFO)

def create_app():
    """
    Creates and configures the Flask application.

    Returns:
        Flask: The configured Flask application.
    """
    try:
        # Initialize the RAG agent
        open_API_key = os.getenv("OPENAI_API_KEY")
        if not open_API_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        rag_agent = RAGAgent(open_API_key)
        return rag_agent
    except Exception as e:
        logging.error(f"Failed to initialize RAG agent: {str(e)}")
        exit(1)

rag_agent = create_app()

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handles POST requests to the /ask endpoint.

    Expects a JSON body with a 'query' parameter.

    Returns:
        JSON: A JSON response containing the answer to the query.

    Raises:
        400: If the query is missing or empty.
        500: If an internal server error occurs.
    """
    try:
        query = request.json.get('query')
        if query is None or not query.strip():
            return jsonify({'error': 'Query is required'}), 400

        logging.info(f"Received query: {query}")
        response = rag_agent.answer(query)
        logging.info(f"Generated response: {response}")

        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)