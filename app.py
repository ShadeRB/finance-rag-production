from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from rag_pipeline import load_and_chunk_pdf, create_vector_store, query_rag
import os

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Vector store init
pdf_path = 'data/sample_10k.pdf'
vector_store = None

if os.path.exists(pdf_path):
    try:
        chunks = load_and_chunk_pdf(pdf_path)
        vector_store = create_vector_store(chunks)
        print("[INFO] Vector store created successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load and process PDF: {e}")
else:
    print(f"[WARNING] PDF not found at: {pdf_path}")

@app.route('/')
def home():
    return 'Finance RAG app is running! Use POST /query to ask questions.'

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if not vector_store:
        return jsonify({"error": "Vector store not initialized"}), 500

    try:
        answer, sources = query_rag(question, vector_store)
        return jsonify({
            "answer": answer,
            "sources": [doc.page_content for doc in sources]
        })
    except Exception as e:
        print(f"[ERROR] Exception in /query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
