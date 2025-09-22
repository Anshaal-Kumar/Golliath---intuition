from flask import Flask, request, jsonify
from io import StringIO
import pandas as pd
import traceback

# Assuming you have already imported or initialized these
# from digital_twin_core import DataIngestionEngine, AIProcessingEngine, OntologyEngine

app = Flask(__name__)

# Initialize engines
ai_engine = AIProcessingEngine()
ontology_engine = OntologyEngine()
current_data = None

@app.route('/api/data/ingest', methods=['POST'])
def ingest_data():
    """Ingest CSV data from frontend"""
    global current_data

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        csv_content = file.read()

        # Define CSV parsing strategies
        parsing_strategies = [
            {'sep': ',', 'engine': 'python', 'on_bad_lines': 'warn'},
            {'sep': ';', 'engine': 'python', 'on_bad_lines': 'warn'},
            {'sep': '\t', 'engine': 'python', 'on_bad_lines': 'warn'},
            {'sep': None, 'engine': 'python', 'on_bad_lines': 'warn'},
            {'sep': ',', 'engine': 'python', 'on_bad_lines': 'skip'}
        ]

        # Try parsing with multiple strategies
        current_data = None
        for i, strategy in enumerate(parsing_strategies):
            try:
                print(f"Trying parsing strategy {i+1}: {strategy}")
                
                # Ensure UTF-8 safe decoding
                if isinstance(csv_content, bytes):
                    csv_str = csv_content.decode('utf-8', errors='ignore')
                else:
                    csv_str = str(csv_content)

                current_data = pd.read_csv(StringIO(csv_str), encoding='utf-8', **strategy)

                if len(current_data) > 0 and len(current_data.columns) > 0:
                    print(f"✅ Success with strategy {i+1}. Shape: {current_data.shape}")
                    break
            except Exception as e:
                print(f"❌ Strategy {i+1} failed: {str(e)}")
                continue

        if current_data is None or len(current_data) == 0:
            return jsonify({'error': 'Could not parse CSV file. Please check the format.'}), 400

        # Clean columns
        current_data.columns = [col.strip().replace('\n', '').replace('\r', '') for col in current_data.columns]

        # Sanitize string data to remove bad characters
        current_data = current_data.applymap(lambda x: str(x).encode('utf-8', errors='ignore').decode('utf-8') 
                                             if isinstance(x, str) else x)

        # Process with AI
        processed_data = ai_engine.clean_and_process(current_data)
        current_insights = ai_engine.extract_patterns(processed_data)

        # Build knowledge graph
        ontology_engine.build_knowledge_graph(processed_data, current_insights)

        return jsonify({
            'success': True,
            'message': f'Successfully ingested {len(current_data)} rows',
            'rows': len(current_data),
            'columns': len(current_data.columns),
            'column_names': current_data.columns.tolist()
        })

    except Exception as e:
        return jsonify({
            'error': f'Data ingestion failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/data/current', methods=['GET'])
def get_current_data():
    """Get current dataset info"""
    global current_data

    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 404

        return jsonify({
            'shape': current_data.shape,
            'columns': current_data.columns.tolist(),
            'dtypes': current_data.dtypes.astype(str).to_dict(),
            'sample': current_data.head(5).to_dict('records'),
            'statistics': current_data.describe().to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8501)
