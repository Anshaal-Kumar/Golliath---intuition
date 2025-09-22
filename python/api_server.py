#!/usr/bin/env python3
# python/api_server.py - Complete Python API Server
import argparse
import sys
import os
import json
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import StringIO
import io

# 🔧 Force stdout/stderr to UTF-8 (fixes Windows 'charmap' codec errors with emojis/logs)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# Import our core engines
from digital_twin_core import DataIngestionEngine, AIProcessingEngine, OntologyEngine, SimulationEngine

def analyze_csv_structure(csv_content):
    """Analyze CSV structure for debugging"""
    lines = csv_content.split('\n')[:10]  # First 10 lines
    analysis = {
        'total_lines': len(csv_content.split('\n')),
        'first_10_lines': lines,
        'separators_found': {}
    }
    for sep in [',', ';', '\t', '|']:
        counts = [line.count(sep) for line in lines if line.strip()]
        if counts:
            analysis['separators_found'][sep] = {
                'avg_count': sum(counts) / len(counts),
                'consistent': len(set(counts)) <= 2
            }
    return analysis

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize engines
data_engine = DataIngestionEngine()
ai_engine = AIProcessingEngine()
ontology_engine = OntologyEngine()
simulation_engine = SimulationEngine()

# --------------------------
# Helper: Safe convert for JSON serialization
# --------------------------
def safe_convert(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.ndarray, list)):
        return [safe_convert(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    return obj

# --------------------------
# Global state
# --------------------------
current_data = None
current_insights = None

# --------------------------
# Routes
# --------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/data/ingest', methods=['POST'])
def ingest_data():
    global current_data, current_insights

    try:
        data = request.get_json()
        csv_content = data.get('csv_content')
        if not csv_content:
            return jsonify({'error': 'No CSV content provided'}), 400

        # DEBUG: CSV structure analysis
        csv_analysis = analyze_csv_structure(csv_content)
        print("📊 CSV Analysis:", json.dumps(csv_analysis, indent=2))

        # --- CSV parsing strategies ---
        current_data = None
        parsing_strategies = [
            {'sep': ',', 'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'},
            {'sep': ';', 'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'},
            {'sep': '\t', 'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'},
        ]

        for i, strategy in enumerate(parsing_strategies):
            try:
                print(f"Trying parsing strategy {i+1}: {strategy}")
                current_data = pd.read_csv(StringIO(csv_content), **strategy)
                if len(current_data) > 0 and len(current_data.columns) > 0:
                    print(f"✅ Success with strategy {i+1}. Shape: {current_data.shape}")
                    break
            except Exception as e:
                print(f"❌ Strategy {i+1} failed: {e}")
                continue

        if current_data is None or len(current_data) == 0:
            return jsonify({'error': 'Could not parse CSV file. Please check the format.'}), 400

        # Clean columns and remove empty rows
        current_data.columns = [col.strip().replace('\n','').replace('\r','') for col in current_data.columns]
        current_data = current_data.dropna(how='all')

        # Sanitize string cells
        current_data = current_data.applymap(
            lambda x: str(x).encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else x
        )

        # AI processing
        processed_data = ai_engine.clean_and_process(current_data)
        current_insights = ai_engine.extract_patterns(processed_data)

        # Build knowledge graph
        ontology_engine.build_knowledge_graph(processed_data, current_insights)

       # Inside ingest_data(), replace the return statement at the end with:

    return jsonify(safe_convert({
        'success': True,
        'message': f'Successfully ingested {len(current_data)} rows',
        'rows': len(current_data),
        'columns': len(current_data.columns),
        'column_names': current_data.columns.tolist(),
        'insights': current_insights  # safely convert any NumPy types
    }))


    except Exception as e:
        return jsonify({
            'error': f'Data ingestion failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/data/current', methods=['GET'])
def get_current_data():
    global current_data
    
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 404
        
        response = {
            'shape': tuple(map(int, current_data.shape)),
            'columns': current_data.columns.tolist(),
            'dtypes': current_data.dtypes.astype(str).to_dict(),
            'sample': current_data.head(5).to_dict('records'),
            'statistics': current_data.describe().to_dict()
        }
        return jsonify(safe_convert(response))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description='Digital Twin Intelligence API Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8501, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting Digital Twin Intelligence API Server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Server running at: http://{args.host}:{args.port}")
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
