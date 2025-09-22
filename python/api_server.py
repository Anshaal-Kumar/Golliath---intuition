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
import base64

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
    
    # Check for different separators
    for sep in [',', ';', '\t', '|']:
        counts = [line.count(sep) for line in lines if line.strip()]
        if counts:
            analysis['separators_found'][sep] = {
                'avg_count': sum(counts) / len(counts),
                'consistent': len(set(counts)) <= 2  # Allow for header difference
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

# Global state
current_data = None
current_insights = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/data/ingest', methods=['POST'])
def ingest_data():
    """Ingest CSV data"""
    global current_data, current_insights

    try:
        data = request.get_json()
        csv_content = data.get('csv_content')
        source_name = data.get('source_name', 'unknown')

        if not csv_content:
            return jsonify({'error': 'No CSV content provided'}), 400

        # Debugging CSV structure
        csv_analysis = analyze_csv_structure(csv_content)
        print("📊 CSV Analysis:", json.dumps(csv_analysis, indent=2))

        # Try multiple parsing strategies
        parsing_strategies = [
            {'sep': ',', 'engine': 'python'},
            {'sep': ';', 'engine': 'python'},
            {'sep': '\t', 'engine': 'python'},
            {'sep': ',', 'engine': 'python', 'quotechar': '"'},
            {'sep': ',', 'engine': 'python', 'skipinitialspace': True},
        ]

        current_data = None
        for i, strategy in enumerate(parsing_strategies):
            try:
                print(f"Trying parsing strategy {i+1}: {strategy}")
                current_data = pd.read_csv(StringIO(csv_content), **strategy)
                if len(current_data) > 0 and len(current_data.columns) > 0:
                    print(f"✅ Success with strategy {i+1}. Shape: {current_data.shape}")
                    break
            except Exception as e:
                print(f"❌ Strategy {i+1} failed: {str(e)}")
                continue

        if current_data is None or len(current_data) == 0:
            return jsonify({'error': 'Could not parse CSV file. Please check the format.'}), 400

        # Clean columns and remove empty rows
        current_data.columns = [col.strip().replace('\n', '').replace('\r', '') for col in current_data.columns]
        current_data = current_data.dropna(how='all')

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


# -----------------------
# Other endpoints remain unchanged
# -----------------------

# (keep your other routes: analyze_patterns, detect_anomalies, get_knowledge_graph, query_ontology,
# build_model, generate_forecast, run_what_if, assess_risk, chat_query, system_status, etc.)

def main():
    """Start the API server"""
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
