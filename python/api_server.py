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

# Import our core engines
from digital_twin_core import DataIngestionEngine, AIProcessingEngine, OntologyEngine, SimulationEngine

# -----------------------------
# Helper Functions
# -----------------------------
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

def generate_ai_response(question, data, insights, ai_engine, simulation_engine):
    """Generate AI response based on question and dataset"""
    if data is None:
        return "Please load some data first before asking questions."
    
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['pattern', 'correlation', 'relationship']):
        if insights and 'correlations' in insights:
            correlations = insights['correlations']
            if correlations:
                response = "I found these key patterns in your data:\n"
                for corr in correlations[:3]:
                    response += f"• {corr['feature1']} and {corr['feature2']} are strongly correlated ({corr['correlation']:.2f})\n"
                return response
        return "I didn't find any strong correlations in your data."
    
    elif any(word in question_lower for word in ['anomaly', 'outlier', 'unusual']):
        anomalies = ai_engine.detect_anomalies(data)
        return f"I detected {len(anomalies)} anomalous data points in your dataset."
    
    elif any(word in question_lower for word in ['summary', 'overview', 'describe']):
        return f"""Here's an overview of your data:
        • {len(data)} total records
        • {len(data.columns)} features/columns
        • {data.select_dtypes(include=[np.number]).shape[1]} numeric features
        """
    
    elif any(word in question_lower for word in ['predict', 'forecast', 'future']):
        models_available = len(simulation_engine.models)
        if models_available > 0:
            return f"I have {models_available} predictive models ready for forecasting."
        else:
            return "I don't have any predictive models yet. Please build a model first."
    
    elif any(word in question_lower for word in ['help', 'what can you do']):
        return """I can help you with:
        • 🔍 Analyzing patterns and correlations
        • 🚨 Detecting anomalies and outliers
        • 📊 Providing data summaries
        • 🔮 Predictions and forecasts
        • 🧠 Knowledge graph insights
        
        Just ask me in natural language!"""
    
    else:
        return "I'm not sure about that. Try asking about patterns, anomalies, data summary, or predictions."

# -----------------------------
# Flask App and Engines
# -----------------------------
app = Flask(__name__)
CORS(app)

# Global state
current_data = None
current_insights = None

# Engines
data_engine = DataIngestionEngine()
ai_engine = AIProcessingEngine()
ontology_engine = OntologyEngine()
simulation_engine = SimulationEngine()

# -----------------------------
# Endpoints
# -----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/data/ingest', methods=['POST'])
def ingest_data():
    """Ingest CSV content and process"""
    global current_data, current_insights
    
    try:
        data = request.get_json()
        csv_content = data.get('csv_content')
        if not csv_content:
            return jsonify({'error': 'No CSV content provided'}), 400

        # DEBUG: CSV structure analysis
        csv_analysis = analyze_csv_structure(csv_content)
        print("📊 CSV Analysis:", json.dumps(csv_analysis, indent=2))

    current_data = None
    parsing_strategies = [
       {'sep': ',', 'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'},
       {'sep': ';', 'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'},
       {'sep': '\t', 'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'},
]

for i, strategy in enumerate(parsing_strategies):
    try:
        current_data = pd.read_csv(StringIO(csv_content), **strategy)
        if len(current_data) > 0 and len(current_data.columns) > 0:
            break
    except Exception as e:
        continue

# Sanitize all string columns to UTF-8 safely
current_data = current_data.applymap(
    lambda x: str(x).encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else x
)

        if current_data is None or len(current_data) == 0:
            return jsonify({'error': 'Could not parse CSV file. Please check the format.'}), 400

        # Clean columns and remove empty rows
        current_data.columns = [col.strip().replace('\n','').replace('\r','') for col in current_data.columns]
        current_data = current_data.dropna(how='all')

        # Sanitize all string cells to UTF-8 safely
        current_data = current_data.applymap(
            lambda x: str(x).encode('utf-8', errors='ignore').decode('utf-8') 
            if isinstance(x, str) else x
        )

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
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/analysis/patterns', methods=['POST'])
def analyze_patterns():
    global current_data, current_insights
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 404
        current_insights = ai_engine.extract_patterns(current_data)
        return jsonify({'success': True, 'insights': current_insights})
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/analysis/anomalies', methods=['POST'])
def detect_anomalies():
    global current_data
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 404
        data = request.get_json() or {}
        contamination = data.get('contamination', 0.1)
        anomaly_indices = ai_engine.detect_anomalies(current_data, contamination)
        anomalies = current_data.iloc[anomaly_indices] if anomaly_indices else pd.DataFrame()
        return jsonify({
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'anomalies': anomalies.to_dict('records') if not anomalies.empty else []
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/ontology/graph', methods=['GET'])
def get_knowledge_graph():
    try:
        graph = ontology_engine.get_knowledge_graph()
        nodes = [{'id': n, 'label': n, **d} for n,d in graph.nodes(data=True)]
        edges = [{'source': s, 'target': t, **d} for s,t,d in graph.edges(data=True)]
        return jsonify({
            'nodes': nodes,
            'edges': edges,
            'entities': ontology_engine.ontology.get('entities', {}),
            'relationships': ontology_engine.ontology.get('relationships', {})
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/ontology/query', methods=['POST'])
def query_ontology():
    try:
        data = request.get_json()
        query = data.get('query', '')
        results = ontology_engine.query_ontology(query)
        return jsonify({'query': query, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/simulation/model/build', methods=['POST'])
def build_model():
    global current_data
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 404
        data = request.get_json()
        target_column = data.get('target_column')
        if not target_column:
            return jsonify({'error': 'Target column required'}), 400
        model, message = simulation_engine.build_predictive_model(current_data, target_column)
        if model is None:
            return jsonify({'error': message}), 400
        model_info = simulation_engine.models[target_column]
        feature_importance = {
            feature: float(importance)
            for feature, importance in zip(model_info['features'], model_info['model'].feature_importances_)
        }
        return jsonify({
            'success': True,
            'message': message,
            'target_column': target_column,
            'features': model_info['features'],
            'mae': model_info['mae'],
            'feature_importance': feature_importance
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/simulation/forecast', methods=['POST'])
def generate_forecast():
    try:
        data = request.get_json()
        target_column = data.get('target_column')
        periods = data.get('periods', 30)
        if target_column not in simulation_engine.models:
            return jsonify({'error': 'Model not found'}), 404
        forecast_data, message = simulation_engine.run_forecast(target_column, periods)
        if forecast_data is None:
            return jsonify({'error': message}), 400
        for item in forecast_data:
            item['date'] = item['date'].isoformat()
        return jsonify({'success': True, 'message': message, 'forecast': forecast_data, 'periods': periods})
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/simulation/whatif', methods=['POST'])
def run_what_if():
    try:
        data = request.get_json()
        target_column = data.get('target_column')
        feature_changes = data.get('feature_changes', {})
        if target_column not in simulation_engine.models:
            return jsonify({'error': 'Model not found'}), 404
        scenario_result, message = simulation_engine.run_what_if_analysis(target_column, feature_changes)
        if scenario_result is None:
            return jsonify({'error': message}), 400
        return jsonify({'success': True, 'message': message, 'scenario': scenario_result})
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/simulation/risk', methods=['POST'])
def assess_risk():
    global current_data
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 404
        data = request.get_json()
        target_column = data.get('target_column')
        threshold_percentile = data.get('threshold_percentile', 95)
        risk_analysis, message = simulation_engine.assess_risks(current_data, target_column, threshold_percentile)
        if risk_analysis is None:
            return jsonify({'error': message}), 400
        return jsonify({'success': True, 'message': message, 'risk_analysis': risk_analysis})
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/chat/query', methods=['POST'])
def chat_query():
    global current_data, current_insights
    try:
        data = request.get_json()
        question = data.get('question', '')
        response = generate_ai_response(question, current_data, current_insights, ai_engine, simulation_engine)
        return jsonify({'question': question, 'response': response, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    global current_data
    try:
        status = {
            'server': 'running',
            'timestamp': datetime.now().isoformat(),
            'data_loaded': current_data is not None,
            'models_count': len(simulation_engine.models),
            'entities_count': len(ontology_engine.ontology.get('entities', {})),
            'relationships_count': len(ontology_engine.ontology.get('relationships', {}))
        }
        if current_data is not None:
            status.update({
                'data_shape': current_data.shape,
                'data_columns': len(current_data.columns),
                'numeric_columns': len(current_data.select_dtypes(include=[np.number]).columns)
            })
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Digital Twin Intelligence API Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8501, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    print(f"Starting Digital Twin Intelligence API Server at http://{args.host}:{args.port}")
    
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
