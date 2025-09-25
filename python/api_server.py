#!/usr/bin/env python3
# python/api_server.py - Complete Python API Server - FIXED JSON SERIALIZATION
import argparse
import sys
import json
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import StringIO
import io

# 🔧 Force stdout/stderr to UTF-8 (Windows fix)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# Import our core engines
from digital_twin_core import DataIngestionEngine, AIProcessingEngine, OntologyEngine, SimulationEngine

# --------------------------
# Helper functions
# --------------------------
def analyze_csv_structure(csv_content):
    """Analyze CSV structure for debugging"""
    lines = csv_content.split('\n')[:10]
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

def safe_convert(obj):
    """Recursively convert NumPy types to Python types for JSON"""
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [safe_convert(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [safe_convert(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): safe_convert(v) for k, v in obj.items()}
    if isinstance(obj, pd.Series):
        return safe_convert(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return safe_convert(obj.to_dict('records'))
    if hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    # Handle pandas dtypes
    if hasattr(obj, 'dtype'):
        return str(obj)
    return obj

def safe_jsonify(data):
    """Safely convert data to JSON-serializable format and return Flask response"""
    try:
        safe_data = safe_convert(data)
        return jsonify(safe_data)
    except Exception as e:
        print(f"JSON serialization error: {e}")
        return jsonify({'error': f'Serialization error: {str(e)}'}), 500

# --------------------------
# Flask setup
# --------------------------
app = Flask(__name__)
CORS(app)

# Custom JSON encoder for Flask
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        return safe_convert(obj)

app.json_encoder = SafeJSONEncoder

# --------------------------
# Initialize engines
# --------------------------
data_engine = DataIngestionEngine()
ai_engine = AIProcessingEngine()
ontology_engine = OntologyEngine()
simulation_engine = SimulationEngine()

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
    return safe_jsonify({
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
        source_name = data.get('source_name', 'Unnamed Source')
        
        if not csv_content:
            return safe_jsonify({'error': 'No CSV content provided'}), 400

        print(f"📥 Processing data ingestion for source: {source_name}")

        # DEBUG: CSV structure analysis
        csv_analysis = analyze_csv_structure(csv_content)
        print("📊 CSV Analysis:", json.dumps(safe_convert(csv_analysis), indent=2))

        # --- CSV parsing strategies ---
        current_data = None
        parsing_strategies = [
            {'sep': ',', 'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'},
            {'sep': ';', 'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'},
            {'sep': '\t', 'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'},
            {'sep': ',', 'encoding': 'latin-1', 'engine': 'python', 'on_bad_lines': 'skip'},
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
            return safe_jsonify({'error': 'Could not parse CSV file. Please check the format.'}), 400

        # Clean columns and remove empty rows
        current_data.columns = [str(col).strip().replace('\n','').replace('\r','') for col in current_data.columns]
        current_data = current_data.dropna(how='all')

        # Sanitize string cells and handle mixed types
        for col in current_data.columns:
            if current_data[col].dtype == 'object':
                current_data[col] = current_data[col].astype(str)
                current_data[col] = current_data[col].apply(
                    lambda x: str(x).encode('utf-8', errors='ignore').decode('utf-8') if pd.notna(x) else ''
                )

        print(f"✅ Data cleaned. Final shape: {current_data.shape}")
        print(f"Columns: {list(current_data.columns)}")
        print(f"Data types: {current_data.dtypes.to_dict()}")

        # AI processing
        try:
            print("🧠 Starting AI processing...")
            processed_data = ai_engine.clean_and_process(current_data)
            current_insights = ai_engine.extract_patterns(processed_data)
            print(f"✅ AI processing complete. Insights: {safe_convert(current_insights)}")
        except Exception as e:
            print(f"⚠️ AI processing failed: {e}")
            current_insights = {'error': str(e)}

        # Build knowledge graph
        try:
            print("🔗 Building knowledge graph...")
            ontology_engine.build_knowledge_graph(current_data, current_insights)
            print("✅ Knowledge graph built successfully")
        except Exception as e:
            print(f"⚠️ Knowledge graph building failed: {e}")

        # ✅ Prepare safe response
        response_data = {
            'success': True,
            'message': f'Successfully ingested {len(current_data)} rows',
            'rows': int(len(current_data)),
            'columns': int(len(current_data.columns)),
            'column_names': [str(col) for col in current_data.columns],
            'sample_data': safe_convert(current_data.head(3).to_dict('records')),
            'data_types': {str(k): str(v) for k, v in current_data.dtypes.to_dict().items()},
            'insights': safe_convert(current_insights) if current_insights else {}
        }

        return safe_jsonify(response_data)

    except Exception as e:
        error_response = {
            'success': False,
            'error': f'Data ingestion failed: {str(e)}',
            'traceback': traceback.format_exc()
        }
        print(f"❌ Ingestion error: {error_response}")
        return safe_jsonify(error_response), 500

@app.route('/api/data/current', methods=['GET'])
def get_current_data():
    global current_data
    try:
        if current_data is None:
            return safe_jsonify({'error': 'No data loaded'}), 404

        # Get basic statistics safely
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        stats_dict = {}
        
        if len(numeric_cols) > 0:
            stats_df = current_data[numeric_cols].describe()
            stats_dict = {str(col): safe_convert(stats_df[col].to_dict()) for col in stats_df.columns}

        response = {
            'shape': [int(current_data.shape[0]), int(current_data.shape[1])],
            'columns': [str(col) for col in current_data.columns],
            'dtypes': {str(k): str(v) for k, v in current_data.dtypes.to_dict().items()},
            'sample': safe_convert(current_data.head(5).to_dict('records')),
            'statistics': stats_dict
        }
        
        return safe_jsonify(response)

    except Exception as e:
        return safe_jsonify({'error': f'Failed to get current data: {str(e)}'}), 500

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    try:
        # Get database stats
        entities_count = len(ontology_engine.ontology.get('entities', {}))
        relationships_count = len(ontology_engine.ontology.get('relationships', {}))
        models_count = len(simulation_engine.models)
        
        status = {
            'data_shape': [int(current_data.shape[0]), int(current_data.shape[1])] if current_data is not None else [0, 0],
            'entities_count': int(entities_count),
            'relationships_count': int(relationships_count),
            'models_count': int(models_count),
            'status': 'active' if current_data is not None else 'waiting',
            'last_update': datetime.now().isoformat()
        }
        
        return safe_jsonify(status)
        
    except Exception as e:
        return safe_jsonify({
            'error': f'Failed to get system status: {str(e)}',
            'data_shape': [0, 0],
            'entities_count': 0,
            'relationships_count': 0,
            'models_count': 0,
            'status': 'error'
        }), 500

@app.route('/api/analysis/patterns', methods=['POST'])
def run_pattern_analysis():
    global current_data, current_insights
    
    try:
        if current_data is None:
            return safe_jsonify({'error': 'No data loaded'}), 400
        
        print("🔍 Running pattern analysis...")
        insights = ai_engine.extract_patterns(current_data)
        current_insights = insights
        
        return safe_jsonify({
            'success': True,
            'insights': safe_convert(insights),
            'message': 'Pattern analysis completed successfully'
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'Pattern analysis failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analysis/anomalies', methods=['POST'])
def detect_anomalies():
    global current_data
    
    try:
        if current_data is None:
            return safe_jsonify({'error': 'No data loaded'}), 400
            
        data = request.get_json()
        contamination = float(data.get('contamination', 0.1))
        
        print(f"🚨 Running anomaly detection with contamination: {contamination}")
        anomaly_indices = ai_engine.detect_anomalies(current_data, contamination)
        
        return safe_jsonify({
            'success': True,
            'anomaly_count': int(len(anomaly_indices)),
            'anomaly_indices': [int(idx) for idx in anomaly_indices],
            'total_records': int(len(current_data)),
            'contamination': float(contamination)
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'Anomaly detection failed: {str(e)}'
        }), 500

@app.route('/api/simulation/model/build', methods=['POST'])
def build_model():
    global current_data
    
    try:
        if current_data is None:
            return safe_jsonify({'error': 'No data loaded'}), 400
            
        data = request.get_json()
        target_column = data.get('target_column')
        
        if not target_column:
            return safe_jsonify({'error': 'Target column not specified'}), 400
            
        print(f"🤖 Building predictive model for: {target_column}")
        model, message = simulation_engine.build_predictive_model(current_data, target_column)
        
        if model is None:
            return safe_jsonify({'success': False, 'error': message}), 400
            
        model_info = simulation_engine.models[target_column]
        
        # Get feature importance safely
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, feature in enumerate(model_info['features']):
                feature_importance[feature] = float(model.feature_importances_[i])
        
        return safe_jsonify({
            'success': True,
            'target_column': str(target_column),
            'features': [str(f) for f in model_info['features']],
            'mae': float(model_info['mae']),
            'feature_importance': feature_importance,
            'message': message
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'Model building failed: {str(e)}'
        }), 500

@app.route('/api/simulation/forecast', methods=['POST'])
def generate_forecast():
    global current_data
    
    try:
        data = request.get_json()
        target_column = data.get('target_column')
        periods = int(data.get('periods', 30))
        
        print(f"🔮 Generating forecast for {target_column}, {periods} periods")
        forecast_data, message = simulation_engine.run_forecast(target_column, periods)
        
        if forecast_data is None:
            return safe_jsonify({'success': False, 'error': message}), 400
            
        # Convert forecast data to JSON-safe format
        safe_forecast = []
        for item in forecast_data:
            safe_forecast.append({
                'date': item['date'].strftime('%Y-%m-%d'),
                'predicted_value': float(item['predicted_value']),
                'confidence_interval': float(item['confidence_interval'])
            })
        
        return safe_jsonify({
            'success': True,
            'forecast': safe_forecast,
            'periods': int(periods),
            'target_column': str(target_column),
            'message': message
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'Forecast generation failed: {str(e)}'
        }), 500

@app.route('/api/chat/query', methods=['POST'])
def chat_query():
    global current_data, current_insights
    
    try:
        data = request.get_json()
        question = data.get('question', '').lower()
        
        if not question:
            return safe_jsonify({'response': 'Please ask me a question about your data.'})
            
        if current_data is None:
            return safe_jsonify({
                'response': 'I don\'t have any data to analyze yet. Please upload a CSV file first.'
            })
        
        # Simple rule-based responses
        if any(word in question for word in ['shape', 'size', 'rows', 'columns']):
            response = f"Your dataset has {current_data.shape[0]} rows and {current_data.shape[1]} columns."
        elif 'columns' in question or 'features' in question:
            cols = ', '.join(current_data.columns[:10])
            if len(current_data.columns) > 10:
                cols += f" ... (and {len(current_data.columns) - 10} more)"
            response = f"The columns in your dataset are: {cols}"
        elif any(word in question for word in ['summary', 'describe', 'overview']):
            numeric_cols = len(current_data.select_dtypes(include=[np.number]).columns)
            text_cols = len(current_data.select_dtypes(include=[object]).columns)
            response = f"Dataset overview: {current_data.shape[0]} records, {current_data.shape[1]} features ({numeric_cols} numeric, {text_cols} text columns)."
        elif 'pattern' in question or 'correlation' in question:
            if current_insights and 'correlations' in current_insights:
                corr_count = len(current_insights['correlations'])
                if corr_count > 0:
                    strongest = current_insights['correlations'][0]
                    response = f"Found {corr_count} strong correlations. The strongest is between {strongest['feature1']} and {strongest['feature2']} ({strongest['correlation']:.2f})."
                else:
                    response = "No strong correlations found in your data."
            else:
                response = "Run pattern analysis first to discover correlations."
        else:
            response = "I can help you analyze your data. Try asking about data shape, columns, patterns, or correlations."
        
        return safe_jsonify({'response': response})
        
    except Exception as e:
        return safe_jsonify({
            'response': f'Sorry, I encountered an error: {str(e)}'
        })

@app.route('/api/ontology/query', methods=['POST'])
def query_ontology():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        results = ontology_engine.query_ontology(query)
        
        return safe_jsonify({
            'success': True,
            'query': str(query),
            'results': [str(result) for result in results]
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'Ontology query failed: {str(e)}'
        }), 500

@app.route('/api/ontology/graph', methods=['GET'])
def get_knowledge_graph():
    try:
        graph = ontology_engine.get_knowledge_graph()
        
        # Convert NetworkX graph to JSON-safe format
        nodes = []
        edges = []
        
        for node in graph.nodes(data=True):
            nodes.append({
                'id': str(node[0]),
                'label': str(node[0]),
                'type': str(node[1].get('type', 'unknown')),
                'data_type': str(node[1].get('data_type', ''))
            })
        
        for edge in graph.edges(data=True):
            edges.append({
                'source': str(edge[0]),
                'target': str(edge[1]),
                'relationship': str(edge[2].get('relationship', 'unknown')),
                'strength': float(edge[2].get('strength', 0))
            })
        
        return safe_jsonify({
            'success': True,
            'nodes': nodes,
            'edges': edges,
            'entities': safe_convert(ontology_engine.ontology['entities']),
            'relationships': safe_convert(ontology_engine.ontology['relationships'])
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'Failed to get knowledge graph: {str(e)}'
        }), 500

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description='Digital Twin Intelligence API Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8501, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print(f"🚀 Starting Digital Twin Intelligence API Server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Server running at: http://{args.host}:{args.port}")
    print(f"Health check: http://{args.host}:{args.port}/health")

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()