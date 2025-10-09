#!/usr/bin/env python3
# python/api_server.py - Complete Python API Server - FIXED STRUCTURE
import argparse
import sys
import json
import traceback
from datetime import datetime
from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
import pandas as pd
import numpy as np
from io import StringIO
import io

# 🔧 Force stdout/stderr to UTF-8 (Windows fix)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# Import our core engines
from digital_twin_core import DataIngestionEngine, AIProcessingEngine, OntologyEngine, SimulationEngine, CausalAnalysisEngine

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
    
    # Handle NaN and infinity FIRST (critical for JSON serialization)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj):
            return None  # Convert NaN to JSON null
        if np.isinf(obj):
            return None  # Convert infinity to JSON null
        return float(obj)
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    
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
        item_val = obj.item()
        if isinstance(item_val, float) and (np.isnan(item_val) or np.isinf(item_val)):
            return None
        return item_val
    
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
causal_engine = CausalAnalysisEngine()

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

@app.route('/api/data/ingest/excel', methods=['POST'])
def ingest_excel():
    global current_data, current_insights

    try:
        data = request.get_json()
        excel_content = data.get('excel_content')  # Base64 encoded
        source_name = data.get('source_name', 'Unnamed Excel Source')
        sheet_name = data.get('sheet_name', 0)  # Default to first sheet
        
        if not excel_content:
            return safe_jsonify({'error': 'No Excel content provided'}), 400

        print(f"📥 Processing Excel ingestion for source: {source_name}")

        # Decode base64 content
        import base64
        from io import BytesIO
        
        excel_bytes = base64.b64decode(excel_content)
        excel_buffer = BytesIO(excel_bytes)
        
        # Ingest Excel file
        current_data = data_engine.ingest_excel(excel_buffer, source_name, sheet_name)
        
        if current_data is None or len(current_data) == 0:
            return safe_jsonify({'error': 'Could not parse Excel file'}), 400

        # Clean and sanitize data (same as CSV processing)
        current_data.columns = [str(col).strip() for col in current_data.columns]
        current_data = current_data.dropna(how='all')

        for col in current_data.columns:
            if current_data[col].dtype == 'object':
                current_data[col] = current_data[col].astype(str)
                current_data[col] = current_data[col].apply(
                    lambda x: str(x).encode('utf-8', errors='ignore').decode('utf-8') if pd.notna(x) else ''
                )

        print(f"✅ Excel data cleaned. Final shape: {current_data.shape}")

        # AI processing
        try:
            processed_data = ai_engine.clean_and_process(current_data)
            current_insights = ai_engine.extract_patterns(processed_data)
        except Exception as e:
            print(f"⚠️ AI processing failed: {e}")
            current_insights = {'error': str(e)}

        # Build knowledge graph
        try:
            ontology_engine.build_knowledge_graph(current_data, current_insights)
        except Exception as e:
            print(f"⚠️ Knowledge graph building failed: {e}")

        response_data = {
            'success': True,
            'message': f'Successfully ingested {len(current_data)} rows from Excel',
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
            'error': f'Excel ingestion failed: {str(e)}',
            'traceback': traceback.format_exc()
        }
        print(f"❌ Excel ingestion error: {error_response}")
        return safe_jsonify(error_response), 500


@app.route('/api/data/excel/sheets', methods=['POST'])
def get_excel_sheets():
    """Get list of sheet names from uploaded Excel file"""
    try:
        data = request.get_json()
        excel_content = data.get('excel_content')
        
        if not excel_content:
            return safe_jsonify({'error': 'No Excel content provided'}), 400
        
        import base64
        from io import BytesIO
        
        excel_bytes = base64.b64decode(excel_content)
        excel_buffer = BytesIO(excel_bytes)
        
        sheet_names = data_engine.get_excel_sheets(excel_buffer)
        
        return safe_jsonify({
            'success': True,
            'sheets': sheet_names
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'Failed to read sheets: {str(e)}'
        }), 500

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

@app.route('/api/data/pivot', methods=['POST'])
def generate_pivot():
    global current_data
    
    try:
        if current_data is None:
            return safe_jsonify({'error': 'No data loaded'}), 400
        
        data = request.get_json()
        rows = data.get('rows', [])
        columns = data.get('columns')
        values = data.get('values')
        agg_func = data.get('agg_func', 'sum')
        
        if not rows:
            return safe_jsonify({'error': 'At least one row field required'}), 400
        
        if not values:
            return safe_jsonify({'error': 'Value field required'}), 400
        
        print(f"🔄 Generating pivot: rows={rows}, columns={columns}, values={values}, agg={agg_func}")
        
        # Create pivot table
        if columns and columns != 'None':
            pivot = pd.pivot_table(
                current_data,
                values=values,
                index=rows,
                columns=columns,
                aggfunc=agg_func,
                fill_value=0
            )
        else:
            # Simple groupby without columns
            pivot = current_data.groupby(rows)[values].agg(agg_func)
        
        # Convert to dict for JSON
        if isinstance(pivot, pd.Series):
            pivot_dict = pivot.reset_index().to_dict('records')
        else:
            pivot_dict = pivot.reset_index().to_dict('records')
        
        # Get column names
        column_names = list(pivot.reset_index().columns)
        
        print(f"✅ Pivot generated: {len(pivot_dict)} rows")
        
        return safe_jsonify({
            'success': True,
            'pivot_data': safe_convert(pivot_dict),
            'columns': column_names,
            'row_count': len(pivot_dict),
            'summary': {
                'rows': rows,
                'columns': columns,
                'values': values,
                'aggregation': agg_func
            }
        })
        
    except Exception as e:
        print(f"❌ Pivot generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return safe_jsonify({
            'success': False,
            'error': f'Pivot generation failed: {str(e)}'
        }), 500


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
        
        print("🔍 Running enhanced pattern analysis...")
        print(f"Data shape: {current_data.shape}")
        print(f"Numeric columns: {list(current_data.select_dtypes(include=[np.number]).columns)}")
        
        # Run enhanced pattern analysis
        insights = ai_engine.extract_patterns(current_data)
        current_insights = insights
        
        # Debug output
        if 'correlations' in insights:
            print(f"✅ Found {len(insights['correlations'])} strong correlations")
            for corr in insights['correlations'][:3]:  # Show top 3
                print(f"   {corr['feature1']} ↔ {corr['feature2']}: r={corr['correlation']:.3f}")
        
        if 'moderate_correlations' in insights:
            print(f"✅ Found {len(insights['moderate_correlations'])} moderate correlations")
        
        if 'clusters' in insights:
            print(f"✅ Created {insights['clusters']['n_clusters']} data clusters")
        
        # Format response with detailed insights
        response = {
            'success': True,
            'insights': safe_convert(insights),
            'summary': {
                'strong_correlations_found': len(insights.get('correlations', [])),
                'moderate_correlations_found': len(insights.get('moderate_correlations', [])),
                'clusters_found': insights.get('clusters', {}).get('n_clusters', 0),
                'data_quality_score': 100 - (insights.get('data_quality', {}).get('missing_values', 0) / insights.get('data_quality', {}).get('total_rows', 1) * 100)
            },
            'message': 'Enhanced pattern analysis completed successfully'
        }
        
        return safe_jsonify(response)
        
    except Exception as e:
        error_msg = f'Pattern analysis failed: {str(e)}'
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        return safe_jsonify({
            'success': False,
            'error': error_msg,
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
        
        print(f"🚨 Running enhanced anomaly detection with contamination: {contamination}")
        print(f"Data shape: {current_data.shape}")
        
        # Run enhanced anomaly detection
        anomaly_result = ai_engine.detect_anomalies(current_data, contamination)
        
        if isinstance(anomaly_result, dict):
            # New enhanced format
            response = {
                'success': True,
                'anomaly_count': int(len(anomaly_result['indices'])),
                'anomaly_indices': [int(idx) for idx in anomaly_result['indices']],
                'anomaly_scores': [float(score) for score in anomaly_result['scores']],
                'threshold_score': float(anomaly_result['threshold_score']),
                'total_records': int(len(current_data)),
                'contamination': float(contamination),
                'anomaly_rate': float(len(anomaly_result['indices']) / len(current_data) * 100) if len(current_data) > 0 else 0
            }
        else:
            # Fallback to old format
            anomaly_indices = anomaly_result if isinstance(anomaly_result, list) else []
            response = {
                'success': True,
                'anomaly_count': int(len(anomaly_indices)),
                'anomaly_indices': [int(idx) for idx in anomaly_indices],
                'total_records': int(len(current_data)),
                'contamination': float(contamination),
                'anomaly_rate': float(len(anomaly_indices) / len(current_data) * 100) if len(current_data) > 0 else 0
            }
        
        print(f"✅ Anomaly detection completed: {response['anomaly_count']} anomalies found")
        return safe_jsonify(response)
        
    except Exception as e:
        error_msg = f'Anomaly detection failed: {str(e)}'
        print(f"❌ {error_msg}")
        
        return safe_jsonify({
            'success': False,
            'error': error_msg,
            'anomaly_count': 0,
            'total_records': len(current_data) if current_data is not None else 0
        }), 500

@app.route('/api/analysis/causality', methods=['POST'])
def run_causal_analysis():
    global current_data
    
    try:
        if current_data is None:
            return safe_jsonify({'error': 'No data loaded'}), 400
        
        data = request.get_json()
        treatment_col = data.get('treatment_column')
        outcome_col = data.get('outcome_column')
        
        if not treatment_col or not outcome_col:
            return safe_jsonify({'error': 'Both treatment and outcome columns required'}), 400
        
        print(f"Running causal analysis: {treatment_col} -> {outcome_col}")
        
        # Run causal analysis
        result, message = causal_engine.analyze_causality(
            current_data, 
            treatment_col, 
            outcome_col
        )
        
        if result is None:
            return safe_jsonify({'success': False, 'error': message}), 400
        
        return safe_jsonify({
            'success': True,
            'causal_analysis': safe_convert(result),
            'message': message
        })
        
    except Exception as e:
        print(f"Causal analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return safe_jsonify({
            'success': False,
            'error': f'Causal analysis failed: {str(e)}'
        }), 500

@app.route('/api/analysis/cluster/advanced', methods=['POST'])
def run_advanced_clustering():
    global current_data
    
    try:
        if current_data is None:
            return safe_jsonify({'error': 'No data loaded'}), 400
        
        data = request.get_json()
        method = data.get('method', 'dbscan')
        eps = float(data.get('eps', 0.5))
        min_samples = int(data.get('min_samples', 5))
        
        print(f"Running advanced clustering: {method}")
        result, message = ai_engine.advanced_clustering(current_data, method, eps, min_samples)
        
        if result is None:
            return safe_jsonify({'success': False, 'error': message}), 400
        
        return safe_jsonify({
            'success': True,
            'clusters': result,
            'message': message
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'Advanced clustering failed: {str(e)}'
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

@app.route('/api/simulation/model/explain', methods=['POST'])
def explain_model():
    """Generate SHAP explanations for a model"""
    try:
        data = request.get_json()
        target_column = data.get('target_column')
        
        if not target_column:
            return safe_jsonify({'error': 'Target column not specified'}), 400
        
        print(f"Generating SHAP explanations for: {target_column}")
        explanation, message = simulation_engine.explain_predictions(target_column)
        
        if explanation is None:
            return safe_jsonify({'success': False, 'error': message}), 400
        
        return safe_jsonify({
            'success': True,
            'explanation': safe_convert(explanation),
            'message': message
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'Explanation generation failed: {str(e)}'
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

@app.route('/api/simulation/forecast/advanced', methods=['POST'])
def generate_advanced_forecast():
    global current_data
    
    try:
        data = request.get_json()
        target_column = data.get('target_column')
        periods = int(data.get('periods', 30))
        method = data.get('method', 'exponential')
        
        print(f"Generating advanced forecast for {target_column}, {periods} periods")
        forecast_data, message = simulation_engine.run_advanced_forecast(target_column, periods, method)
        
        if forecast_data is None:
            return safe_jsonify({'success': False, 'error': message}), 400
            
        # Convert forecast data to JSON-safe format
        safe_forecast = []
        for item in forecast_data:
            safe_forecast.append({
                'date': item['date'].strftime('%Y-%m-%d'),
                'predicted_value': float(item['predicted_value']),
                'confidence_interval': float(item['confidence_interval']),
                'upper_bound': float(item.get('upper_bound', 0)),
                'lower_bound': float(item.get('lower_bound', 0))
            })
        
        return safe_jsonify({
            'success': True,
            'forecast': safe_forecast,
            'periods': int(periods),
            'target_column': str(target_column),
            'method': str(method),
            'message': message
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'Advanced forecast generation failed: {str(e)}'
        }), 500

@app.route('/api/simulation/whatif', methods=['POST'])
def run_whatif_analysis():
    try:
        data = request.get_json()
        target_column = data.get('target_column')
        feature_changes = data.get('feature_changes', {})
        
        print(f"🎭 Running what-if analysis for {target_column}")
        analysis, message = simulation_engine.run_what_if_analysis(target_column, feature_changes)
        
        if analysis is None:
            return safe_jsonify({'success': False, 'error': message}), 400
            
        return safe_jsonify({
            'success': True,
            'analysis': analysis,
            'message': message
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': f'What-if analysis failed: {str(e)}'
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
                'response': 'I don\'t have any data to analyze yet. Please upload a CSV file first, then I can help you explore patterns, correlations, and insights!'
            })
        
        # Enhanced responses with actual data analysis
        if any(word in question for word in ['shape', 'size', 'rows', 'columns', 'dimension']):
            numeric_cols = len(current_data.select_dtypes(include=[np.number]).columns)
            text_cols = len(current_data.select_dtypes(include=[object]).columns)
            response = f"📊 Your dataset has **{current_data.shape[0]} rows** and **{current_data.shape[1]} columns** ({numeric_cols} numeric, {text_cols} text features)."
            
        elif any(word in question for word in ['columns', 'features', 'variables']):
            cols = list(current_data.columns[:10])
            if len(current_data.columns) > 10:
                cols_text = ', '.join(cols) + f" ... (and {len(current_data.columns) - 10} more)"
            else:
                cols_text = ', '.join(cols)
            response = f"📋 **Columns in your dataset:** {cols_text}"
            
        elif any(word in question for word in ['summary', 'describe', 'overview']):
            numeric_cols = len(current_data.select_dtypes(include=[np.number]).columns)
            text_cols = len(current_data.select_dtypes(include=[object]).columns)
            missing_pct = (current_data.isnull().sum().sum() / (current_data.shape[0] * current_data.shape[1]) * 100)
            
            response = f"""📈 **Dataset Overview:**
• **Size:** {current_data.shape[0]} records × {current_data.shape[1]} features
• **Data Types:** {numeric_cols} numeric, {text_cols} categorical
• **Data Quality:** {missing_pct:.1f}% missing values
• **Memory Usage:** {current_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"""
            
        elif any(word in question for word in ['pattern', 'correlation', 'relationship']):
            if current_insights and 'correlations' in current_insights:
                strong_corrs = current_insights['correlations']
                moderate_corrs = current_insights.get('moderate_correlations', [])
                
                if strong_corrs:
                    top_corr = strong_corrs[0]
                    response = f"""🔗 **Correlation Analysis Results:**
• **Strong correlations found:** {len(strong_corrs)}
• **Moderate correlations found:** {len(moderate_corrs)}
• **Strongest relationship:** {top_corr['feature1']} ↔ {top_corr['feature2']} (r={top_corr['correlation']:.3f})

💡 Strong correlations indicate features that move together predictably."""
                else:
                    response = f"""📊 **Correlation Analysis Results:**
• Found {len(moderate_corrs)} moderate correlations (0.3-0.5 range)
• No very strong correlations (>0.5) detected
• This suggests your features are relatively independent, which can be good for modeling!"""
            else:
                response = "🔍 I haven't analyzed correlations yet. Click **'Discover Patterns'** in the AI Analysis section to find relationships in your data!"
                
        elif any(word in question for word in ['anomaly', 'anomalies', 'outlier', 'unusual']):
            response = """🚨 **Anomaly Detection:** Use the Anomaly Detection tool in the AI Analysis section! 

• Adjust the **sensitivity slider** (lower = more sensitive)
• 0.1 = finds top 10% most unusual records
• 0.05 = finds top 5% most unusual records

Anomalies can reveal data quality issues or interesting edge cases."""
            
        elif any(word in question for word in ['cluster', 'segment', 'group']):
            if current_insights and 'clusters' in current_insights:
                cluster_info = current_insights['clusters']
                response = f"""🎯 **Clustering Results:**
• **Found {cluster_info['n_clusters']} distinct data patterns**
• **Cluster sizes:** {cluster_info.get('cluster_sizes', [])}
• **Features used:** {len(cluster_info.get('features_used', []))} numeric columns

This segmentation can help identify different types of records in your dataset."""
            else:
                response = "🎯 Run **pattern analysis** first to discover data clusters and segments!"
                
        elif any(word in question for word in ['model', 'predict', 'forecast']):
            response = """🤖 **Predictive Modeling:** 

1. Go to the **Simulations & Predictions** page
2. Choose a numeric column as your target variable
3. Click **'Build Model'** to create an AI predictor
4. Generate forecasts and run what-if scenarios!

The AI will automatically select the best features and build a Random Forest model."""
            
        elif any(word in question for word in ['missing', 'null', 'empty']):
            missing_data = current_data.isnull().sum()
            cols_with_missing = missing_data[missing_data > 0]
            
            if len(cols_with_missing) > 0:
                total_missing = missing_data.sum()
                response = f"""❓ **Missing Data Analysis:**
• **Total missing values:** {total_missing}
• **Columns with missing data:** {len(cols_with_missing)}
• **Most missing:** {cols_with_missing.index[0]} ({cols_with_missing.iloc[0]} values)

The AI automatically handles missing values during analysis."""
            else:
                response = "✅ **Great news!** Your dataset has no missing values. This makes analysis more reliable!"
                
        elif any(word in question for word in ['help', 'what can you do', 'capabilities']):
            response = """🤖 **I can help you with:**

**📊 Data Analysis:**
• Describe your dataset structure and quality
• Find correlations and patterns
• Detect anomalies and outliers
• Segment data into clusters

**🔮 Predictions:**
• Build AI models for any numeric target
• Generate forecasts and predictions
• Run what-if scenario analysis

**💬 Ask me about:**
• "What patterns do you see?"
• "Are there any anomalies?"
• "Summarize my data"
• "What correlations exist?"
• "How can I predict [column name]?"

Just upload your CSV and start exploring! 🚀"""
            
        else:
            # Default helpful response
            response = """🤔 I can analyze your data in many ways! Try asking:

• **"What patterns do you see?"** - For correlation analysis
• **"Are there any anomalies?"** - For outlier detection  
• **"Summarize my data"** - For an overview
• **"What correlations exist?"** - For relationships
• **"Help me predict [column]"** - For modeling advice

Or use the analysis tools in the sidebar! 📊"""
        
        return safe_jsonify({'response': response})
        
    except Exception as e:
        return safe_jsonify({
            'response': f'Sorry, I encountered an error: {str(e)}. Please try rephrasing your question or check if your data is loaded properly.'
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