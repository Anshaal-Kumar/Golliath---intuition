# python/digital_twin_core.py - Complete AI Digital Twin Engine
import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
import networkx as nx
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# AI and ML imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class DataIngestionEngine:
    """Universal data ingestion layer"""
    
    def __init__(self, db_path="digital_twin.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_type TEXT,
                raw_content TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT,
                entity_name TEXT,
                attributes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT,
                target_entity TEXT,
                relationship_type TEXT,
                strength REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def ingest_csv(self, file_path_or_data, source_name):
        """Ingest data from CSV with robust parsing"""
        try:
            # Multiple CSV parsing strategies
            csv_read_options = [
                {'sep': ',', 'engine': 'python', 'on_bad_lines': 'skip'},
                {'sep': ';', 'engine': 'python', 'on_bad_lines': 'skip'},
                {'sep': '\t', 'engine': 'python', 'on_bad_lines': 'skip'},
                {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip'}
            ]
            
            df = None
            error_messages = []
            
            for i, options in enumerate(csv_read_options):
                try:
                    print(f"Trying CSV parsing method {i+1}: {options}")
                    
                    if isinstance(file_path_or_data, str):
                        df = pd.read_csv(file_path_or_data, encoding='utf-8', **options)
                    else:
                        df = pd.read_csv(file_path_or_data, encoding='utf-8', **options)
                    
                    if len(df) > 0:
                        print(f"✅ Successfully parsed CSV with method {i+1}")
                        print(f"Shape: {df.shape}, Columns: {list(df.columns)}")
                        break
                except Exception as e:
                    error_messages.append(f"Method {i+1}: {str(e)}")
                    continue
            
            if df is None or len(df) == 0:
                raise Exception(f"Failed to parse CSV with all methods. Errors: {'; '.join(error_messages)}")
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            raw_content = df.to_json(orient='records', force_ascii=False)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO raw_data (source_name, data_type, raw_content)
                VALUES (?, ?, ?)
            ''', (source_name, 'CSV', raw_content))
            conn.commit()
            conn.close()
            
            return df
        except Exception as e:
            print(f"Error ingesting CSV: {str(e)}")
            return None
    
    def get_all_data_sources(self):
        """Get list of all data sources"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT source_name, data_type, timestamp, processed
            FROM raw_data
            ORDER BY timestamp DESC
        ''', conn)
        conn.close()
        return df

class AIProcessingEngine:
    """AI-powered data processing engine"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
    
    def clean_and_process(self, df):
        """Clean and preprocess data"""
        processed_df = df.copy()
        for column in processed_df.columns:
            if processed_df[column].dtype in ['int64', 'float64']:
                processed_df[column].fillna(processed_df[column].median(), inplace=True)
            else:
                processed_df[column].fillna(processed_df[column].mode().iloc[0] if not processed_df[column].mode().empty else 'Unknown', inplace=True)
        return processed_df
    
    def detect_anomalies(self, df, contamination=0.1):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return []
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        numeric_data = df[numeric_columns].fillna(0)
        anomaly_labels = iso_forest.fit_predict(numeric_data)
        anomalies = df[anomaly_labels == -1]
        return anomalies.index.tolist()
    
    def extract_patterns(self, df):
        insights = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_correlations.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            insights['correlations'] = high_correlations
        if len(numeric_columns) >= 2:
            try:
                kmeans = KMeans(n_clusters=min(5, len(df)), random_state=42, n_init=10)
                cluster_data = self.scaler.fit_transform(df[numeric_columns].fillna(0))
                clusters = kmeans.fit_predict(cluster_data)
                insights['clusters'] = {
                    'labels': clusters.tolist(),
                    'centers': kmeans.cluster_centers_.tolist(),
                    'n_clusters': len(set(clusters))
                }
            except Exception as e:
                print(f"Clustering analysis failed: {str(e)}")
        return insights

class OntologyEngine:
    """Knowledge graph construction engine"""
    
    def __init__(self, db_path="digital_twin.db"):
        self.db_path = db_path
        self.knowledge_graph = nx.Graph()
        self.ontology = {'entities': {}, 'relationships': {}, 'rules': []}
    
    def build_knowledge_graph(self, df, insights):
        self.knowledge_graph.clear()
        for column in df.columns:
            self.knowledge_graph.add_node(
                column, 
                type='attribute',
                data_type=str(df[column].dtype),
                unique_values=df[column].nunique(),
                description=f"Data attribute: {column}"
            )
            self.ontology['entities'][column] = {
                'type': 'attribute',
                'data_type': str(df[column].dtype),
                'statistics': self._get_column_stats(df[column])
            }
        if 'correlations' in insights:
            for corr in insights['correlations']:
                self.knowledge_graph.add_edge(
                    corr['feature1'],
                    corr['feature2'],
                    relationship='correlation',
                    strength=abs(corr['correlation']),
                    description=f"Correlation: {corr['correlation']:.2f}"
                )
                self.ontology['relationships'][f"{corr['feature1']}-{corr['feature2']}"] = {
                    'type': 'correlation',
                    'strength': corr['correlation'],
                    'significance': 'high' if abs(corr['correlation']) > 0.8 else 'medium'
                }
        if 'clusters' in insights:
            cluster_node = 'data_clusters'
            self.knowledge_graph.add_node(
                cluster_node,
                type='pattern',
                n_clusters=insights['clusters']['n_clusters'],
                description=f"Data segmentation with {insights['clusters']['n_clusters']} patterns"
            )
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                self.knowledge_graph.add_edge(
                    cluster_node,
                    column,
                    relationship='influences',
                    strength=0.5,
                    description=f"{column} influences clustering pattern"
                )
        self._save_ontology()
    
    def _get_column_stats(self, series):
        stats = {'count': len(series), 'null_count': series.isnull().sum(), 'unique_count': series.nunique()}
        if series.dtype in ['int64', 'float64']:
            stats.update({'mean': float(series.mean()), 'std': float(series.std()), 'min': float(series.min()), 'max': float(series.max()), 'median': float(series.median())})
        return stats
    
    def _save_ontology(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM entities')
        cursor.execute('DELETE FROM relationships')
        for entity_name, entity_data in self.ontology['entities'].items():
            cursor.execute('''
                INSERT INTO entities (entity_type, entity_name, attributes)
                VALUES (?, ?, ?)
            ''', (entity_data['type'], entity_name, json.dumps(entity_data, ensure_ascii=False)))
        for rel_name, rel_data in self.ontology['relationships'].items():
            source, target = rel_name.split('-')
            cursor.execute('''
                INSERT INTO relationships (source_entity, target_entity, relationship_type, strength)
                VALUES (?, ?, ?, ?)
            ''', (source, target, rel_data['type'], rel_data['strength']))
        conn.commit()
        conn.close()
    
    def get_knowledge_graph(self):
        return self.knowledge_graph
    
    def query_ontology(self, query):
        results = []
        query_lower = query.lower()
        if 'correlation' in query_lower:
            for rel_name, rel_data in self.ontology['relationships'].items():
                if rel_data['type'] == 'correlation':
                    source, target = rel_name.split('-')
                    results.append(f"{source} and {target} are correlated (strength: {rel_data['strength']:.2f})")
        if 'entities' in query_lower or 'attributes' in query_lower:
            results.extend(list(self.ontology['entities'].keys()))
        return results if results else ["No relevant information found."]

class SimulationEngine:
    """Predictive simulation engine"""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
    
    def build_predictive_model(self, df, target_column):
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column not in numeric_columns:
                return None, "Target column must be numeric"
            feature_columns = [col for col in numeric_columns if col != target_column]
            if len(feature_columns) == 0:
                return None, "No feature columns available"
            X = df[feature_columns].fillna(0)
            y = df[target_column].fillna(df[target_column].median())
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            self.models[target_column] = {'model': model, 'features': feature_columns, 'mae': mae, 'target': target_column}
            return model, f"Model trained successfully. MAE: {mae:.2f}"
        except Exception as e:
            return None, f"Error building model: {str(e)}"
    
    def run_forecast(self, target_column, periods=30):
        if target_column not in self.models:
            return None, "No model found"
        model_info = self.models[target_column]
        model = model_info['model']
        forecast_data = []
        base_date = datetime.now()
        for i in range(periods):
            future_features = np.random.normal(0, 1, len(model_info['features'])).reshape(1, -1)
            prediction = model.predict(future_features)[0]
            forecast_data.append({'date': base_date + timedelta(days=i), 'predicted_value': prediction, 'confidence_interval': prediction * 0.1})
        return forecast_data, "Forecast generated"
    
    def run_what_if_analysis(self, target_column, feature_changes):
        if target_column not in self.models:
            return None, "No model found"
        model_info = self.models[target_column]
        model = model_info['model']
        features = model_info['features']
        baseline_features = np.zeros(len(features)).reshape(1, -1)
        baseline_prediction = model.predict(baseline_features)[0]
        modified_features = baseline_features.copy()
        for feature, change in feature_changes.items():
            if feature in features:
                feature_idx = features.index(feature)
                modified_features[0, feature_idx] = change
        modified_prediction = model.predict(modified_features)[0]
        impact = modified_prediction - baseline_prediction
        return {'baseline_prediction': baseline_prediction, 'modified_prediction': modified_prediction, 'impact': impact, 'percent_change': (impact / baseline_prediction * 100) if baseline_prediction != 0 else 0, 'changes': feature_changes}, "Analysis completed"
    
    def assess_risks(self, df, target_column, threshold_percentile=95):
        if target_column not in self.models:
            return None, "No model found"
        target_values = df[target_column].dropna()
        risk_threshold = np.percentile(target_values, threshold_percentile)
        high_risk_data = df[df[target_column] > risk_threshold]
        risk_factors = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if column != target_column:
                normal_mean = df[column].mean()
                risk_mean = high_risk_data[column].mean() if len(high_risk_data) > 0 else normal_mean
                if abs(risk_mean - normal_mean) > 0.1 * normal_mean:
                    risk_factors[column] = {'normal_average': normal_mean, 'high_risk_average': risk_mean, 'risk_multiplier': risk_mean / normal_mean if normal_mean != 0 else 1}
        return {'risk_threshold': risk_threshold, 'high_risk_cases': len(high_risk_data), 'risk_factors': risk_factors, 'risk_probability': len(high_risk_data) / len(df) * 100}, "Risk assessment completed"
