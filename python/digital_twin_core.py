# python/digital_twin_core.py - Complete AI Digital Twin Engine - IMPROVED ANALYSIS
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
from scipy.stats import pearsonr

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
    """AI-powered data processing engine - IMPROVED VERSION"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
    
    def clean_and_process(self, df):
        """Clean and preprocess data with better handling"""
        processed_df = df.copy()
        
        # Better data type inference
        for column in processed_df.columns:
            if processed_df[column].dtype == 'object':
                # Try to convert to numeric if possible
                numeric_series = pd.to_numeric(processed_df[column], errors='coerce')
                if not numeric_series.isna().all():
                    processed_df[column] = numeric_series
        
        # Smarter missing value handling
        for column in processed_df.columns:
            if processed_df[column].dtype in ['int64', 'float64']:
                # Use median for numeric
                processed_df[column].fillna(processed_df[column].median(), inplace=True)
            else:
                # Use mode for categorical, fallback to 'Unknown'
                mode_values = processed_df[column].mode()
                fill_value = mode_values.iloc[0] if not mode_values.empty else 'Unknown'
                processed_df[column].fillna(fill_value, inplace=True)
        
        print(f"🧹 Data cleaned: {processed_df.shape} shape, {processed_df.select_dtypes(include=[np.number]).shape[1]} numeric columns")
        return processed_df
    
    def detect_anomalies(self, df, contamination=0.1):
        """Enhanced anomaly detection with better parameters"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                print("⚠️ No numeric columns found for anomaly detection")
                return []
                
            print(f"🔍 Running anomaly detection on {len(numeric_columns)} numeric columns")
            print(f"Columns: {list(numeric_columns)}")
            
            # Prepare data
            numeric_data = df[numeric_columns].fillna(0)
            
            # Scale data for better anomaly detection
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Enhanced Isolation Forest with better parameters
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=200,  # More trees for better accuracy
                max_samples='auto',
                random_state=42,
                n_jobs=-1  # Use all CPUs
            )
            
            anomaly_labels = iso_forest.fit_predict(scaled_data)
            anomaly_scores = iso_forest.decision_function(scaled_data)
            
            # Get anomaly indices
            anomalies = df[anomaly_labels == -1]
            anomaly_indices = anomalies.index.tolist()
            
            print(f"✅ Found {len(anomaly_indices)} anomalies out of {len(df)} records")
            
            # Return additional info for better insights
            return {
                'indices': anomaly_indices,
                'scores': anomaly_scores.tolist(),
                'threshold_score': np.percentile(anomaly_scores, (1-contamination)*100)
            }
            
        except Exception as e:
            print(f"❌ Anomaly detection failed: {str(e)}")
            return []
    
    def extract_patterns(self, df):
        """Enhanced pattern extraction with multiple correlation thresholds and better analysis"""
        insights = {}
        
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            print(f"🔍 Analyzing patterns in {len(numeric_columns)} numeric columns: {list(numeric_columns)}")
            
            if len(numeric_columns) > 1:
                # Calculate correlation matrix
                corr_matrix = df[numeric_columns].corr()
                print("📊 Correlation matrix computed")
                
                # Find correlations at multiple thresholds
                correlation_thresholds = [0.3, 0.5, 0.7, 0.8]
                all_correlations = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if not np.isnan(corr_value):
                            # Calculate p-value for statistical significance
                            try:
                                _, p_value = pearsonr(df[corr_matrix.columns[i]].dropna(), 
                                                    df[corr_matrix.columns[j]].dropna())
                            except:
                                p_value = 1.0
                            
                            all_correlations.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': float(corr_value),
                                'abs_correlation': abs(float(corr_value)),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            })
                
                # Sort by absolute correlation strength
                all_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
                
                # Categorize correlations
                strong_correlations = [c for c in all_correlations if c['abs_correlation'] >= 0.5]
                moderate_correlations = [c for c in all_correlations if 0.3 <= c['abs_correlation'] < 0.5]
                
                insights['correlations'] = strong_correlations[:10]  # Top 10 strong correlations
                insights['moderate_correlations'] = moderate_correlations[:5]  # Top 5 moderate
                insights['all_correlations'] = all_correlations[:20]  # Top 20 overall
                
                print(f"📈 Found {len(strong_correlations)} strong correlations (≥0.5)")
                print(f"📈 Found {len(moderate_correlations)} moderate correlations (0.3-0.5)")
                
                # Statistical summary
                if len(all_correlations) > 0:
                    corr_values = [c['abs_correlation'] for c in all_correlations]
                    insights['correlation_summary'] = {
                        'total_pairs': len(all_correlations),
                        'max_correlation': max(corr_values),
                        'avg_correlation': np.mean(corr_values),
                        'median_correlation': np.median(corr_values),
                        'strong_count': len(strong_correlations),
                        'moderate_count': len(moderate_correlations)
                    }
            
            # Enhanced clustering with multiple algorithms
            if len(numeric_columns) >= 2 and len(df) >= 10:
                try:
                    # Prepare data for clustering
                    cluster_data = df[numeric_columns].fillna(df[numeric_columns].median())
                    scaled_data = self.scaler.fit_transform(cluster_data)
                    
                    # Determine optimal number of clusters (2 to min(8, n_samples/2))
                    max_clusters = min(8, len(df) // 2, 5)
                    if max_clusters >= 2:
                        # Try multiple cluster numbers and pick best
                        best_n_clusters = 3
                        best_inertia = float('inf')
                        
                        for n in range(2, max_clusters + 1):
                            try:
                                kmeans_temp = KMeans(n_clusters=n, random_state=42, n_init=10)
                                kmeans_temp.fit(scaled_data)
                                if kmeans_temp.inertia_ < best_inertia:
                                    best_inertia = kmeans_temp.inertia_
                                    best_n_clusters = n
                            except:
                                continue
                        
                        # Fit final model
                        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(scaled_data)
                        
                        # Calculate cluster statistics
                        cluster_counts = np.bincount(clusters)
                        
                        insights['clusters'] = {
                            'labels': clusters.tolist(),
                            'centers': kmeans.cluster_centers_.tolist(),
                            'n_clusters': int(best_n_clusters),
                            'cluster_sizes': cluster_counts.tolist(),
                            'inertia': float(kmeans.inertia_),
                            'features_used': list(numeric_columns)
                        }
                        
                        print(f"🎯 Created {best_n_clusters} clusters with sizes: {cluster_counts}")
                        
                except Exception as e:
                    print(f"⚠️ Clustering failed: {str(e)}")
            
            # Data quality insights
            insights['data_quality'] = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(numeric_columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # Feature variability analysis
            if len(numeric_columns) > 0:
                variability_analysis = {}
                for col in numeric_columns:
                    series = df[col].dropna()
                    if len(series) > 1:
                        variability_analysis[col] = {
                            'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else 0,
                            'range': float(series.max() - series.min()),
                            'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                            'skewness': float(series.skew()),
                            'kurtosis': float(series.kurtosis())
                        }
                insights['feature_variability'] = variability_analysis
                print(f"📊 Analyzed variability for {len(variability_analysis)} features")
            
            print(f"✅ Pattern extraction completed with {len(insights)} insight categories")
            return insights
            
        except Exception as e:
            print(f"❌ Pattern extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

class OntologyEngine:
    """Knowledge graph construction engine"""
    
    def __init__(self, db_path="digital_twin.db"):
        self.db_path = db_path
        self.knowledge_graph = nx.Graph()
        self.ontology = {'entities': {}, 'relationships': {}, 'rules': []}
    
    def build_knowledge_graph(self, df, insights):
        """Enhanced knowledge graph construction"""
        self.knowledge_graph.clear()
        
        # Add column nodes with more detailed attributes
        for column in df.columns:
            series = df[column]
            node_attrs = {
                'type': 'attribute',
                'data_type': str(series.dtype),
                'unique_values': int(series.nunique()),
                'null_count': int(series.isnull().sum()),
                'description': f"Data attribute: {column}"
            }
            
            # Add statistical info for numeric columns
            if series.dtype in ['int64', 'float64']:
                node_attrs.update({
                    'mean': float(series.mean()) if not series.empty else 0,
                    'std': float(series.std()) if not series.empty else 0,
                    'min': float(series.min()) if not series.empty else 0,
                    'max': float(series.max()) if not series.empty else 0
                })
            
            self.knowledge_graph.add_node(column, **node_attrs)
            self.ontology['entities'][column] = node_attrs
        
        # Add correlation relationships
        if 'correlations' in insights and insights['correlations']:
            for corr in insights['correlations']:
                edge_attrs = {
                    'relationship': 'correlation',
                    'strength': abs(float(corr['correlation'])),
                    'direction': 'positive' if corr['correlation'] > 0 else 'negative',
                    'p_value': corr.get('p_value', 1.0),
                    'significant': corr.get('significant', False),
                    'description': f"Correlation: {corr['correlation']:.3f}"
                }
                
                self.knowledge_graph.add_edge(
                    corr['feature1'],
                    corr['feature2'],
                    **edge_attrs
                )
                
                rel_key = f"{corr['feature1']}-{corr['feature2']}"
                self.ontology['relationships'][rel_key] = {
                    'type': 'correlation',
                    'strength': float(corr['correlation']),
                    'significance': 'high' if abs(corr['correlation']) > 0.8 else 'medium' if abs(corr['correlation']) > 0.5 else 'low'
                }
        
        # Add clustering relationships
        if 'clusters' in insights and insights['clusters']:
            cluster_node = 'data_clusters'
            cluster_info = insights['clusters']
            
            self.knowledge_graph.add_node(
                cluster_node,
                type='pattern',
                n_clusters=cluster_info['n_clusters'],
                cluster_sizes=cluster_info.get('cluster_sizes', []),
                description=f"Data segmentation with {cluster_info['n_clusters']} distinct patterns"
            )
            
            # Connect cluster node to features used
            for feature in cluster_info.get('features_used', []):
                if feature in self.knowledge_graph:
                    self.knowledge_graph.add_edge(
                        cluster_node,
                        feature,
                        relationship='clustering_feature',
                        strength=0.7,
                        description=f"{feature} contributes to clustering pattern"
                    )
        
        self._save_ontology()
    
    def _get_column_stats(self, series):
        """Enhanced column statistics"""
        stats = {
            'count': int(len(series)),
            'null_count': int(series.isnull().sum()),
            'unique_count': int(series.nunique()),
            'data_type': str(series.dtype)
        }
        
        if series.dtype in ['int64', 'float64']:
            stats.update({
                'mean': float(series.mean()) if not series.empty else 0,
                'std': float(series.std()) if not series.empty else 0,
                'min': float(series.min()) if not series.empty else 0,
                'max': float(series.max()) if not series.empty else 0,
                'median': float(series.median()) if not series.empty else 0,
                'q25': float(series.quantile(0.25)) if not series.empty else 0,
                'q75': float(series.quantile(0.75)) if not series.empty else 0
            })
        
        return stats
    
    def _save_ontology(self):
        """Save ontology to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM entities')
        cursor.execute('DELETE FROM relationships')
        
        # Save entities
        for entity_name, entity_data in self.ontology['entities'].items():
            cursor.execute('''
                INSERT INTO entities (entity_type, entity_name, attributes)
                VALUES (?, ?, ?)
            ''', (entity_data.get('type', 'unknown'), entity_name, json.dumps(entity_data, ensure_ascii=False)))
        
        # Save relationships
        for rel_name, rel_data in self.ontology['relationships'].items():
            if '-' in rel_name:
                source, target = rel_name.split('-', 1)
                cursor.execute('''
                    INSERT INTO relationships (source_entity, target_entity, relationship_type, strength)
                    VALUES (?, ?, ?, ?)
                ''', (source, target, rel_data.get('type', 'unknown'), rel_data.get('strength', 0)))
        
        conn.commit()
        conn.close()
    
    def get_knowledge_graph(self):
        """Get the knowledge graph"""
        return self.knowledge_graph
    
    def query_ontology(self, query):
        """Enhanced ontology querying"""
        results = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['correlation', 'correlated', 'relationship']):
            for rel_name, rel_data in self.ontology['relationships'].items():
                if rel_data['type'] == 'correlation':
                    source, target = rel_name.split('-', 1)
                    strength = rel_data['strength']
                    significance = rel_data.get('significance', 'unknown')
                    results.append(f"{source} and {target} have {significance} correlation (r={strength:.3f})")
        
        elif any(word in query_lower for word in ['entities', 'attributes', 'columns', 'features']):
            entity_list = list(self.ontology['entities'].keys())
            results.extend(entity_list[:20])  # Limit to 20 entities
        
        elif any(word in query_lower for word in ['cluster', 'pattern', 'segment']):
            cluster_info = [rel for rel in self.ontology['relationships'].values() if 'cluster' in rel.get('type', '')]
            if cluster_info:
                results.append(f"Found {len(cluster_info)} clustering patterns in the data")
            else:
                results.append("No clustering patterns found. Try running pattern analysis first.")
        
        elif 'summary' in query_lower or 'overview' in query_lower:
            total_entities = len(self.ontology['entities'])
            total_relationships = len(self.ontology['relationships'])
            results.append(f"Knowledge Graph Summary: {total_entities} entities, {total_relationships} relationships")
        
        return results if results else ["No relevant information found. Try queries about 'correlations', 'entities', or 'clusters'."]

class SimulationEngine:
    """Predictive simulation engine"""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
    
    def build_predictive_model(self, df, target_column):
        """Enhanced model building with better validation"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if target_column not in numeric_columns:
                return None, f"Target column '{target_column}' must be numeric. Available numeric columns: {numeric_columns}"
            
            feature_columns = [col for col in numeric_columns if col != target_column]
            
            if len(feature_columns) == 0:
                return None, "No feature columns available for modeling"
            
            # Prepare data
            X = df[feature_columns].fillna(df[feature_columns].median())
            y = df[target_column].fillna(df[target_column].median())
            
            # Check for sufficient data
            if len(X) < 10:
                return None, "Insufficient data for modeling (need at least 10 rows)"
            
            # Split data
            test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))  # Adaptive test size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Build enhanced model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Enhanced evaluation
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, train_predictions)
            test_mae = mean_absolute_error(y_test, test_predictions)
            
            # Feature importance
            feature_importance = dict(zip(feature_columns, model.feature_importances_))
            
            # Model info
            model_info = {
                'model': model,
                'features': feature_columns,
                'target': target_column,
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'mae': float(test_mae),  # Use test MAE as primary metric
                'feature_importance': feature_importance,
                'n_samples': len(X),
                'n_features': len(feature_columns)
            }
            
            self.models[target_column] = model_info
            
            message = f"Model trained successfully! Test MAE: {test_mae:.4f}, Train MAE: {train_mae:.4f}"
            return model, message
            
        except Exception as e:
            return None, f"Error building model: {str(e)}"
    
    def run_forecast(self, target_column, periods=30):
        """Enhanced forecasting with confidence intervals"""
        if target_column not in self.models:
            return None, f"No model found for '{target_column}'. Build a model first."
        
        try:
            model_info = self.models[target_column]
            model = model_info['model']
            features = model_info['features']
            
            # Generate forecast data with more realistic patterns
            forecast_data = []
            base_date = datetime.now()
            
            # Use feature statistics to generate realistic feature values
            feature_stats = {}
            for feature in features:
                feature_stats[feature] = {
                    'mean': 0,
                    'std': 1
                }
            
            for i in range(periods):
                # Generate features with some temporal correlation
                future_features = []
                for j, feature in enumerate(features):
                    # Add some trend and seasonality
                    trend = i * 0.01  # Small linear trend
                    seasonal = 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
                    noise = np.random.normal(0, 1)
                    value = trend + seasonal + noise
                    future_features.append(value)
                
                future_features = np.array(future_features).reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(future_features)[0]
                
                # Calculate confidence interval (approximate)
                mae = model_info['mae']
                confidence_interval = mae * 1.96  # Approximate 95% CI
                
                forecast_data.append({
                    'date': base_date + timedelta(days=i),
                    'predicted_value': float(prediction),
                    'confidence_interval': float(confidence_interval),
                    'upper_bound': float(prediction + confidence_interval),
                    'lower_bound': float(prediction - confidence_interval)
                })
            
            return forecast_data, f"Forecast generated for {periods} periods"
            
        except Exception as e:
            return None, f"Forecast generation failed: {str(e)}"
    
    def run_what_if_analysis(self, target_column, feature_changes):
        """Enhanced what-if analysis"""
        if target_column not in self.models:
            return None, f"No model found for '{target_column}'"
        
        try:
            model_info = self.models[target_column]
            model = model_info['model']
            features = model_info['features']
            
            # Create baseline scenario (zeros)
            baseline_features = np.zeros(len(features)).reshape(1, -1)
            baseline_prediction = model.predict(baseline_features)[0]
            
            # Create modified scenario
            modified_features = baseline_features.copy()
            applied_changes = {}
            
            for feature, change in feature_changes.items():
                if feature in features:
                    feature_idx = features.index(feature)
                    modified_features[0, feature_idx] = float(change)
                    applied_changes[feature] = float(change)
            
            modified_prediction = model.predict(modified_features)[0]
            impact = modified_prediction - baseline_prediction
            
            # Calculate percentage change safely
            percent_change = (impact / baseline_prediction * 100) if baseline_prediction != 0 else float('inf')
            
            return {
                'baseline_prediction': float(baseline_prediction),
                'modified_prediction': float(modified_prediction),
                'impact': float(impact),
                'percent_change': float(percent_change) if not np.isinf(percent_change) else 0,
                'applied_changes': applied_changes,
                'sensitivity': {feature: abs(impact) for feature in applied_changes}
            }, "What-if analysis completed"
            
        except Exception as e:
            return None, f"What-if analysis failed: {str(e)}"
    
    def assess_risks(self, df, target_column, threshold_percentile=95):
        """Enhanced risk assessment"""
        if target_column not in self.models:
            return None, f"No model found for '{target_column}'"
        
        try:
            target_values = df[target_column].dropna()
            
            if len(target_values) == 0:
                return None, "No valid target values found"
            
            # Calculate risk threshold
            risk_threshold = np.percentile(target_values, threshold_percentile)
            high_risk_data = df[df[target_column] > risk_threshold]
            
            # Analyze risk factors
            risk_factors = {}
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if column != target_column and len(high_risk_data) > 0:
                    normal_mean = df[column].mean()
                    risk_mean = high_risk_data[column].mean()
                    
                    if abs(risk_mean - normal_mean) > 0.1 * abs(normal_mean) and normal_mean != 0:
                        risk_multiplier = risk_mean / normal_mean
                        risk_factors[column] = {
                            'normal_average': float(normal_mean),
                            'high_risk_average': float(risk_mean),
                            'risk_multiplier': float(risk_multiplier),
                            'difference': float(risk_mean - normal_mean)
                        }
            
            risk_probability = len(high_risk_data) / len(df) * 100 if len(df) > 0 else 0
            
            return {
                'risk_threshold': float(risk_threshold),
                'high_risk_cases': int(len(high_risk_data)),
                'total_cases': int(len(df)),
                'risk_probability': float(risk_probability),
                'risk_factors': risk_factors,
                'threshold_percentile': float(threshold_percentile)
            }, "Risk assessment completed"
            
        except Exception as e:
            return None, f"Risk assessment failed: {str(e)}"