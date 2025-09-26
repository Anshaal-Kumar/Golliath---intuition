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