// Digital Twin Intelligence Platform - Frontend JavaScript
// Complete desktop application logic - FIXED VERSION

class DigitalTwinApp {
    constructor() {
        this.serverPort = 8501;
        this.serverUrl = `http://localhost:${this.serverPort}`;
        this.currentData = null;
        this.models = {};
        this.activityLog = [];
        this.serverCheckInterval = null;
        this.init();
    }

    async init() {
        console.log('Initializing Digital Twin Desktop Application...');
        
        // Get server port from main process
        try {
            if (window.electronAPI && typeof window.electronAPI.getServerPort === 'function') {
                this.serverPort = await window.electronAPI.getServerPort();
                this.serverUrl = `http://localhost:${this.serverPort}`;
                console.log(`Using server port: ${this.serverPort}`);
            }
        } catch (error) {
            console.log('Using default port 8501:', error.message);
        }

        this.setupEventListeners();
        this.setupNavigation();
        this.setupFileHandling();
        this.startServerHealthCheck();
        this.addActivity('System Started', 'Digital Twin Intelligence Platform initialized');
    }

    setupEventListeners() {
        console.log('Setting up event listeners...');

        // Navigation buttons
        document.querySelectorAll('.nav-button').forEach(button => {
            button.addEventListener('click', () => {
                const page = button.dataset.page;
                if (page) {
                    this.showPage(page);
                }
            });
        });

        // File input handler
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (event) => this.handleFileSelect(event));
        }

        // Process data button
        const processButton = document.getElementById('processButton');
        if (processButton) {
            processButton.addEventListener('click', () => this.processData());
        }

        // Analysis buttons
        const patternBtn = document.getElementById('patternAnalysisBtn');
        if (patternBtn) {
            patternBtn.addEventListener('click', () => this.runPatternAnalysis());
        }

        const anomalyBtn = document.getElementById('anomalyDetectionBtn');
        if (anomalyBtn) {
            anomalyBtn.addEventListener('click', () => this.detectAnomalies());
        }

        // Model building
        const buildModelBtn = document.getElementById('buildModelBtn');
        if (buildModelBtn) {
            buildModelBtn.addEventListener('click', () => this.buildModel());
        }

        // Forecasting
        const forecastBtn = document.getElementById('forecastBtn');
        if (forecastBtn) {
            forecastBtn.addEventListener('click', () => this.generateForecast());
        }

        // Chat functionality
        const chatSendBtn = document.getElementById('chatSendBtn');
        const chatInput = document.getElementById('chatInput');
        
        if (chatSendBtn) {
            chatSendBtn.addEventListener('click', () => this.sendChatMessage());
        }
        
        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendChatMessage();
                }
            });
        }

        // Ontology query
        const queryBtn = document.getElementById('queryBtn');
        if (queryBtn) {
            queryBtn.addEventListener('click', () => this.queryOntology());
        }

        // Contamination slider
        const contaminationSlider = document.getElementById('contaminationSlider');
        const contaminationValue = document.getElementById('contaminationValue');
        if (contaminationSlider && contaminationValue) {
            contaminationSlider.addEventListener('input', (e) => {
                contaminationValue.textContent = e.target.value;
            });
        }

        // Window resize handler
        window.addEventListener('resize', () => this.handleResize());

        // Handle menu actions from main process
        if (window.electronAPI && typeof window.electronAPI.onFileSelected === 'function') {
            window.electronAPI.onFileSelected((event, filePath) => {
                this.handleFileFromMenu(filePath);
            });
        }
    }

    setupNavigation() {
        // Initialize with dashboard page
        this.showPage('dashboard');
    }

    setupFileHandling() {
        // Handle drag and drop
        const uploadZone = document.getElementById('uploadZone');
        if (uploadZone) {
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });

            uploadZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
            });

            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const file = files[0];
                    if (file.type === 'text/csv' || file.name.toLowerCase().endsWith('.csv')) {
                        this.handleFileSelect({ target: { files } });
                    } else {
                        this.showNotification('Please drop a CSV file', 'warning');
                    }
                }
            });

            // Handle click on upload zone
            uploadZone.addEventListener('click', () => {
                const fileInput = document.getElementById('fileInput');
                if (fileInput) {
                    fileInput.click();
                }
            });
        }
    }

    showPage(pageId) {
        if (!pageId) return;
        
        console.log(`Switching to page: ${pageId}`);

        // Hide all pages
        document.querySelectorAll('.page').forEach(page => {
            page.style.display = 'none';
        });

        // Show selected page
        const targetPage = document.getElementById(pageId);
        if (targetPage) {
            targetPage.style.display = 'block';
        }

        // Update navigation
        document.querySelectorAll('.nav-button').forEach(button => {
            button.classList.remove('active');
        });
        
        const activeButton = document.querySelector(`[data-page="${pageId}"]`);
        if (activeButton) {
            activeButton.classList.add('active');
        }

        // Page-specific initialization
        if (pageId === 'simulations') {
            this.populateTargetColumns();
        } else if (pageId === 'ontology') {
            this.loadKnowledgeGraph();
        }

        this.addActivity('Navigation', `Switched to ${pageId} page`);
    }

    startServerHealthCheck() {
        // Initial check
        this.checkServerStatus();
        
        // Set up periodic health checks
        this.serverCheckInterval = setInterval(() => {
            this.checkServerStatus();
        }, 10000); // Check every 10 seconds
    }

    async checkServerStatus() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch(`${this.serverUrl}/health`, { 
                signal: controller.signal,
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                const data = await response.json();
                this.updateServerStatus('Connected', true);
                this.updateSystemStats();
                console.log('Server health check passed:', data);
            } else {
                this.updateServerStatus('Error', false);
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Server health check timeout');
            } else {
                console.log('Server not ready:', error.message);
            }
            this.updateServerStatus('Connecting...', false);
        }
    }

    updateServerStatus(status, connected) {
        const statusElement = document.getElementById('serverStatus');
        if (statusElement) {
            const statusText = statusElement.querySelector('.status-text');
            const statusDot = statusElement.querySelector('.status-dot');

            if (statusText) statusText.textContent = status;
            if (statusDot) {
                statusDot.style.background = connected ? '#10b981' : '#ef4444';
            }
        }
    }

    async updateSystemStats() {
        try {
            const response = await fetch(`${this.serverUrl}/api/system/status`);
            if (!response.ok) return;
            
            const status = await response.json();

            // Update sidebar stats
            this.updateElement('totalRecords', status.data_shape ? status.data_shape[0] : 0);
            this.updateElement('totalModels', status.models_count || 0);
            this.updateElement('totalEntities', status.entities_count || 0);

            // Update dashboard stats
            this.updateElement('dashRecords', status.data_shape ? status.data_shape[0] : 0);
            this.updateElement('dashFeatures', status.data_shape ? status.data_shape[1] : 0);
            this.updateElement('dashModels', status.models_count || 0);
            this.updateElement('dashRelationships', status.relationships_count || 0);

        } catch (error) {
            console.error('Failed to update system stats:', error);
        }
    }

    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = String(value);
        }
    }

    handleFileSelect(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;

        const file = files[0];
        
        // Check file type more thoroughly
        const isCSV = file.type === 'text/csv' || 
                     file.type === 'application/csv' ||
                     file.name.toLowerCase().endsWith('.csv');
                     
        if (!isCSV) {
            this.showNotification('Please select a CSV file', 'warning');
            return;
        }

        console.log('File selected:', file.name);
        
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const fileInfo = document.getElementById('fileInfo');
        const sourceName = document.getElementById('sourceName');

        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = `Size: ${this.formatFileSize(file.size)}`;
        if (sourceName) sourceName.value = file.name.replace(/\.[^/.]+$/, ""); // Remove extension
        if (fileInfo) fileInfo.style.display = 'block';

        // Store file for processing
        this.selectedFile = file;
        this.addActivity('File Selected', `${file.name} ready for processing`);
    }

    handleFileFromMenu(filePath) {
        console.log('File selected from menu:', filePath);
        this.showNotification('Menu file selection not yet implemented', 'info');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async processData() {
        if (!this.selectedFile) {
            this.showNotification('Please select a file first', 'warning');
            return;
        }

        const sourceNameElement = document.getElementById('sourceName');
        const sourceName = sourceNameElement ? sourceNameElement.value || 'Unknown Source' : 'Unknown Source';
        
        this.showLoading('Processing data with AI...');
        this.addActivity('Processing', `Ingesting ${this.selectedFile.name}`);

        try {
            const fileContent = await this.readFileContent(this.selectedFile);
            
            const response = await fetch(`${this.serverUrl}/api/data/ingest`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    csv_content: fileContent,
                    source_name: sourceName
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showNotification(`✅ Successfully processed ${result.rows} rows!`, 'success');
                await this.showDataPreview();
                await this.updateSystemStats();
                this.addActivity('Data Processed', `${result.rows} rows, ${result.columns} columns`);
            } else {
                this.showNotification(`❌ Processing failed: ${result.error}`, 'error');
                this.addActivity('Processing Failed', result.error);
            }
        } catch (error) {
            console.error('Processing failed:', error);
            this.showNotification(`❌ Processing failed: ${error.message}`, 'error');
            this.addActivity('Processing Failed', error.message);
        } finally {
            this.hideLoading();
        }
    }

    readFileContent(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                if (e.target && e.target.result) {
                    resolve(e.target.result);
                } else {
                    reject(new Error('Failed to read file content'));
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    async showDataPreview() {
        try {
            const response = await fetch(`${this.serverUrl}/api/data/current`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();

            if (data.shape && data.columns && data.sample) {
                const previewDiv = document.getElementById('dataPreview');
                const tableDiv = document.getElementById('dataTable');

                if (previewDiv && tableDiv) {
                    let tableHTML = '<div class="data-table"><table><thead><tr>';
                    data.columns.forEach(col => {
                        tableHTML += `<th>${this.escapeHtml(col)}</th>`;
                    });
                    tableHTML += '</tr></thead><tbody>';

                    data.sample.slice(0, 10).forEach(row => {
                        tableHTML += '<tr>';
                        data.columns.forEach(col => {
                            const value = row[col];
                            let displayValue = '';
                            if (value !== null && value !== undefined) {
                                if (typeof value === 'number' && value % 1 !== 0) {
                                    displayValue = value.toFixed(2);
                                } else {
                                    displayValue = String(value);
                                }
                            }
                            tableHTML += `<td>${this.escapeHtml(displayValue)}</td>`;
                        });
                        tableHTML += '</tr>';
                    });
                    tableHTML += '</tbody></table></div>';

                    tableDiv.innerHTML = tableHTML;
                    previewDiv.style.display = 'block';
                    this.currentData = data;
                }
            }
        } catch (error) {
            console.error('Failed to show preview:', error);
            this.showNotification('Failed to load data preview', 'error');
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async runPatternAnalysis() {
        if (!this.currentData) {
            this.showNotification('Please load data first', 'warning');
            return;
        }

        this.showLoading('Analyzing patterns with AI...');
        this.addActivity('Analysis', 'Running pattern analysis');

        try {
            const response = await fetch(`${this.serverUrl}/api/analysis/patterns`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.success) {
                this.displayPatternResults(result.insights);
                const correlationCount = result.insights.correlations ? result.insights.correlations.length : 0;
                this.addActivity('Analysis Complete', `Found ${correlationCount} correlations`);
            } else {
                this.showNotification(`Analysis failed: ${result.error}`, 'error');
                this.addActivity('Analysis Failed', result.error);
            }
        } catch (error) {
            console.error('Pattern analysis failed:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
            this.addActivity('Analysis Failed', error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayPatternResults(insights) {
        const resultsDiv = document.getElementById('patternResults');
        if (!resultsDiv) return;

        let html = '<div class="analysis-results">';

        if (insights && insights.correlations && insights.correlations.length > 0) {
            html += '<h4>🔗 Strong Correlations Found:</h4><ul>';
            insights.correlations.forEach(corr => {
                const strength = Math.abs(corr.correlation);
                const strengthText = strength > 0.8 ? 'very strong' : strength > 0.6 ? 'strong' : 'moderate';
                html += `<li><strong>${this.escapeHtml(corr.feature1)}</strong> ↔ <strong>${this.escapeHtml(corr.feature2)}</strong>: ${corr.correlation.toFixed(3)} (${strengthText})</li>`;
            });
            html += '</ul>';
        } else {
            html += '<p>No strong correlations found in the data.</p>';
        }

        if (insights && insights.clusters) {
            html += `<h4>📊 Data Segmentation:</h4>`;
            html += `<p>Discovered <strong>${insights.clusters.n_clusters}</strong> distinct patterns in your data using unsupervised clustering.</p>`;
        }

        html += '</div>';
        resultsDiv.innerHTML = html;
    }

    async detectAnomalies() {
        if (!this.currentData) {
            this.showNotification('Please load data first', 'warning');
            return;
        }

        const contaminationSlider = document.getElementById('contaminationSlider');
        const contamination = contaminationSlider ? parseFloat(contaminationSlider.value) : 0.1;
        
        this.showLoading('Detecting anomalies with AI...');
        this.addActivity('Analysis', `Detecting anomalies (sensitivity: ${contamination})`);

        try {
            const response = await fetch(`${this.serverUrl}/api/analysis/anomalies`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ contamination })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayAnomalyResults(result);
            this.addActivity('Anomalies Detected', `Found ${result.anomaly_count || 0} anomalies`);
        } catch (error) {
            console.error('Anomaly detection failed:', error);
            this.showNotification(`Anomaly detection failed: ${error.message}`, 'error');
            this.addActivity('Analysis Failed', error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayAnomalyResults(result) {
        const resultsDiv = document.getElementById('anomalyResults');
        if (!resultsDiv) return;

        let html = '<div class="analysis-results">';
        html += `<h4>🚨 Anomaly Detection Results:</h4>`;
        
        const anomalyCount = result.anomaly_count || 0;
        
        if (anomalyCount > 0) {
            const totalRecords = this.currentData && this.currentData.shape ? this.currentData.shape[0] : 1;
            html += `<p>Found <strong>${anomalyCount}</strong> anomalous data points out of ${totalRecords} total records.</p>`;
            html += `<p><strong>Anomaly Rate:</strong> ${((anomalyCount / totalRecords) * 100).toFixed(2)}%</p>`;
            
            if (result.anomaly_indices && result.anomaly_indices.length > 0) {
                const displayIndices = result.anomaly_indices.slice(0, 10);
                html += `<p><strong>Sample anomaly indices:</strong> ${displayIndices.join(', ')}`;
                if (result.anomaly_indices.length > 10) {
                    html += ` <em>(showing first 10 of ${result.anomaly_indices.length})</em>`;
                }
                html += '</p>';
            }
        } else {
            html += '<p>No anomalies detected with the current sensitivity settings.</p>';
            html += '<p><em>Try adjusting the sensitivity slider to detect more subtle anomalies.</em></p>';
        }
        
        html += '</div>';
        resultsDiv.innerHTML = html;
    }

    populateTargetColumns() {
        if (!this.currentData) return;

        const targetSelect = document.getElementById('targetColumn');
        if (!targetSelect) return;

        targetSelect.innerHTML = '<option value="">Choose a numeric column...</option>';

        let hasNumericColumns = false;

        if (this.currentData.columns && this.currentData.dtypes) {
            this.currentData.columns.forEach(column => {
                const dataType = this.currentData.dtypes[column];
                if (dataType && (dataType.includes('int') || dataType.includes('float'))) {
                    const option = document.createElement('option');
                    option.value = column;
                    option.textContent = column;
                    targetSelect.appendChild(option);
                    hasNumericColumns = true;
                }
            });
        }

        if (!hasNumericColumns) {
            targetSelect.innerHTML = '<option value="">No numeric columns found</option>';
        }
    }

    async buildModel() {
        const targetSelect = document.getElementById('targetColumn');
        const targetColumn = targetSelect ? targetSelect.value : '';
        
        if (!targetColumn) {
            this.showNotification('Please select a target column', 'warning');
            return;
        }

        this.showLoading('Building AI predictive model...');
        this.addActivity('Modeling', `Building model for ${targetColumn}`);

        try {
            const response = await fetch(`${this.serverUrl}/api/simulation/model/build`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ target_column: targetColumn })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.success) {
                this.displayModelResults(result);
                this.models[targetColumn] = result;
                
                const forecastBtn = document.getElementById('forecastBtn');
                if (forecastBtn) {
                    forecastBtn.disabled = false;
                }
                
                await this.updateSystemStats();
                this.addActivity('Model Built', `${targetColumn} prediction model (MAE: ${result.mae.toFixed(4)})`);
            } else {
                this.showNotification(`Model building failed: ${result.error}`, 'error');
                this.addActivity('Model Failed', result.error);
            }
        } catch (error) {
            console.error('Model building failed:', error);
            this.showNotification(`Model building failed: ${error.message}`, 'error');
            this.addActivity('Model Failed', error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayModelResults(result) {
        const resultsDiv = document.getElementById('modelResults');
        if (!resultsDiv) return;

        let html = '<div class="model-results">';
        html += `<h4>🤖 AI Model Successfully Built!</h4>`;
        html += `<p><strong>Target Variable:</strong> ${this.escapeHtml(result.target_column)}</p>`;
        
        if (result.features && Array.isArray(result.features)) {
            html += `<p><strong>Input Features:</strong> ${result.features.map(f => this.escapeHtml(f)).join(', ')}</p>`;
        }
        
        html += `<p><strong>Model Accuracy (MAE):</strong> ${result.mae.toFixed(4)}</p>`;
        
        if (result.feature_importance) {
            html += '<h5>📊 Feature Importance (Top 5):</h5><ul>';
            const sortedFeatures = Object.entries(result.feature_importance)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5);
                
            sortedFeatures.forEach(([feature, importance]) => {
                const percentage = (importance * 100).toFixed(1);
                html += `<li><strong>${this.escapeHtml(feature)}:</strong> ${percentage}%</li>`;
            });
            html += '</ul>';
        }
        
        html += '<p class="model-info">💡 <em>Lower MAE indicates better model accuracy</em></p>';
        html += '</div>';
        
        resultsDiv.innerHTML = html;
    }

    async generateForecast() {
        const targetSelect = document.getElementById('targetColumn');
        const forecastPeriodsInput = document.getElementById('forecastPeriods');
        
        const targetColumn = targetSelect ? targetSelect.value : '';
        const periods = forecastPeriodsInput ? parseInt(forecastPeriodsInput.value) : 30;

        if (!this.models[targetColumn]) {
            this.showNotification('Please build a model first', 'warning');
            return;
        }

        if (periods < 1 || periods > 365) {
            this.showNotification('Forecast periods must be between 1 and 365', 'warning');
            return;
        }

        this.showLoading(`Generating ${periods}-period forecast...`);
        this.addActivity('Forecasting', `Generating ${periods}-day forecast for ${targetColumn}`);

        try {
            const response = await fetch(`${this.serverUrl}/api/simulation/forecast`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    target_column: targetColumn,
                    periods: periods
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.success && result.forecast) {
                this.displayForecastResults(result);
                this.plotForecast(result.forecast, targetColumn);
                this.addActivity('Forecast Complete', `${periods}-day forecast generated`);
            } else {
                this.showNotification(`Forecast failed: ${result.error || 'Unknown error'}`, 'error');
                this.addActivity('Forecast Failed', result.error || 'Unknown error');
            }
        } catch (error) {
            console.error('Forecast failed:', error);
            this.showNotification(`Forecast failed: ${error.message}`, 'error');
            this.addActivity('Forecast Failed', error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayForecastResults(result) {
        const resultsDiv = document.getElementById('forecastResults');
        if (!resultsDiv || !result.forecast || !Array.isArray(result.forecast)) return;

        const avgValue = result.forecast.reduce((sum, item) => sum + (item.predicted_value || 0), 0) / result.forecast.length;
        const values = result.forecast.map(item => item.predicted_value || 0);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);

        let html = '<div class="forecast-results">';
        html += `<h4>📈 Forecast Generated Successfully!</h4>`;
        html += `<p><strong>Forecast Period:</strong> ${result.periods} days</p>`;
        html += `<p><strong>Average Predicted Value:</strong> ${avgValue.toFixed(2)}</p>`;
        html += `<p><strong>Range:</strong> ${minValue.toFixed(2)} to ${maxValue.toFixed(2)}</p>`;
        html += '<p>📊 <em>Chart visualization displayed below</em></p>';
        html += '</div>';
        
        resultsDiv.innerHTML = html;
    }

    plotForecast(forecastData, targetColumn) {
        const chartsDiv = document.getElementById('simulationCharts');
        const plotDiv = document.getElementById('simulationPlot');
        
        if (!plotDiv || !window.Plotly || !Array.isArray(forecastData)) {
            console.error('Plotly not available, plot div not found, or invalid forecast data');
            return;
        }
        
        try {
            const dates = forecastData.map(item => item.date);
            const values = forecastData.map(item => item.predicted_value || 0);
            const confidenceIntervals = forecastData.map(item => item.confidence_interval || 0);
            const upperBound = values.map((val, i) => val + confidenceIntervals[i]);
            const lowerBound = values.map((val, i) => val - confidenceIntervals[i]);

            const traces = [
                {
                    x: dates,
                    y: upperBound,
                    mode: 'lines',
                    line: { width: 0 },
                    showlegend: false,
                    hoverinfo: 'skip',
                    fillcolor: 'rgba(59, 130, 246, 0.2)'
                },
                {
                    x: dates,
                    y: lowerBound,
                    mode: 'lines',
                    fill: 'tonexty',
                    fillcolor: 'rgba(59, 130, 246, 0.2)',
                    line: { width: 0 },
                    name: 'Confidence Interval',
                    hoverinfo: 'skip'
                },
                {
                    x: dates,
                    y: values,
                    mode: 'lines+markers',
                    name: 'Forecast',
                    line: { color: '#3b82f6', width: 3 },
                    marker: { size: 4, color: '#3b82f6' }
                }
            ];

            const layout = {
                title: {
                    text: `${targetColumn} Forecast`,
                    font: { size: 18, color: '#e2e8f0' }
                },
                xaxis: { 
                    title: 'Date',
                    color: '#e2e8f0',
                    gridcolor: '#4b5563'
                },
                yaxis: { 
                    title: targetColumn,
                    color: '#e2e8f0',
                    gridcolor: '#4b5563'
                },
                plot_bgcolor: '#374151',
                paper_bgcolor: '#374151',
                font: { color: '#e2e8f0' },
                legend: {
                    bgcolor: 'rgba(55, 65, 81, 0.8)',
                    bordercolor: '#4b5563',
                    borderwidth: 1
                }
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                displaylogo: false
            };

            Plotly.newPlot(plotDiv, traces, layout, config);
            
            if (chartsDiv) {
                chartsDiv.style.display = 'block';
            }
        } catch (error) {
            console.error('Failed to plot forecast:', error);
            this.showNotification('Failed to generate forecast chart', 'error');
        }
    }

    async sendChatMessage() {
        const chatInput = document.getElementById('chatInput');
        if (!chatInput) return;

        const message = chatInput.value.trim();
        if (!message) return;

        // Add user message to chat
        this.addChatMessage(message, 'user');
        chatInput.value = '';

        // Show typing indicator
        const typingId = this.addChatMessage('🤔 Thinking...', 'assistant');

        try {
            const response = await fetch(`${this.serverUrl}/api/chat/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: message })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            // Remove typing indicator
            this.removeChatMessage(typingId);
            
            // Add AI response
            this.addChatMessage(result.response || 'No response received', 'assistant');
            this.addActivity('AI Chat', `User asked: "${message.substring(0, 30)}..."`);
        } catch (error) {
            // Remove typing indicator
            this.removeChatMessage(typingId);
            
            console.error('Chat query failed:', error);
            this.addChatMessage('Sorry, I encountered an error processing your question. Please make sure you have loaded some data and try again.', 'assistant');
            this.addActivity('Chat Error', error.message);
        }
    }

    addChatMessage(message, sender) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return null;

        const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;
        messageDiv.id = messageId;
        
        const avatar = sender === 'user' ? '👤' : '🤖';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.formatChatMessage(message)}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageId;
    }

    removeChatMessage(messageId) {
        if (messageId) {
            const messageElement = document.getElementById(messageId);
            if (messageElement && messageElement.parentNode) {
                messageElement.parentNode.removeChild(messageElement);
            }
        }
    }

    formatChatMessage(message) {
        if (!message) return '';
        
        // Basic formatting for chat messages
        return this.escapeHtml(message)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/\n/g, '<br>') // Line breaks
            .replace(/•/g, '•'); // Bullet points
    }

    async queryOntology() {
        const queryInput = document.getElementById('ontologyQuery');
        if (!queryInput) return;

        const query = queryInput.value.trim();
        if (!query) {
            this.showNotification('Please enter a query', 'warning');
            return;
        }

        this.showLoading('Querying knowledge graph...');
        this.addActivity('Ontology Query', query);

        try {
            const response = await fetch(`${this.serverUrl}/api/ontology/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayQueryResults(result);
            this.addActivity('Query Complete', `Found ${result.results ? result.results.length : 0} results`);
        } catch (error) {
            console.error('Ontology query failed:', error);
            this.showNotification(`Query failed: ${error.message}`, 'error');
            this.addActivity('Query Failed', error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayQueryResults(result) {
        const resultsDiv = document.getElementById('queryResults');
        if (!resultsDiv) return;

        let html = '<div class="query-results">';
        html += `<h4>🔍 Query Results for: "${this.escapeHtml(result.query || '')}"</h4>`;
        
        if (result.results && Array.isArray(result.results) && result.results.length > 0) {
            html += '<ul>';
            result.results.forEach(item => {
                html += `<li>${this.escapeHtml(String(item))}</li>`;
            });
            html += '</ul>';
        } else {
            html += '<p>No results found. Try a different query or load some data first.</p>';
            html += '<p><em>Example queries: "show correlations", "list entities", "describe relationships"</em></p>';
        }
        
        html += '</div>';
        resultsDiv.innerHTML = html;
    }

    async loadKnowledgeGraph() {
        if (!this.currentData) {
            const graphContainer = document.getElementById('knowledgeGraph');
            if (graphContainer) {
                graphContainer.innerHTML = '<div class="loading-message">Load and analyze data first to build knowledge graph</div>';
            }
            return;
        }

        try {
            const response = await fetch(`${this.serverUrl}/api/ontology/graph`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.displayKnowledgeGraph(result);
            this.displayOntologyInfo(result);
        } catch (error) {
            console.error('Failed to load knowledge graph:', error);
            const graphContainer = document.getElementById('knowledgeGraph');
            if (graphContainer) {
                graphContainer.innerHTML = '<div class="loading-message">Failed to load knowledge graph</div>';
            }
        }
    }

    displayKnowledgeGraph(data) {
        const graphContainer = document.getElementById('knowledgeGraph');
        if (!graphContainer) return;

        if (!data || !data.nodes || !Array.isArray(data.nodes) || data.nodes.length === 0) {
            graphContainer.innerHTML = '<div class="loading-message">No knowledge graph data available</div>';
            return;
        }

        // Simple text-based representation of the knowledge graph
        let html = '<div class="graph-display">';
        html += '<h4>📊 Data Entities</h4><ul>';
        
        data.nodes.forEach(node => {
            html += `<li><strong>${this.escapeHtml(node.label || node.id || 'Unknown')}</strong> (${this.escapeHtml(node.type || 'entity')})`;
            if (node.data_type) {
                html += ` - ${this.escapeHtml(node.data_type)}`;
            }
            html += '</li>';
        });
        
        html += '</ul>';

        if (data.edges && Array.isArray(data.edges) && data.edges.length > 0) {
            html += '<h4>🔗 Relationships</h4><ul>';
            data.edges.forEach(edge => {
                html += `<li><strong>${this.escapeHtml(edge.source || '')}</strong> → <strong>${this.escapeHtml(edge.target || '')}</strong>`;
                if (edge.relationship) {
                    html += ` (${this.escapeHtml(edge.relationship)})`;
                }
                if (typeof edge.strength === 'number') {
                    html += ` - strength: ${edge.strength.toFixed(2)}`;
                }
                html += '</li>';
            });
            html += '</ul>';
        }
        
        html += '</div>';
        graphContainer.innerHTML = html;
    }

    displayOntologyInfo(data) {
        const ontologyInfo = document.getElementById('ontologyInfo');
        if (!ontologyInfo) return;

        let html = '<div class="ontology-display">';
        
        if (data && data.entities && typeof data.entities === 'object' && Object.keys(data.entities).length > 0) {
            html += `<h4>📋 Entities (${Object.keys(data.entities).length})</h4>`;
            html += '<div class="entity-list">';
            
            Object.entries(data.entities).slice(0, 10).forEach(([name, entity]) => {
                html += `<div class="entity-item">`;
                html += `<strong>${this.escapeHtml(name)}</strong> - ${this.escapeHtml(entity.type || 'unknown')}`;
                if (entity.data_type) {
                    html += ` (${this.escapeHtml(entity.data_type)})`;
                }
                html += `</div>`;
            });
            
            if (Object.keys(data.entities).length > 10) {
                html += `<p><em>... and ${Object.keys(data.entities).length - 10} more</em></p>`;
            }
            
            html += '</div>';
        }
        
        if (data && data.relationships && typeof data.relationships === 'object' && Object.keys(data.relationships).length > 0) {
            html += `<h4>🔗 Relationships (${Object.keys(data.relationships).length})</h4>`;
            html += '<div class="relationship-list">';
            
            Object.entries(data.relationships).slice(0, 5).forEach(([name, rel]) => {
                if (name && name.includes('-')) {
                    const [source, target] = name.split('-');
                    html += `<div class="relationship-item">`;
                    html += `<strong>${this.escapeHtml(source)}</strong> ↔ <strong>${this.escapeHtml(target)}</strong> - ${this.escapeHtml(rel.type || 'unknown')}`;
                    if (typeof rel.strength === 'number') {
                        html += ` (${rel.strength.toFixed(2)})`;
                    }
                    html += `</div>`;
                }
            });
            
            html += '</div>';
        }
        
        html += '</div>';
        ontologyInfo.innerHTML = html;
    }

    addActivity(type, description) {
        const timestamp = new Date().toLocaleTimeString();
        this.activityLog.unshift({ timestamp, type: String(type), description: String(description) });
        
        // Keep only last 10 activities
        if (this.activityLog.length > 10) {
            this.activityLog = this.activityLog.slice(0, 10);
        }
        
        this.updateActivityFeed();
    }

    updateActivityFeed() {
        const activityFeed = document.getElementById('activityFeed');
        if (!activityFeed) return;

        let html = '';
        this.activityLog.forEach(activity => {
            html += `
                <div class="activity-item">
                    <span class="activity-time">${this.escapeHtml(activity.timestamp)}</span>
                    <span class="activity-desc"><strong>${this.escapeHtml(activity.type)}:</strong> ${this.escapeHtml(activity.description)}</span>
                </div>
            `;
        });
        
        activityFeed.innerHTML = html;
    }

    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const messageElement = document.getElementById('loadingMessage');
        
        if (overlay) {
            overlay.style.display = 'flex';
        }
        if (messageElement) {
            messageElement.textContent = String(message);
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = String(message);
        
        // Style the notification
        const styles = {
            position: 'fixed',
            top: '80px',
            right: '20px',
            padding: '16px 20px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: '500',
            zIndex: '10000',
            maxWidth: '400px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            animation: 'slideIn 0.3s ease-out',
            fontSize: '14px',
            lineHeight: '1.4'
        };

        Object.assign(notification.style, styles);

        // Set color based on type
        const colors = {
            success: '#059669',
            error: '#dc2626',
            warning: '#d97706',
            info: '#3b82f6'
        };
        
        notification.style.background = colors[type] || colors.info;

        // Add to page
        document.body.appendChild(notification);

        // Remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOut 0.3s ease-in';
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }
        }, 5000);
    }

    handleResize() {
        // Handle window resize events
        if (window.Plotly) {
            const plotDiv = document.getElementById('simulationPlot');
            if (plotDiv && plotDiv.data) {
                try {
                    Plotly.Plots.resize(plotDiv);
                } catch (error) {
                    console.warn('Failed to resize Plotly chart:', error);
                }
            }
        }
    }

    // Cleanup method
    destroy() {
        if (this.serverCheckInterval) {
            clearInterval(this.serverCheckInterval);
            this.serverCheckInterval = null;
        }
        
        // Clean up event listeners
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('error', this.handleError);
        window.removeEventListener('unhandledrejection', this.handleUnhandledRejection);
    }
}

// Utility functions for global access
function goToPage(pageId) {
    if (window.app && typeof window.app.showPage === 'function') {
        window.app.showPage(pageId);
    }
}

function sendSuggestion(suggestion) {
    const chatInput = document.getElementById('chatInput');
    if (chatInput && window.app && typeof window.app.sendChatMessage === 'function') {
        chatInput.value = String(suggestion);
        window.app.sendChatMessage();
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing Digital Twin App...');
    
    try {
        window.app = new DigitalTwinApp();
        
        // Add additional animation styles
        const additionalStyles = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        
        const styleSheet = document.createElement('style');
        styleSheet.textContent = additionalStyles;
        document.head.appendChild(styleSheet);
        
    } catch (error) {
        console.error('Failed to initialize application:', error);
        document.body.innerHTML = '<div style="padding: 20px; color: red;">Failed to initialize application. Please refresh the page.</div>';
    }
});

// Handle app errors gracefully
window.addEventListener('error', (event) => {
    console.error('Application error:', event.error);
    if (window.app && typeof window.app.addActivity === 'function') {
        window.app.addActivity('Error', event.error ? event.error.message : 'Unknown error');
    }
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    if (window.app && typeof window.app.addActivity === 'function') {
        window.app.addActivity('Promise Error', event.reason ? event.reason.toString() : 'Unknown promise error');
    }
});

// Handle page unload cleanup
window.addEventListener('beforeunload', () => {
    if (window.app && typeof window.app.destroy === 'function') {
        window.app.destroy();
    }
});