// Digital Twin Intelligence Platform - Frontend JavaScript
// Complete desktop application logic

class DigitalTwinApp {
    constructor() {
        this.serverPort = 8501;
        this.serverUrl = `http://localhost:${this.serverPort}`;
        this.currentData = null;
        this.models = {};
        this.activityLog = [];
        this.init();
    }

    async init() {
        console.log('Initializing Digital Twin Desktop Application...');
        
        // Get server port from main process
        try {
            if (window.electronAPI && window.electronAPI.getServerPort) {
                this.serverPort = await window.electronAPI.getServerPort();
                this.serverUrl = `http://localhost:${this.serverPort}`;
                console.log(`Using server port: ${this.serverPort}`);
            }
        } catch (error) {
            console.log('Using default port 8501');
        }

        this.setupEventListeners();
        this.setupNavigation();
        this.setupFileHandling();
        this.checkServerStatus();
        this.addActivity('System Started', 'Digital Twin Intelligence Platform initialized');
    }

    setupEventListeners() {
        console.log('Setting up event listeners...');

        // Navigation buttons
        document.querySelectorAll('.nav-button').forEach(button => {
            button.addEventListener('click', () => {
                const page = button.dataset.page;
                this.showPage(page);
            });
        });

        // File input handler
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Process data button
        const processButton = document.getElementById('processButton');
        if (processButton) {
            processButton.addEventListener('click', this.processData.bind(this));
        }

        // Analysis buttons
        const patternBtn = document.getElementById('patternAnalysisBtn');
        if (patternBtn) {
            patternBtn.addEventListener('click', this.runPatternAnalysis.bind(this));
        }

        const anomalyBtn = document.getElementById('anomalyDetectionBtn');
        if (anomalyBtn) {
            anomalyBtn.addEventListener('click', this.detectAnomalies.bind(this));
        }

        // Model building
        const buildModelBtn = document.getElementById('buildModelBtn');
        if (buildModelBtn) {
            buildModelBtn.addEventListener('click', this.buildModel.bind(this));
        }

        // Forecasting
        const forecastBtn = document.getElementById('forecastBtn');
        if (forecastBtn) {
            forecastBtn.addEventListener('click', this.generateForecast.bind(this));
        }

        // Chat functionality
        const chatSendBtn = document.getElementById('chatSendBtn');
        const chatInput = document.getElementById('chatInput');
        
        if (chatSendBtn) {
            chatSendBtn.addEventListener('click', this.sendChatMessage.bind(this));
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
            queryBtn.addEventListener('click', this.queryOntology.bind(this));
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
        window.addEventListener('resize', this.handleResize.bind(this));

        // Handle menu actions from main process
        if (window.electronAPI) {
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
                if (files.length > 0 && files[0].type === 'text/csv') {
                    this.handleFileSelect({ target: { files } });
                } else {
                    this.showNotification('Please drop a CSV file', 'warning');
                }
            });

            // Handle click on upload zone
            uploadZone.addEventListener('click', () => {
                document.getElementById('fileInput').click();
            });
        }
    }

    showPage(pageId) {
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

    async checkServerStatus() {
        try {
            const response = await fetch(`${this.serverUrl}/health`, { 
                timeout: 5000 
            });
            
            if (response.ok) {
                const data = await response.json();
                this.updateServerStatus('Connected', true);
                this.updateSystemStats();
                console.log('Server health check passed:', data);
            } else {
                this.updateServerStatus('Error', false);
            }
        } catch (error) {
            console.log('Server not ready, retrying...', error.message);
            this.updateServerStatus('Connecting...', false);
            // Retry after 2 seconds
            setTimeout(() => this.checkServerStatus(), 2000);
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
            element.textContent = value;
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (file.type !== 'text/csv') {
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
        if (sourceName) sourceName.value = file.name.replace('.csv', '');
        if (fileInfo) fileInfo.style.display = 'block';

        // Store file for processing
        this.selectedFile = file;
        this.addActivity('File Selected', `${file.name} ready for processing`);
    }

    handleFileFromMenu(filePath) {
        // Handle file selection from menu
        console.log('File selected from menu:', filePath);
        // This would need additional implementation for file reading from path
        this.showNotification('File selected from menu: ' + filePath, 'info');
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

        const sourceName = document.getElementById('sourceName').value || 'Unknown Source';
        
        this.showLoading('Processing data with AI...');
        this.addActivity('Processing', `Ingesting ${this.selectedFile.name}`);

        try {
            // Read file content
            const fileContent = await this.readFileContent(this.selectedFile);
            
            // Send to backend
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
                this.showDataPreview();
                this.updateSystemStats();
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

    async readFileContent(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    async showDataPreview() {
        try {
            const response = await fetch(`${this.serverUrl}/api/data/current`);
            const data = await response.json();

            if (data.shape) {
                const previewDiv = document.getElementById('dataPreview');
                const tableDiv = document.getElementById('dataTable');

                if (previewDiv && tableDiv) {
                    // Create table HTML
                    let tableHTML = '<div class="data-table"><table><thead><tr>';
                    data.columns.forEach(col => {
                        tableHTML += `<th>${col}</th>`;
                    });
                    tableHTML += '</tr></thead><tbody>';

                    // Add sample rows
                    data.sample.slice(0, 10).forEach(row => {
                        tableHTML += '<tr>';
                        data.columns.forEach(col => {
                            const value = row[col];
                            const displayValue = value !== null && value !== undefined ? 
                                (typeof value === 'number' && value % 1 !== 0 ? value.toFixed(2) : value) : 
                                '';
                            tableHTML += `<td>${displayValue}</td>`;
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

            const result = await response.json();

            if (result.success) {
                this.displayPatternResults(result.insights);
                this.addActivity('Analysis Complete', `Found ${result.insights.correlations?.length || 0} correlations`);
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

        if (insights.correlations && insights.correlations.length > 0) {
            html += '<h4>🔗 Strong Correlations Found:</h4><ul>';
            insights.correlations.forEach(corr => {
                const strength = Math.abs(corr.correlation);
                const strengthText = strength > 0.8 ? 'very strong' : strength > 0.6 ? 'strong' : 'moderate';
                html += `<li><strong>${corr.feature1}</strong> ↔ <strong>${corr.feature2}</strong>: ${corr.correlation.toFixed(3)} (${strengthText})</li>`;
            });
            html += '</ul>';
        } else {
            html += '<p>No strong correlations found in the data.</p>';
        }

        if (insights.clusters) {
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

        const contamination = parseFloat(document.getElementById('contaminationSlider').value);
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

            const result = await response.json();
            this.displayAnomalyResults(result);
            this.addActivity('Anomalies Detected', `Found ${result.anomaly_count} anomalies`);
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
        
        if (result.anomaly_count > 0) {
            html += `<p>Found <strong>${result.anomaly_count}</strong> anomalous data points out of ${this.currentData.shape[0]} total records.</p>`;
            html += `<p><strong>Anomaly Rate:</strong> ${((result.anomaly_count / this.currentData.shape[0]) * 100).toFixed(2)}%</p>`;
            
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

        // Add only numeric columns
        this.currentData.columns.forEach(column => {
            const dataType = this.currentData.dtypes[column];
            if (dataType && (dataType.includes('int') || dataType.includes('float'))) {
                const option = document.createElement('option');
                option.value = column;
                option.textContent = column;
                targetSelect.appendChild(option);
            }
        });

        if (targetSelect.children.length === 1) {
            targetSelect.innerHTML = '<option value="">No numeric columns found</option>';
        }
    }

    async buildModel() {
        const targetColumn = document.getElementById('targetColumn').value;
        
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

            const result = await response.json();

            if (result.success) {
                this.displayModelResults(result);
                this.models[targetColumn] = result;
                
                // Enable forecast button
                const forecastBtn = document.getElementById('forecastBtn');
                if (forecastBtn) {
                    forecastBtn.disabled = false;
                }
                
                this.updateSystemStats();
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
        html += `<p><strong>Target Variable:</strong> ${result.target_column}</p>`;
        html += `<p><strong>Input Features:</strong> ${result.features.join(', ')}</p>`;
        html += `<p><strong>Model Accuracy (MAE):</strong> ${result.mae.toFixed(4)}</p>`;
        
        // Feature importance
        html += '<h5>📊 Feature Importance (Top 5):</h5><ul>';
        const sortedFeatures = Object.entries(result.feature_importance)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);
            
        sortedFeatures.forEach(([feature, importance]) => {
            const percentage = (importance * 100).toFixed(1);
            html += `<li><strong>${feature}:</strong> ${percentage}%</li>`;
        });
        
        html += '</ul>';
        html += '<p class="model-info">💡 <em>Lower MAE indicates better model accuracy</em></p>';
        html += '</div>';
        
        resultsDiv.innerHTML = html;
    }

    async generateForecast() {
        const targetColumn = document.getElementById('targetColumn').value;
        const periods = parseInt(document.getElementById('forecastPeriods').value);

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

            const result = await response.json();

            if (result.success) {
                this.displayForecastResults(result);
                this.plotForecast(result.forecast, targetColumn);
                this.addActivity('Forecast Complete', `${periods}-day forecast generated`);
            } else {
                this.showNotification(`Forecast failed: ${result.error}`, 'error');
                this.addActivity('Forecast Failed', result.error);
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
        if (!resultsDiv) return;

        const avgValue = result.forecast.reduce((sum, item) => sum + item.predicted_value, 0) / result.forecast.length;
        const minValue = Math.min(...result.forecast.map(item => item.predicted_value));
        const maxValue = Math.max(...result.forecast.map(item => item.predicted_value));

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
        
        if (!plotDiv || !window.Plotly) {
            console.error('Plotly not available or plot div not found');
            return;
        }
        
        try {
            const dates = forecastData.map(item => item.date);
            const values = forecastData.map(item => item.predicted_value);
            const upperBound = forecastData.map(item => item.predicted_value + item.confidence_interval);
            const lowerBound = forecastData.map(item => item.predicted_value - item.confidence_interval);

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

            const result = await response.json();
            
            // Remove typing indicator
            this.removeChatMessage(typingId);
            
            // Add AI response
            this.addChatMessage(result.response, 'assistant');
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