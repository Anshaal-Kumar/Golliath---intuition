// Digital Twin Intelligence Platform - Frontend JavaScript
// Complete desktop application logic - ENHANCED VERSION

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

        // SetupEventListeners button listeners
        const causalBtn = document.getElementById('causalAnalysisBtn');
        if (causalBtn) {
            causalBtn.addEventListener('click', () => this.runCausalAnalysis());
        }

        // Chat functionality
        const chatSendBtn = document.getElementById('chatSendBtn');
        const chatInput = document.getElementById('chatInput');
        // Pivot table functionality
        const generatePivotBtn = document.getElementById('generatePivotBtn');
        if (generatePivotBtn) {
            generatePivotBtn.addEventListener('click', () => this.generatePivot());
        }

        const downloadPivotBtn = document.getElementById('downloadPivotCSV');
        if (downloadPivotBtn) {
            downloadPivotBtn.addEventListener('click', () => this.downloadPivotCSV());
        }

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
        } else if (pageId === 'analysis') {
            this.populateCausalColumns();
        } else if (pageId === 'pivot') {
            this.populatePivotSelectors();
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

        const isExcel = file.type === 'application/vnd.ms-excel' ||
            file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
            file.name.toLowerCase().endsWith('.xls') ||
            file.name.toLowerCase().endsWith('.xlsx');

        if (!isCSV && !isExcel) {
            this.showNotification('Please select a CSV or Excel file', 'warning');
            return;
        }

        console.log('File selected:', file.name, 'Type:', file.type);

        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const fileInfo = document.getElementById('fileInfo');
        const sourceName = document.getElementById('sourceName');

        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = `Size: ${this.formatFileSize(file.size)}`;
        if (sourceName) sourceName.value = file.name.replace(/\.[^/.]+$/, "");
        if (fileInfo) fileInfo.style.display = 'block';

        // Store file for processing
        this.selectedFile = file;
        this.selectedFileType = isExcel ? 'excel' : 'csv';
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

        this.showLoading(`Processing ${this.selectedFileType} file with AI...`);
        this.addActivity('Processing', `Ingesting ${this.selectedFile.name}`);

        try {
            if (this.selectedFileType === 'excel') {
                await this.processExcelFile(sourceName);
            } else {
                await this.processCSVFile(sourceName);
            }
        } catch (error) {
            console.error('Processing failed:', error);
            this.showNotification(`Processing failed: ${error.message}`, 'error');
            this.addActivity('Processing Failed', error.message);
        } finally {
            this.hideLoading();
        }
    }

    async processCSVFile(sourceName) {
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
            this.showNotification(`Successfully processed ${result.rows} rows!`, 'success');
            await this.showDataPreview();
            await this.updateSystemStats();
            this.addActivity('Data Processed', `${result.rows} rows, ${result.columns} columns`);
        } else {
            this.showNotification(`Processing failed: ${result.error}`, 'error');
            this.addActivity('Processing Failed', result.error);
        }
    }

    async processExcelFile(sourceName) {
        // Read file as base64
        const fileContent = await this.readFileAsBase64(this.selectedFile);

        // First, get sheet names
        const sheetsResponse = await fetch(`${this.serverUrl}/api/data/excel/sheets`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                excel_content: fileContent
            })
        });

        const sheetsResult = await sheetsResponse.json();

        // For now, use first sheet (you can add sheet selector UI later)
        const sheetName = sheetsResult.sheets && sheetsResult.sheets.length > 0 ? sheetsResult.sheets[0] : 0;

        console.log(`Processing Excel sheet: ${sheetName}`);

        // Process the Excel file
        const response = await fetch(`${this.serverUrl}/api/data/ingest/excel`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                excel_content: fileContent,
                source_name: sourceName,
                sheet_name: sheetName
            })
        });

        const result = await response.json();

        if (result.success) {
            this.showNotification(`Successfully processed ${result.rows} rows from Excel!`, 'success');
            await this.showDataPreview();
            await this.updateSystemStats();
            this.addActivity('Excel Processed', `${result.rows} rows, ${result.columns} columns`);
        } else {
            this.showNotification(`Processing failed: ${result.error}`, 'error');
            this.addActivity('Processing Failed', result.error);
        }
    }

    readFileAsBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                if (e.target && e.target.result) {
                    // Extract base64 content (remove data:...; prefix)
                    const base64 = e.target.result.split(',')[1];
                    resolve(base64);
                } else {
                    reject(new Error('Failed to read file content'));
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
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

    // FIXED DATA PREVIEW - Replace showDataPreview() method in renderer.js (around line 600)

async showDataPreview() {
    console.log('Attempting to show data preview...');
    
    try {
        const response = await fetch(`${this.serverUrl}/api/data/current`);
        
        console.log('Preview response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const responseText = await response.text();
        console.log('Preview response (first 200 chars):', responseText.substring(0, 200));
        
        let data;
        try {
            data = JSON.parse(responseText);
        } catch (e) {
            console.error('Failed to parse preview JSON:', e);
            throw new Error('Invalid JSON response from server');
        }

        console.log('Parsed preview data:', {
            shape: data.shape,
            columnsCount: data.columns?.length,
            sampleCount: data.sample?.length
        });

        // Validate data structure
        if (!data.shape || !data.columns || !data.sample) {
            console.error('Invalid data structure:', data);
            throw new Error('Invalid data structure received from server');
        }

        const previewDiv = document.getElementById('dataPreview');
        const tableDiv = document.getElementById('dataTable');

        if (!previewDiv || !tableDiv) {
            console.error('Preview div or table div not found');
            return;
        }

        // Build table HTML
        let tableHTML = '<div class="data-table"><table><thead><tr>';
        
        // Headers
        data.columns.forEach(col => {
            tableHTML += `<th>${this.escapeHtml(String(col))}</th>`;
        });
        tableHTML += '</tr></thead><tbody>';

        // Data rows (limit to 10)
        const sampleRows = data.sample.slice(0, 10);
        console.log(`Displaying ${sampleRows.length} sample rows`);

        if (sampleRows.length === 0) {
            tableHTML += '<tr><td colspan="' + data.columns.length + '" style="text-align: center; padding: 20px;">No sample data available</td></tr>';
        } else {
            sampleRows.forEach((row, rowIndex) => {
                tableHTML += '<tr>';
                data.columns.forEach(col => {
                    const value = row[col];
                    let displayValue = '';
                    
                    try {
                        if (value !== null && value !== undefined) {
                            if (typeof value === 'number') {
                                if (value % 1 !== 0) {
                                    displayValue = value.toFixed(2);
                                } else {
                                    displayValue = String(value);
                                }
                            } else {
                                displayValue = String(value);
                            }
                        } else {
                            displayValue = '-';
                        }
                    } catch (e) {
                        console.warn(`Error formatting value at row ${rowIndex}, col ${col}:`, e);
                        displayValue = '-';
                    }
                    
                    tableHTML += `<td>${this.escapeHtml(displayValue)}</td>`;
                });
                tableHTML += '</tr>';
            });
        }

        tableHTML += '</tbody></table></div>';

        // Add data info
        tableHTML += `<div style="margin-top: 16px; padding: 12px; background: rgba(59, 130, 246, 0.1); border-radius: 8px;">
            <p style="color: #60a5fa; margin: 0;"><strong>📊 Data Info:</strong> ${data.shape[0]} rows × ${data.shape[1]} columns</p>
        </div>`;

        tableDiv.innerHTML = tableHTML;
        previewDiv.style.display = 'block';
        
        // Store current data
        this.currentData = data;
        
        console.log('✅ Data preview displayed successfully');

        // Populate selectors for other features
        try {
            this.populateCausalColumns(data.columns, data.dtypes);
            console.log('Populated causal columns');
        } catch (e) {
            console.warn('Failed to populate causal columns:', e);
        }

        // Show success notification
        this.showNotification('Data preview loaded successfully', 'success');
        
    } catch (error) {
        console.error('❌ Failed to show preview:', error);
        
        const previewDiv = document.getElementById('dataPreview');
        const tableDiv = document.getElementById('dataTable');
        
        if (tableDiv) {
            tableDiv.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #ef4444;">
                    <p><strong>⚠️ Failed to load data preview</strong></p>
                    <p style="font-size: 14px; color: #94a3b8;">${this.escapeHtml(error.message)}</p>
                    <p style="font-size: 12px; color: #64748b; margin-top: 8px;">Check the console for more details</p>
                </div>
            `;
        }
        
        if (previewDiv) {
            previewDiv.style.display = 'block';
        }
        
        this.showNotification(`Failed to load preview: ${error.message}`, 'error');
    }
}


    // FIXED PIVOT TABLE METHODS - Replace in renderer.js starting around line 730

async generatePivot() {
    const rowsSelect = document.getElementById('pivotRows');
    const columnsSelect = document.getElementById('pivotColumns');
    const valuesSelect = document.getElementById('pivotValues');
    const aggFuncSelect = document.getElementById('pivotAggFunc');
    
    // Check if elements exist
    if (!rowsSelect || !columnsSelect || !valuesSelect || !aggFuncSelect) {
        console.error('Pivot form elements not found');
        this.showNotification('Pivot table form not properly loaded', 'error');
        return;
    }
    
    // Get selected rows (multi-select)
    const rows = Array.from(rowsSelect.selectedOptions).map(opt => opt.value);
    const columns = columnsSelect.value;
    const values = valuesSelect.value;
    const aggFunc = aggFuncSelect.value;
    
    console.log('Pivot request:', { rows, columns, values, aggFunc });
    
    // VALIDATION
    if (rows.length === 0) {
        this.showNotification('❌ Please select at least one row field (use Ctrl+Click for multiple)', 'warning');
        rowsSelect.focus();
        return;
    }
    
    if (!values || values === '') {
        this.showNotification('❌ Please select a numeric value field to aggregate', 'warning');
        valuesSelect.focus();
        return;
    }
    
    if (!aggFunc || aggFunc === '') {
        this.showNotification('❌ Please select an aggregation function', 'warning');
        aggFuncSelect.focus();
        return;
    }
    
    this.showLoading('Generating pivot table...');
    this.addActivity('Pivot Analysis', `Grouping by ${rows.join(', ')}, aggregating ${values}`);
    
    try {
        const requestBody = {
            rows: rows,
            columns: (columns && columns !== 'None') ? columns : null,
            values: values,
            agg_func: aggFunc
        };
        
        console.log('Sending pivot request:', requestBody);
        
        const response = await fetch(`${this.serverUrl}/api/data/pivot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        const responseText = await response.text();
        console.log('Pivot response status:', response.status);
        console.log('Pivot response text:', responseText.substring(0, 500));
        
        let result;
        try {
            result = JSON.parse(responseText);
        } catch (e) {
            console.error('JSON parse error:', e);
            throw new Error(`Server returned invalid JSON. Status: ${response.status}. Response: ${responseText.substring(0, 200)}`);
        }
        
        if (!response.ok) {
            const errorMsg = result.error || `Server error (HTTP ${response.status})`;
            throw new Error(errorMsg);
        }
        
        if (result.success) {
            console.log('Pivot generated successfully:', result);
            this.displayPivotTable(result);
            this.showNotification(`✅ Pivot table generated! ${result.row_count} rows created`, 'success');
            this.addActivity('Pivot Complete', `${result.row_count} rows generated`);
        } else {
            throw new Error(result.error || 'Pivot generation failed');
        }
        
    } catch (error) {
        console.error('Pivot generation failed:', error);
        this.showNotification(`❌ Failed to generate pivot: ${error.message}`, 'error');
        this.addActivity('Pivot Failed', error.message);
    } finally {
        this.hideLoading();
    }
}

displayPivotTable(result) {
    const resultsDiv = document.getElementById('pivotResults');
    const tableDiv = document.getElementById('pivotTable');
    
    if (!resultsDiv || !tableDiv) return;
    
    // Validate data
    if (!result.pivot_data || result.pivot_data.length === 0) {
        tableDiv.innerHTML = '<p class="loading-message">No data to display</p>';
        resultsDiv.style.display = 'block';
        return;
    }
    
    if (!result.columns || result.columns.length === 0) {
        tableDiv.innerHTML = '<p class="loading-message">No columns to display</p>';
        resultsDiv.style.display = 'block';
        return;
    }
    
    let html = '<div class="data-table"><table><thead><tr>';
    
    // Headers - filter out empty columns
    const validColumns = result.columns.filter(col => col && col.trim() !== '');
    
    validColumns.forEach(col => {
        html += `<th>${this.escapeHtml(col)}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Data rows - limit display and filter empty values
    const maxDisplayRows = 100;
    const displayData = result.pivot_data.slice(0, maxDisplayRows);
    
    displayData.forEach(row => {
        html += '<tr>';
        validColumns.forEach(col => {
            const value = row[col];
            let displayValue = '';
            
            if (value !== null && value !== undefined && value !== '') {
                if (typeof value === 'number') {
                    // Format numbers nicely
                    if (Number.isInteger(value)) {
                        displayValue = value.toLocaleString();
                    } else {
                        displayValue = value.toFixed(2);
                    }
                } else {
                    const strVal = String(value).trim();
                    if (strVal !== 'nan' && strVal !== 'None' && strVal !== '') {
                        displayValue = strVal;
                    } else {
                        displayValue = '-';
                    }
                }
            } else {
                displayValue = '-';
            }
            
            html += `<td>${this.escapeHtml(displayValue)}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table></div>';
    
    // Summary info with better labels
    const aggFuncLabels = {
        'sum': 'Sum',
        'mean': 'Average',
        'count': 'Count',
        'min': 'Minimum',
        'max': 'Maximum',
        'median': 'Median',
        'std': 'Standard Deviation'
    };
    
    const aggLabel = aggFuncLabels[result.summary.aggregation] || result.summary.aggregation;
    
    html += `<div class="pivot-summary">
        <p><strong>📊 Pivot Table Summary:</strong></p>
        <ul>
            <li>Grouped by: <strong>${result.summary.rows.join(', ')}</strong></li>
            ${result.summary.columns ? `<li>Columns: <strong>${result.summary.columns}</strong></li>` : ''}
            <li>Values: <strong>${result.summary.values}</strong></li>
            <li>Aggregation: <strong>${aggLabel}</strong></li>
            <li>Rows shown: <strong>${displayData.length.toLocaleString()}</strong> ${result.row_count > maxDisplayRows ? `(of ${result.row_count.toLocaleString()} total)` : ''}</li>
        </ul>
    </div>`;
    
    if (result.row_count > maxDisplayRows) {
        html += `<div class="pivot-notice" style="padding: 12px; background: rgba(245, 158, 11, 0.1); border-radius: 8px; margin-top: 12px; border-left: 4px solid #f59e0b;">
            <p style="color: #f59e0b; margin: 0;"><strong>ℹ️ Note:</strong> Displaying first ${maxDisplayRows} rows. Download CSV for complete data.</p>
        </div>`;
    }
    
    tableDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
    
    // Store pivot data for download
    window.currentPivotData = result;
}

populatePivotSelectors() {
    if (!this.currentData) {
        console.log('No current data available for pivot selectors');
        return;
    }
    
    const rowsSelect = document.getElementById('pivotRows');
    const columnsSelect = document.getElementById('pivotColumns');
    const valuesSelect = document.getElementById('pivotValues');
    
    if (!rowsSelect || !columnsSelect || !valuesSelect) {
        console.error('Pivot selector elements not found');
        return;
    }
    
    console.log('Populating pivot selectors with columns:', this.currentData.columns);
    
    // Clear existing options
    rowsSelect.innerHTML = '';
    columnsSelect.innerHTML = '<option value="None">None</option>';
    valuesSelect.innerHTML = '<option value="">-- Select a numeric column --</option>';
    
    let numericCount = 0;
    
    // Populate with all columns
    this.currentData.columns.forEach(col => {
        // Rows selector (all columns can be grouped)
        const rowOption = document.createElement('option');
        rowOption.value = col;
        rowOption.textContent = col;
        rowsSelect.appendChild(rowOption);
        
        // Columns selector (all columns can be pivoted)
        const colOption = document.createElement('option');
        colOption.value = col;
        colOption.textContent = col;
        columnsSelect.appendChild(colOption);
        
        // Values selector (only numeric columns can be aggregated)
        const dtype = this.currentData.dtypes[col];
        if (dtype && (dtype.includes('int') || dtype.includes('float'))) {
            const valOption = document.createElement('option');
            valOption.value = col;
            valOption.textContent = `${col} (${dtype})`;
            valuesSelect.appendChild(valOption);
            numericCount++;
        }
    });
    
    console.log(`Pivot selectors populated: ${this.currentData.columns.length} total columns, ${numericCount} numeric columns`);
    
    // Show warning if no numeric columns
    if (numericCount === 0) {
        valuesSelect.innerHTML = '<option value="">⚠️ No numeric columns found</option>';
        this.showNotification('⚠️ Your data has no numeric columns. Pivot tables require at least one numeric column to aggregate.', 'warning');
    }
}

downloadPivotCSV() {
    if (!window.currentPivotData) {
        this.showNotification('No pivot data to download', 'warning');
        return;
    }
    
    const data = window.currentPivotData.pivot_data;
    const columns = window.currentPivotData.columns;
    
    if (!data || data.length === 0) {
        this.showNotification('No data to download', 'warning');
        return;
    }
    
    try {
        // Convert to CSV with proper escaping
        let csv = columns.join(',') + '\n';
        
        data.forEach(row => {
            const values = columns.map(col => {
                const val = row[col];
                if (val === null || val === undefined) return '';
                
                const strVal = String(val);
                // Escape values that contain commas or quotes
                if (strVal.includes(',') || strVal.includes('"') || strVal.includes('\n')) {
                    return `"${strVal.replace(/"/g, '""')}"`;
                }
                return strVal;
            });
            csv += values.join(',') + '\n';
        });
        
        // Create download
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Generate filename with timestamp
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        a.download = `pivot_table_${timestamp}.csv`;
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        this.showNotification('Pivot table downloaded successfully!', 'success');
        this.addActivity('Export', 'Pivot table downloaded as CSV');
    } catch (error) {
        console.error('Download failed:', error);
        this.showNotification(`Download failed: ${error.message}`, 'error');
    }
}

    // Enhanced pattern analysis display
    displayPatternResults(insights) {
        const resultsDiv = document.getElementById('patternResults');
        if (!resultsDiv) return;

        let html = '<div class="analysis-results">';

        // Data quality overview
        if (insights.data_quality) {
            const quality = insights.data_quality;
            const qualityScore = 100 - (quality.missing_values / quality.total_rows * 100);
            html += `
                <div class="quality-summary">
                    <h4>📊 Data Quality Overview</h4>
                    <div class="quality-metrics">
                        <span class="metric">Quality Score: <strong>${qualityScore.toFixed(1)}%</strong></span>
                        <span class="metric">Rows: <strong>${quality.total_rows.toLocaleString()}</strong></span>
                        <span class="metric">Columns: <strong>${quality.total_columns}</strong></span>
                        <span class="metric">Missing: <strong>${quality.missing_values.toLocaleString()}</strong></span>
                    </div>
                </div>
            `;
        }

        // Strong correlations
        if (insights.correlations && insights.correlations.length > 0) {
            html += '<h4>🔗 Strong Correlations Found (≥0.5):</h4>';
            html += '<div class="correlation-grid">';

            insights.correlations.slice(0, 10).forEach(corr => {
                const strength = Math.abs(corr.correlation);
                const strengthText = strength > 0.8 ? 'very strong' :
                    strength > 0.6 ? 'strong' : 'moderate';
                const strengthClass = strength > 0.8 ? 'very-strong' :
                    strength > 0.6 ? 'strong' : 'moderate';
                const direction = corr.correlation > 0 ? '↗️' : '↘️';

                html += `
                    <div class="correlation-item ${strengthClass}">
                        <div class="correlation-pair">
                            <strong>${this.escapeHtml(corr.feature1)}</strong> 
                            <span class="correlation-arrow">${direction}</span> 
                            <strong>${this.escapeHtml(corr.feature2)}</strong>
                        </div>
                        <div class="correlation-strength">
                            <span class="correlation-value">r = ${corr.correlation.toFixed(3)}</span>
                            <span class="correlation-label">${strengthText}</span>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
        }

        // Moderate correlations
        if (insights.moderate_correlations && insights.moderate_correlations.length > 0) {
            html += '<h4>📈 Moderate Correlations (0.3-0.5):</h4>';
            html += '<div class="moderate-correlations">';

            insights.moderate_correlations.slice(0, 5).forEach(corr => {
                const direction = corr.correlation > 0 ? 'positive' : 'negative';
                html += `
                    <div class="moderate-corr-item">
                        <span class="corr-features">${this.escapeHtml(corr.feature1)} ↔ ${this.escapeHtml(corr.feature2)}</span>
                        <span class="corr-value ${direction}">${corr.correlation.toFixed(3)}</span>
                    </div>
                `;
            });
            html += '</div>';
        }

        // No correlations found
        if ((!insights.correlations || insights.correlations.length === 0) &&
            (!insights.moderate_correlations || insights.moderate_correlations.length === 0)) {

            if (insights.correlation_summary && insights.correlation_summary.total_pairs > 0) {
                const summary = insights.correlation_summary;
                html += `
                    <h4>📊 Correlation Analysis Results</h4>
                    <div class="correlation-summary">
                        <p><strong>Analysis Summary:</strong></p>
                        <ul>
                            <li>Analyzed <strong>${summary.total_pairs}</strong> feature pairs</li>
                            <li>Maximum correlation found: <strong>${summary.max_correlation.toFixed(3)}</strong></li>
                            <li>Average correlation: <strong>${summary.avg_correlation.toFixed(3)}</strong></li>
                        </ul>
                        <p class="correlation-insight">
                            💡 <em>Your features show relatively low correlations, which suggests they capture different aspects of your data. This is often good for machine learning models!</em>
                        </p>
                    </div>
                `;
            } else {
                html += `
                    <h4>🔍 No Strong Correlations Found</h4>
                    <div class="no-correlations">
                        <p>This could mean:</p>
                        <ul>
                            <li>Your features are independent (good for modeling)</li>
                            <li>You may have mostly categorical data</li>
                            <li>The relationships might be non-linear</li>
                        </ul>
                        <p class="tip">💡 Try building a predictive model to discover hidden patterns!</p>
                    </div>
                `;
            }
        }

        // Clustering results
        if (insights.clusters) {
            const clusters = insights.clusters;
            html += `
                <h4>🎯 Data Segmentation Results</h4>
                <div class="cluster-results">
                    <p>Discovered <strong>${clusters.n_clusters}</strong> distinct patterns in your data.</p>
                    <div class="cluster-info">
                        <span>Cluster sizes: <strong>[${clusters.cluster_sizes.join(', ')}]</strong></span>
                        <span>Features used: <strong>${clusters.features_used.length}</strong></span>
                    </div>
                    <p class="cluster-insight">
                        💡 <em>These clusters represent different groups in your data that behave similarly.</em>
                    </p>
                </div>
            `;
        }

        // Feature variability insights
        if (insights.feature_variability && Object.keys(insights.feature_variability).length > 0) {
            const topVariable = Object.entries(insights.feature_variability)
                .sort(([, a], [, b]) => b.coefficient_of_variation - a.coefficient_of_variation)[0];

            if (topVariable) {
                html += `
                    <h4>📏 Feature Variability</h4>
                    <div class="variability-insight">
                        <p>Most variable feature: <strong>${this.escapeHtml(topVariable[0])}</strong> 
                        (CV: ${topVariable[1].coefficient_of_variation.toFixed(2)})</p>
                        <p class="tip">💡 High variability features often contain important information for predictions.</p>
                    </div>
                `;
            }
        }

        html += '</div>';
        resultsDiv.innerHTML = html;

        // Add enhanced styles
        this.addAnalysisStyles();
    }

    // Add this NEW METHOD after displayPatternResults() ends

    async visualizePatterns(insights) {
        if (!this.currentData) return;

        try {
            // Fetch current data for visualization
            const response = await fetch(`${this.serverUrl}/api/data/current`);
            if (!response.ok) return;

            const dataInfo = await response.json();

            // Create visualization container
            const analysisPage = document.getElementById('analysis');
            let vizContainer = document.getElementById('patternVizContainer');

            if (!vizContainer) {
                vizContainer = document.createElement('div');
                vizContainer.id = 'patternVizContainer';
                vizContainer.className = 'content-card';
                vizContainer.innerHTML = '<h3>📊 Visual Analysis</h3><div id="patternCharts"></div>';
                analysisPage.appendChild(vizContainer);
            }

            const chartsDiv = document.getElementById('patternCharts');
            if (!chartsDiv) return;

            // Clear previous charts
            chartsDiv.innerHTML = '';

            // 1. CORRELATION HEATMAP
            if (insights.correlations || insights.moderate_correlations) {
                await this.createCorrelationHeatmap(chartsDiv, dataInfo);
            }

            // 2. CLUSTER VISUALIZATION
            if (insights.clusters) {
                await this.createClusterVisualization(chartsDiv, dataInfo, insights.clusters);
            }

            // 3. FEATURE DISTRIBUTION HISTOGRAMS
            await this.createFeatureHistograms(chartsDiv, dataInfo);

        } catch (error) {
            console.error('Visualization failed:', error);
        }
    }

    async createCorrelationHeatmap(container, dataInfo) {
        const div = document.createElement('div');
        div.id = 'correlationHeatmap';
        div.style.height = '500px';
        div.style.marginBottom = '20px';
        container.appendChild(div);

        try {
            // Get numeric columns
            const numericCols = Object.keys(dataInfo.dtypes || {}).filter(col => {
                const dtype = dataInfo.dtypes[col];
                return dtype.includes('int') || dtype.includes('float');
            });

            if (numericCols.length < 2) {
                div.innerHTML = '<p class="loading-message">Need at least 2 numeric columns for correlation heatmap</p>';
                return;
            }

            // Calculate correlation matrix from sample data
            const correlationMatrix = this.calculateCorrelationMatrix(dataInfo.sample, numericCols);

            const data = [{
                z: correlationMatrix.values,
                x: correlationMatrix.labels,
                y: correlationMatrix.labels,
                type: 'heatmap',
                colorscale: [
                    [0, '#ef4444'],      // Strong negative = Red
                    [0.5, '#f3f4f6'],    // Zero = Light gray
                    [1, '#22c55e']       // Strong positive = Green
                ],
                zmid: 0,
                colorbar: {
                    title: 'Correlation',
                    titleside: 'right',
                    tickmode: 'linear',
                    tick0: -1,
                    dtick: 0.5
                },
                hovertemplate: '<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            }];

            const layout = {
                title: {
                    text: '🔥 Correlation Heatmap',
                    font: { size: 18, color: '#e2e8f0' }
                },
                xaxis: {
                    tickangle: -45,
                    color: '#e2e8f0',
                    tickfont: { size: 10 }
                },
                yaxis: {
                    color: '#e2e8f0',
                    tickfont: { size: 10 }
                },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#e2e8f0' },
                margin: { t: 80, l: 120, r: 80, b: 120 }
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };

            Plotly.newPlot(div, data, layout, config);

        } catch (error) {
            console.error('Heatmap creation failed:', error);
            div.innerHTML = '<p class="loading-message">Failed to create correlation heatmap</p>';
        }
    }

    calculateCorrelationMatrix(sampleData, numericCols) {
        const n = numericCols.length;
        const matrix = Array(n).fill(0).map(() => Array(n).fill(0));

        // Calculate correlations between all pairs
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i === j) {
                    matrix[i][j] = 1.0; // Perfect correlation with itself
                } else {
                    // Extract column values
                    const col1 = sampleData.map(row => parseFloat(row[numericCols[i]]) || 0);
                    const col2 = sampleData.map(row => parseFloat(row[numericCols[j]]) || 0);

                    // Calculate Pearson correlation
                    matrix[i][j] = this.pearsonCorrelation(col1, col2);
                }
            }
        }

        return {
            values: matrix,
            labels: numericCols
        };
    }

    pearsonCorrelation(x, y) {
        const n = x.length;
        if (n === 0) return 0;

        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        return denominator === 0 ? 0 : numerator / denominator;
    }

    async createClusterVisualization(container, dataInfo, clusterInfo) {
        const div = document.createElement('div');
        div.id = 'clusterViz';
        div.style.height = '500px';
        div.style.marginBottom = '20px';
        container.appendChild(div);

        try {
            const features = clusterInfo.features_used || [];
            if (features.length < 2) {
                div.innerHTML = '<p class="loading-message">Need at least 2 features for cluster visualization</p>';
                return;
            }

            // Use first two features for 2D visualization
            const feature1 = features[0];
            const feature2 = features[1];
            const labels = clusterInfo.labels || [];

            // Extract feature values from sample data
            const x = dataInfo.sample.map(row => parseFloat(row[feature1]) || 0);
            const y = dataInfo.sample.map(row => parseFloat(row[feature2]) || 0);
            const colors = labels.slice(0, x.length);

            const data = [{
                x: x,
                y: y,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: 10,
                    color: colors,
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {
                        title: 'Cluster',
                        titleside: 'right'
                    },
                    line: {
                        color: '#1e293b',
                        width: 1
                    }
                },
                text: colors.map(c => `Cluster ${c}`),
                hovertemplate: '<b>Cluster %{text}</b><br>%{xaxis.title.text}: %{x:.2f}<br>%{yaxis.title.text}: %{y:.2f}<extra></extra>'
            }];

            const layout = {
                title: {
                    text: `🎯 Cluster Visualization (${clusterInfo.n_clusters} clusters)`,
                    font: { size: 18, color: '#e2e8f0' }
                },
                xaxis: {
                    title: feature1,
                    color: '#e2e8f0',
                    gridcolor: '#374151'
                },
                yaxis: {
                    title: feature2,
                    color: '#e2e8f0',
                    gridcolor: '#374151'
                },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#e2e8f0' },
                hovermode: 'closest'
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };

            Plotly.newPlot(div, data, layout, config);

        } catch (error) {
            console.error('Cluster visualization failed:', error);
            div.innerHTML = '<p class="loading-message">Failed to create cluster visualization</p>';
        }
    }

    async createFeatureHistograms(container, dataInfo) {
        const div = document.createElement('div');
        div.id = 'featureHistograms';
        div.style.height = '400px';
        div.style.marginBottom = '20px';
        container.appendChild(div);

        try {
            // Get numeric columns
            const numericCols = Object.keys(dataInfo.dtypes || {}).filter(col => {
                const dtype = dataInfo.dtypes[col];
                return dtype.includes('int') || dtype.includes('float');
            }).slice(0, 4); // Limit to 4 features

            if (numericCols.length === 0) {
                div.innerHTML = '<p class="loading-message">No numeric columns for histograms</p>';
                return;
            }

            const traces = numericCols.map(col => {
                const values = dataInfo.sample
                    .map(row => parseFloat(row[col]))
                    .filter(v => !isNaN(v));

                return {
                    x: values,
                    type: 'histogram',
                    name: col,
                    opacity: 0.7,
                    marker: {
                        line: {
                            color: '#1e293b',
                            width: 1
                        }
                    }
                };
            });

            const layout = {
                title: {
                    text: '📊 Feature Distributions',
                    font: { size: 18, color: '#e2e8f0' }
                },
                barmode: 'overlay',
                xaxis: {
                    title: 'Value',
                    color: '#e2e8f0',
                    gridcolor: '#374151'
                },
                yaxis: {
                    title: 'Frequency',
                    color: '#e2e8f0',
                    gridcolor: '#374151'
                },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#e2e8f0' },
                legend: {
                    bgcolor: 'rgba(30, 41, 59, 0.8)',
                    bordercolor: '#475569',
                    borderwidth: 1
                }
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };

            Plotly.newPlot(div, traces, layout, config);

        } catch (error) {
            console.error('Histogram creation failed:', error);
            div.innerHTML = '<p class="loading-message">Failed to create histograms</p>';
        }
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
            await this.visualizeAnomalies(result);
            this.addActivity('Anomalies Detected', `Found ${result.anomaly_count || 0} anomalies`);
        } catch (error) {
            console.error('Anomaly detection failed:', error);
            this.showNotification(`Anomaly detection failed: ${error.message}`, 'error');
            this.addActivity('Analysis Failed', error.message);
        } finally {
            this.hideLoading();
        }
    }

    async runCausalAnalysis() {
        const treatmentCol = document.getElementById('treatmentColumn').value;
        const outcomeCol = document.getElementById('outcomeColumn').value;

        if (!treatmentCol || !outcomeCol) {
            this.showNotification('Please select both treatment and outcome variables', 'warning');
            return;
        }

        this.showLoading('Running causal analysis...');

        try {
            const response = await fetch(`${this.serverUrl}/api/analysis/causality`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    treatment_column: treatmentCol,
                    outcome_column: outcomeCol
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.success) {
                this.displayCausalResults(result.causal_analysis);
                this.showNotification('Causal analysis complete!', 'success');
            } else {
                this.showNotification(`Analysis failed: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Causal analysis failed:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayCausalResults(analysis) {
        const resultsDiv = document.getElementById('causalResults');
        if (!resultsDiv) return;

        let html = '<div class="causal-results">';
        html += '<h4>🔗 Causal Effect Results</h4>';
        html += `<p><strong>Treatment:</strong> ${this.escapeHtml(analysis.treatment)}</p>`;
        html += `<p><strong>Outcome:</strong> ${this.escapeHtml(analysis.outcome)}</p>`;
        html += `<p><strong>Causal Effect:</strong> ${analysis.causal_effect.toFixed(4)}</p>`;
        html += `<p class="interpretation">${this.escapeHtml(analysis.interpretation)}</p>`;

        if (analysis.confidence_interval) {
            html += `<p><strong>95% CI:</strong> [${analysis.confidence_interval[0].toFixed(4)}, ${analysis.confidence_interval[1].toFixed(4)}]</p>`;
        }

        if (analysis.confounders && analysis.confounders.length > 0) {
            html += `<p><strong>Controlled for:</strong> ${analysis.confounders.join(', ')}</p>`;
        }

        html += '</div>';
        resultsDiv.innerHTML = html;
    }

    // Enhanced anomaly results display
    displayAnomalyResults(result) {
        const resultsDiv = document.getElementById('anomalyResults');
        if (!resultsDiv) return;

        let html = '<div class="analysis-results anomaly-results">';

        const anomalyCount = result.anomaly_count || 0;
        const totalRecords = result.total_records || 0;
        const anomalyRate = result.anomaly_rate || 0;

        html += '<h4>🚨 Anomaly Detection Results</h4>';

        if (anomalyCount > 0) {
            // Anomaly summary with visual indicators
            html += `
                <div class="anomaly-summary">
                    <div class="anomaly-stat-grid">
                        <div class="anomaly-stat">
                            <div class="stat-number">${anomalyCount.toLocaleString()}</div>
                            <div class="stat-label">Anomalies Found</div>
                        </div>
                        <div class="anomaly-stat">
                            <div class="stat-number">${anomalyRate.toFixed(2)}%</div>
                            <div class="stat-label">Anomaly Rate</div>
                        </div>
                        <div class="anomaly-stat">
                            <div class="stat-number">${totalRecords.toLocaleString()}</div>
                            <div class="stat-label">Total Records</div>
                        </div>
                    </div>
                </div>
            `;

            // Severity assessment
            let severity, severityClass, recommendation;
            if (anomalyRate > 10) {
                severity = 'High';
                severityClass = 'severity-high';
                recommendation = 'Consider reviewing data quality or adjusting sensitivity.';
            } else if (anomalyRate > 5) {
                severity = 'Moderate';
                severityClass = 'severity-moderate';
                recommendation = 'Normal level - these anomalies may represent interesting edge cases.';
            } else {
                severity = 'Low';
                severityClass = 'severity-low';
                recommendation = 'Good data quality - few outliers detected.';
            }

            html += `
                <div class="severity-indicator ${severityClass}">
                    <div class="severity-badge">${severity} Anomaly Level</div>
                    <p class="severity-text">${recommendation}</p>
                </div>
            `;

            // Sample anomaly indices
            if (result.anomaly_indices && result.anomaly_indices.length > 0) {
                const displayCount = Math.min(10, result.anomaly_indices.length);
                const sampleIndices = result.anomaly_indices.slice(0, displayCount);

                html += `
                    <div class="anomaly-indices">
                        <h5>🎯 Sample Anomaly Record IDs:</h5>
                        <div class="indices-list">
                            ${sampleIndices.map(idx => `<span class="anomaly-index">#${idx}</span>`).join('')}
                        </div>
                        ${result.anomaly_indices.length > displayCount ?
                        `<p class="more-indices">... and ${result.anomaly_indices.length - displayCount} more</p>` : ''}
                    </div>
                `;
            }

            // Anomaly scores if available
            if (result.anomaly_scores && result.threshold_score) {
                const avgScore = result.anomaly_scores.reduce((a, b) => a + b, 0) / result.anomaly_scores.length;
                html += `
                    <div class="anomaly-scores">
                        <h5>📊 Anomaly Scoring:</h5>
                        <div class="score-metrics">
                            <span>Threshold: <strong>${result.threshold_score.toFixed(3)}</strong></span>
                            <span>Average Score: <strong>${avgScore.toFixed(3)}</strong></span>
                        </div>
                        <p class="score-explanation">Lower scores indicate more anomalous records.</p>
                    </div>
                `;
            }

        } else {
            // No anomalies found
            html += `
                <div class="no-anomalies">
                    <div class="no-anomaly-icon">✅</div>
                    <h5>No Anomalies Detected</h5>
                    <p>With the current sensitivity setting (<strong>${(result.contamination * 100).toFixed(1)}%</strong>), no anomalous records were found.</p>
                    <div class="suggestions">
                        <p><strong>Try:</strong></p>
                        <ul>
                            <li>Lowering the sensitivity slider to detect subtler anomalies</li>
                            <li>This could indicate good data quality!</li>
                        </ul>
                    </div>
                </div>
            `;
        }

        html += '</div>';
        resultsDiv.innerHTML = html;

        // Add enhanced styles
        this.addAnalysisStyles();
    }

    // Add this NEW METHOD after displayAnomalyResults() ends

    async visualizeAnomalies(result) {
        if (!this.currentData || !result.anomaly_indices || result.anomaly_indices.length === 0) {
            return;
        }

        try {
            // Fetch current data
            const response = await fetch(`${this.serverUrl}/api/data/current`);
            if (!response.ok) return;

            const dataInfo = await response.json();

            // Create visualization container
            const analysisPage = document.getElementById('analysis');
            let vizContainer = document.getElementById('anomalyVizContainer');

            if (!vizContainer) {
                vizContainer = document.createElement('div');
                vizContainer.id = 'anomalyVizContainer';
                vizContainer.className = 'content-card';
                vizContainer.innerHTML = '<h3>🚨 Anomaly Visualizations</h3><div id="anomalyCharts"></div>';
                analysisPage.appendChild(vizContainer);
            }

            const chartsDiv = document.getElementById('anomalyCharts');
            if (!chartsDiv) return;

            chartsDiv.innerHTML = '';

            // Create scatter plot for anomalies
            await this.createAnomalyScatterPlot(chartsDiv, dataInfo, result);

            // Create bar chart showing anomaly scores
            await this.createAnomalyBarChart(chartsDiv, result);

        } catch (error) {
            console.error('Anomaly visualization failed:', error);
        }
    }

    async createAnomalyScatterPlot(container, dataInfo, anomalyResult) {
        const div = document.createElement('div');
        div.id = 'anomalyScatter';
        div.style.height = '500px';
        div.style.marginBottom = '20px';
        container.appendChild(div);

        try {
            // Get first two numeric columns
            const numericCols = Object.keys(dataInfo.dtypes || {}).filter(col => {
                const dtype = dataInfo.dtypes[col];
                return dtype.includes('int') || dtype.includes('float');
            }).slice(0, 2);

            if (numericCols.length < 2) {
                div.innerHTML = '<p class="loading-message">Need at least 2 numeric columns for scatter plot</p>';
                return;
            }

            const feature1 = numericCols[0];
            const feature2 = numericCols[1];

            // Create set of anomaly indices for quick lookup
            const anomalySet = new Set(anomalyResult.anomaly_indices);

            // Separate normal and anomaly points
            const normalX = [], normalY = [];
            const anomalyX = [], anomalyY = [];

            dataInfo.sample.forEach((row, index) => {
                const x = parseFloat(row[feature1]) || 0;
                const y = parseFloat(row[feature2]) || 0;

                if (anomalySet.has(index)) {
                    anomalyX.push(x);
                    anomalyY.push(y);
                } else {
                    normalX.push(x);
                    normalY.push(y);
                }
            });

            const traces = [
                {
                    x: normalX,
                    y: normalY,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Normal',
                    marker: {
                        size: 8,
                        color: '#3b82f6',
                        opacity: 0.6,
                        line: {
                            color: '#1e293b',
                            width: 1
                        }
                    },
                    hovertemplate: '<b>Normal Point</b><br>%{xaxis.title.text}: %{x:.2f}<br>%{yaxis.title.text}: %{y:.2f}<extra></extra>'
                },
                {
                    x: anomalyX,
                    y: anomalyY,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Anomaly',
                    marker: {
                        size: 12,
                        color: '#ef4444',
                        symbol: 'x',
                        line: {
                            color: '#991b1b',
                            width: 2
                        }
                    },
                    hovertemplate: '<b>⚠️ Anomaly</b><br>%{xaxis.title.text}: %{x:.2f}<br>%{yaxis.title.text}: %{y:.2f}<extra></extra>'
                }
            ];

            const layout = {
                title: {
                    text: '🚨 Anomaly Detection Scatter Plot',
                    font: { size: 18, color: '#e2e8f0' }
                },
                xaxis: {
                    title: feature1,
                    color: '#e2e8f0',
                    gridcolor: '#374151'
                },
                yaxis: {
                    title: feature2,
                    color: '#e2e8f0',
                    gridcolor: '#374151'
                },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#e2e8f0' },
                hovermode: 'closest',
                legend: {
                    bgcolor: 'rgba(30, 41, 59, 0.8)',
                    bordercolor: '#475569',
                    borderwidth: 1
                }
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };

            Plotly.newPlot(div, traces, layout, config);

        } catch (error) {
            console.error('Anomaly scatter plot failed:', error);
            div.innerHTML = '<p class="loading-message">Failed to create anomaly scatter plot</p>';
        }
    }

    async createAnomalyBarChart(container, anomalyResult) {
        const div = document.createElement('div');
        div.id = 'anomalyBarChart';
        div.style.height = '400px';
        div.style.marginBottom = '20px';
        container.appendChild(div);

        try {
            if (!anomalyResult.anomaly_scores || anomalyResult.anomaly_scores.length === 0) {
                div.innerHTML = '<p class="loading-message">No anomaly scores available for bar chart</p>';
                return;
            }

            // Get top 20 most anomalous records
            const scores = anomalyResult.anomaly_scores.map((score, idx) => ({
                index: idx,
                score: score,
                isAnomaly: anomalyResult.anomaly_indices.includes(idx)
            })).sort((a, b) => a.score - b.score).slice(0, 20);

            const data = [{
                x: scores.map(s => `Record ${s.index}`),
                y: scores.map(s => s.score),
                type: 'bar',
                marker: {
                    color: scores.map(s => s.isAnomaly ? '#ef4444' : '#3b82f6'),
                    line: {
                        color: '#1e293b',
                        width: 1
                    }
                },
                hovertemplate: '<b>%{x}</b><br>Anomaly Score: %{y:.3f}<extra></extra>'
            }];

            const layout = {
                title: {
                    text: '📊 Top 20 Anomaly Scores (Lower = More Anomalous)',
                    font: { size: 18, color: '#e2e8f0' }
                },
                xaxis: {
                    title: 'Record ID',
                    color: '#e2e8f0',
                    tickangle: -45,
                    tickfont: { size: 10 }
                },
                yaxis: {
                    title: 'Anomaly Score',
                    color: '#e2e8f0',
                    gridcolor: '#374151'
                },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#e2e8f0' },
                margin: { b: 100 }
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };

            Plotly.newPlot(div, data, layout, config);

        } catch (error) {
            console.error('Anomaly bar chart failed:', error);
            div.innerHTML = '<p class="loading-message">Failed to create anomaly bar chart</p>';
        }
    }

    // Add enhanced CSS styles for better analysis display
    addAnalysisStyles() {
        if (document.getElementById('enhanced-analysis-styles')) return;

        const styles = `
            <style id="enhanced-analysis-styles">
            .quality-summary {
                background: rgba(34, 197, 94, 0.1);
                border-left: 4px solid #22c55e;
                padding: 16px;
                margin-bottom: 20px;
                border-radius: 8px;
            }
            
            .quality-metrics {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
                margin-top: 10px;
            }
            
            .quality-metrics .metric {
                background: rgba(34, 197, 94, 0.2);
                padding: 8px 12px;
                border-radius: 16px;
                font-size: 14px;
            }
            
            .correlation-grid {
                display: grid;
                gap: 12px;
                margin-top: 12px;
            }
            
            .correlation-item {
                background: rgba(59, 130, 246, 0.1);
                border: 1px solid rgba(59, 130, 246, 0.2);
                padding: 16px;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .correlation-item.very-strong {
                border-color: #ef4444;
                background: rgba(239, 68, 68, 0.1);
            }
            
            .correlation-item.strong {
                border-color: #f59e0b;
                background: rgba(245, 158, 11, 0.1);
            }
            
            .correlation-pair {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .correlation-arrow {
                font-size: 18px;
                color: #60a5fa;
            }
            
            .correlation-strength {
                display: flex;
                flex-direction: column;
                align-items: flex-end;
            }
            
            .correlation-value {
                font-weight: bold;
                color: #60a5fa;
            }
            
            .correlation-label {
                font-size: 12px;
                color: #94a3b8;
                text-transform: uppercase;
            }
            
            .moderate-correlations {
                display: grid;
                gap: 8px;
                margin-top: 12px;
            }
            
            .moderate-corr-item {
                background: rgba(96, 165, 250, 0.05);
                border: 1px solid rgba(96, 165, 250, 0.1);
                padding: 12px;
                border-radius: 6px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .corr-features {
                color: #e2e8f0;
            }
            
            .corr-value.positive {
                color: #22c55e;
            }
            
            .corr-value.negative {
                color: #ef4444;
            }
            
            .correlation-summary {
                background: rgba(96, 165, 250, 0.1);
                padding: 16px;
                border-radius: 8px;
                border-left: 4px solid #60a5fa;
            }
            
            .correlation-insight {
                margin-top: 12px;
                font-style: italic;
                color: #94a3b8;
            }
            
            .no-correlations {
                background: rgba(156, 163, 175, 0.1);
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            
            .no-correlations ul {
                text-align: left;
                margin: 12px 0;
            }
            
            .tip {
                background: rgba(59, 130, 246, 0.1);
                padding: 12px;
                border-radius: 6px;
                border-left: 3px solid #3b82f6;
                margin-top: 12px;
            }
            
            .cluster-results {
                background: rgba(168, 85, 247, 0.1);
                border-left: 4px solid #a855f7;
                padding: 16px;
                border-radius: 8px;
            }
            
            .cluster-info {
                display: flex;
                gap: 20px;
                margin: 12px 0;
                font-size: 14px;
            }
            
            .cluster-info span {
                background: rgba(168, 85, 247, 0.2);
                padding: 6px 12px;
                border-radius: 12px;
            }
            
            .cluster-insight {
                margin-top: 12px;
                font-style: italic;
                color: #94a3b8;
            }
            
            .variability-insight {
                background: rgba(245, 158, 11, 0.1);
                border-left: 4px solid #f59e0b;
                padding: 16px;
                border-radius: 8px;
            }
            
            /* Anomaly Results Styling */
            .anomaly-results {
                background: linear-gradient(145deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05));
                border: 1px solid rgba(239, 68, 68, 0.2);
            }
            
            .anomaly-summary {
                margin-bottom: 20px;
            }
            
            .anomaly-stat-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
                margin-top: 16px;
            }
            
            .anomaly-stat {
                background: rgba(239, 68, 68, 0.1);
                padding: 16px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid rgba(239, 68, 68, 0.2);
            }
            
            .stat-number {
                font-size: 24px;
                font-weight: bold;
                color: #ef4444;
                margin-bottom: 4px;
            }
            
            .stat-label {
                font-size: 12px;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .severity-indicator {
                padding: 16px;
                border-radius: 8px;
                margin: 16px 0;
                text-align: center;
            }
            
            .severity-high {
                background: rgba(239, 68, 68, 0.15);
                border: 2px solid #ef4444;
            }
            
            .severity-moderate {
                background: rgba(245, 158, 11, 0.15);
                border: 2px solid #f59e0b;
            }
            
            .severity-low {
                background: rgba(34, 197, 94, 0.15);
                border: 2px solid #22c55e;
            }
            
            .severity-badge {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                margin-bottom: 8px;
            }
            
            .severity-high .severity-badge {
                background: #ef4444;
                color: white;
            }
            
            .severity-moderate .severity-badge {
                background: #f59e0b;
                color: white;
            }
            
            .severity-low .severity-badge {
                background: #22c55e;
                color: white;
            }
            
            .severity-text {
                margin: 0;
                font-size: 14px;
                color: #94a3b8;
            }
            
            .anomaly-indices {
                margin-top: 20px;
                padding: 16px;
                background: rgba(55, 65, 81, 0.5);
                border-radius: 8px;
            }
            
            .indices-list {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 10px;
            }
            
            .anomaly-index {
                background: #ef4444;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
            }
            
            .more-indices {
                margin-top: 10px;
                font-style: italic;
                color: #94a3b8;
                font-size: 14px;
            }
            
            .anomaly-scores {
                margin-top: 20px;
                padding: 16px;
                background: rgba(55, 65, 81, 0.3);
                border-radius: 8px;
                border: 1px solid rgba(75, 85, 99, 0.3);
            }
            
            .score-metrics {
                display: flex;
                gap: 20px;
                margin: 10px 0;
            }
            
            .score-metrics span {
                background: rgba(96, 165, 250, 0.2);
                padding: 6px 12px;
                border-radius: 12px;
                font-size: 14px;
            }
            
            .score-explanation {
                margin-top: 10px;
                font-size: 13px;
                color: #94a3b8;
                font-style: italic;
            }
            
            .no-anomalies {
                text-align: center;
                padding: 40px 20px;
            }
            
            .no-anomaly-icon {
                font-size: 48px;
                margin-bottom: 16px;
            }
            
            .no-anomalies h5 {
                color: #22c55e;
                margin-bottom: 12px;
                font-size: 18px;
            }
            
            .suggestions {
                background: rgba(59, 130, 246, 0.1);
                padding: 16px;
                border-radius: 8px;
                margin-top: 20px;
                text-align: left;
            }
            
            .suggestions ul {
                margin-top: 8px;
            }
            
            .suggestions li {
                margin-bottom: 4px;
                color: #cbd5e1;
            }
            
            /* Responsive design for analysis results */
            @media (max-width: 768px) {
                .anomaly-stat-grid {
                    grid-template-columns: 1fr;
                    gap: 12px;
                }
                
                .correlation-item {
                    flex-direction: column;
                    gap: 12px;
                    text-align: center;
                }
                
                .correlation-strength {
                    align-items: center;
                }
                
                .quality-metrics {
                    flex-direction: column;
                    gap: 8px;
                }
                
                .cluster-info {
                    flex-direction: column;
                    gap: 8px;
                }
                
                .score-metrics {
                    flex-direction: column;
                    gap: 8px;
                }
            }
            </style>
        `;

        document.head.insertAdjacentHTML('beforeend', styles);
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

    populateCausalColumns() {
        if (!this.currentData) return;

        const treatmentSelect = document.getElementById('treatmentColumn');
        const outcomeSelect = document.getElementById('outcomeColumn');

        if (!treatmentSelect || !outcomeSelect) return;

        // Clear existing options
        treatmentSelect.innerHTML = '<option value="">Choose treatment column...</option>';
        outcomeSelect.innerHTML = '<option value="">Choose outcome column...</option>';

        // Add numeric columns
        if (this.currentData.columns && this.currentData.dtypes) {
            this.currentData.columns.forEach(column => {
                const dataType = this.currentData.dtypes[column];
                if (dataType && (dataType.includes('int') || dataType.includes('float'))) {
                    const option1 = document.createElement('option');
                    option1.value = column;
                    option1.textContent = column;
                    treatmentSelect.appendChild(option1);

                    const option2 = document.createElement('option');
                    option2.value = column;
                    option2.textContent = column;
                    outcomeSelect.appendChild(option2);
                }
            });
        }
    }

    async buildModel() {
        if (!this.currentData) {
            this.showNotification('Please load data before building a model', 'warning');
            return;
        }

        const targetSelect = document.getElementById('targetColumn');
        if (!targetSelect) {
            this.showNotification('Target column selector not found', 'error');
            return;
        }

        const targetColumn = targetSelect.value;
        if (!targetColumn) {
            this.showNotification('Please select a target column', 'warning');
            return;
        }

        this.showLoading('Building AI predictive model...');
        this.addActivity('Modeling', `Building model for ${targetColumn}`);

        try {
            console.log(`Building model for target column: ${targetColumn}`);

            const response = await fetch(`${this.serverUrl}/api/simulation/model/build`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    target_column: targetColumn,
                    data_shape: this.currentData ? {
                        rows: this.currentData.shape ? this.currentData.shape[0] : 0,
                        columns: this.currentData.columns ? this.currentData.columns.length : 0
                    } : null
                })
            });

            let result;
            const responseText = await response.text();
            try {
                result = JSON.parse(responseText);
            } catch (e) {
                console.error('Failed to parse response:', responseText);
                throw new Error('Invalid response from server');
            }

            if (!response.ok) {
                throw new Error(result.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            if (result.success) {
                console.log('Model built successfully:', result);
                this.displayModelResults(result);
                this.models[targetColumn] = result;

                await this.explainModel(targetColumn);

                // Enable forecasting
                const forecastBtn = document.getElementById('forecastBtn');
                if (forecastBtn) {
                    forecastBtn.disabled = false;
                }

                await this.updateSystemStats();
                this.addActivity('Model Built', `${targetColumn} prediction model (MAE: ${result.mae.toFixed(4)})`);
                this.showNotification(`Model built successfully! MAE: ${result.mae.toFixed(4)}`, 'success');
            } else {
                const errorMsg = result.error || 'Unknown error occurred';
                this.showNotification(`Model building failed: ${errorMsg}`, 'error');
                this.addActivity('Model Failed', errorMsg);
                console.error('Model building failed:', errorMsg);
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
                .sort(([, a], [, b]) => b - a)
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

        this.populateWhatIfControls(result);
    }

    async explainModel(targetColumn) {
        this.showLoading('Generating model explanations...');

        try {
            const response = await fetch(`${this.serverUrl}/api/simulation/model/explain`, {
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
                this.displayShapExplanation(result.explanation);
                this.showNotification('Model explanation generated!', 'success');
            } else {
                this.showNotification(`Explanation failed: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Explanation failed:', error);
            this.showNotification(`Explanation failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayShapExplanation(explanation) {
        const resultsDiv = document.getElementById('modelResults');
        if (!resultsDiv) return;

        let html = '<div class="shap-explanation" style="margin-top: 20px; padding: 16px; background: rgba(168, 85, 247, 0.1); border-radius: 8px; border-left: 4px solid #a855f7;">';
        html += '<h4 style="color: #a855f7; margin-bottom: 12px;">🔍 Model Explainability (SHAP)</h4>';
        html += '<p style="color: #94a3b8; margin-bottom: 12px;">Feature impact on predictions (higher = more important):</p>';
        html += '<ul style="margin-left: 20px;">';

        const sorted = Object.entries(explanation.feature_importance)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 10);

        sorted.forEach(([feature, importance]) => {
            const percentage = (importance * 100).toFixed(1);
            html += `<li style="margin-bottom: 8px; color: #e2e8f0;"><strong>${this.escapeHtml(feature)}:</strong> ${importance.toFixed(4)} (${percentage}% impact)</li>`;
        });

        html += '</ul>';
        html += '<p style="margin-top: 12px; font-size: 13px; color: #94a3b8; font-style: italic;">💡 SHAP values show how much each feature contributes to predictions</p>';
        html += '</div>';

        // Append to existing results instead of replacing
        resultsDiv.innerHTML += html;
    }

    populateWhatIfControls(modelResult) {
        const whatIfControls = document.getElementById('whatIfControls');
        if (!whatIfControls) return;

        const features = modelResult.features || [];

        if (features.length === 0) {
            whatIfControls.innerHTML = '<div class="loading-message">No features available</div>';
            return;
        }

        let html = '<div class="what-if-controls">';

        // Create input for each feature
        features.slice(0, 5).forEach(feature => { // Limit to top 5 features
            html += `
            <div class="form-group">
                <label>${this.escapeHtml(feature)}:</label>
                <input type="number" 
                       id="whatif_${feature}" 
                       class="whatif-input" 
                       placeholder="Enter value" 
                       step="0.1">
            </div>
        `;
        });

        html += '</div>';
        html += `<button class="forecast-button" id="runWhatIfBtn">🎭 Run What-If Scenario</button>`;

        whatIfControls.innerHTML = html;

        // Add event listener for the button
        const runWhatIfBtn = document.getElementById('runWhatIfBtn');
        if (runWhatIfBtn) {
            runWhatIfBtn.addEventListener('click', () => this.runWhatIfAnalysis(modelResult.target_column));
        }
    }

    async runWhatIfAnalysis(targetColumn) {
        // Gather all input values
        const whatIfInputs = document.querySelectorAll('.whatif-input');
        const featureChanges = {};

        whatIfInputs.forEach(input => {
            const featureName = input.id.replace('whatif_', '');
            const value = parseFloat(input.value);
            if (!isNaN(value) && value !== 0) {
                featureChanges[featureName] = value;
            }
        });

        if (Object.keys(featureChanges).length === 0) {
            this.showNotification('Please enter at least one value to test', 'warning');
            return;
        }

        this.showLoading('Running what-if scenario...');

        try {
            const response = await fetch(`${this.serverUrl}/api/simulation/whatif`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    target_column: targetColumn,
                    feature_changes: featureChanges
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.success) {
                this.displayWhatIfResults(result.analysis);
                this.showNotification('What-if analysis complete!', 'success');
            } else {
                this.showNotification(`Analysis failed: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('What-if analysis failed:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayWhatIfResults(analysis) {
        const resultsDiv = document.getElementById('scenarioResults');
        if (!resultsDiv) return;

        let html = '<div class="scenario-results">';
        html += '<h4>🎭 What-If Analysis Results</h4>';
        html += `<p><strong>Baseline Prediction:</strong> ${analysis.baseline_prediction.toFixed(2)}</p>`;
        html += `<p><strong>Modified Prediction:</strong> ${analysis.modified_prediction.toFixed(2)}</p>`;
        html += `<p><strong>Impact:</strong> ${analysis.impact > 0 ? '+' : ''}${analysis.impact.toFixed(2)} (${analysis.percent_change.toFixed(1)}%)</p>`;

        if (analysis.applied_changes && Object.keys(analysis.applied_changes).length > 0) {
            html += '<h5>📊 Changes Applied:</h5><ul>';
            Object.entries(analysis.applied_changes).forEach(([feature, value]) => {
                html += `<li><strong>${this.escapeHtml(feature)}:</strong> ${value.toFixed(2)}</li>`;
            });
            html += '</ul>';
        }

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

        // Add explanation text
        // Calculate container height based on window size
        const containerHeight = Math.max(400, window.innerHeight - 400); // More space for explanation
        graphContainer.innerHTML = `
           <div class="graph-info-compact">
               <div class="graph-legend">
                   <span class="legend-item">
                     <span class="legend-dot" style="background: #3b82f6"></span>
                     <strong>Columns</strong>
                   </span>
                   <span class="legend-item">
                      <span class="legend-dot" style="background: #22c55e"></span>
                      <strong>Patterns</strong>
                   </span>
                   <span class="legend-item">
                      <strong>Lines = Relationships</strong>
                   </span>
               </div>
           </div>
          <div id="graphPlot" style="width: 100%; height: ${containerHeight}px; margin-top: 10px;"></div>
        `;

        // FIXED: Better node positioning with proper spread
        const nodeCount = data.nodes.length;
        const minRadius = 1.5; // Minimum radius
        const maxRadius = 3.0; // Maximum radius
        // Better approach - use D3 force simulation or increase base radius
        const radius = Math.max(2.0, Math.min(4.0, Math.sqrt(nodeCount) * 0.8));

        const nodes = data.nodes.map((node, index) => {
              // Create concentric circles for better distribution
            const layer = Math.floor(index / 6); // 6 nodes per layer instead of 8
            const angleOffset = layer * (Math.PI / 6); // 30-degree rotation per layer
            const currentRadius = radius + (layer * 1.5); // More spacing between layers
            const nodesInLayer = Math.min(6, nodeCount - (layer * 6));
            const angle = (2 * Math.PI * (index % 6)) / nodesInLayer + angleOffset;

            // Add controlled randomization to prevent perfect alignment
            const randomOffset = 0.15;
            const xOffset = (Math.random() - 0.5) * randomOffset;
            const yOffset = (Math.random() - 0.5) * randomOffset;

            return {
                name: node.label || node.id || 'Unknown',
                x: currentRadius * Math.cos(angle) + xOffset,
                y: currentRadius * Math.sin(angle) + yOffset,
                type: 'scatter',
                mode: 'markers+text',
                marker: {
                    size: Math.max(15, Math.min(25, 35 - nodeCount * 0.5)), // Smaller, more reasonable sizes
                    color: node.type === 'attribute' ? '#3b82f6' : '#22c55e',
                    symbol: node.type === 'attribute' ? 'circle' : 'diamond',
                    line: {
                        color: '#ffffff',
                        width: 2
                    },
                    opacity: 0.9
                },
                text: node.label || node.id,
                textposition: 'top center', // Put text ABOVE nodes to avoid overlap
                textfont: {
                    size: Math.max(8, Math.min(11, 13 - nodeCount * 0.1)),
                    color: '#1e293b', // Dark text for contrast on light background
                    family: 'Arial, sans-serif',
                    weight: 'bold'
                },
                hoverinfo: 'text',
                hovertext: `<b>${node.label || node.id}</b><br>Type: ${node.type || 'entity'}<br>Data Type: ${node.data_type || 'N/A'}`,
                hoverlabel: {
                    bgcolor: '#1e293b',
                    bordercolor: '#3b82f6',
                    font: { size: 12, color: '#ffffff' }
                }
            };
        });

        // Debug output - add this after nodes creation
        console.log('Node count:', nodeCount);
        console.log('Radius:', radius);
        console.log('First node position:', nodes[0]);
        console.log('All node positions:', nodes.map(n => ({ name: n.name, x: n.x, y: n.y })));

        // Create edges as lines between nodes
        // Create edges with improved styling
        const edges = [];
        if (data.edges && Array.isArray(data.edges)) {
            data.edges.forEach(edge => {
                const sourceNode = nodes.findIndex(n => n.name === edge.source);
                const targetNode = nodes.findIndex(n => n.name === edge.target);
                if (sourceNode !== -1 && targetNode !== -1) {
                    const lineWidth = edge.strength ? Math.max(2, edge.strength * 5) : 2;
                    const lineColor = edge.strength > 0.7 ? '#ef4444' : // Strong = Red
                        edge.strength > 0.5 ? '#f59e0b' : // Medium = Orange  
                            '#64748b'; // Weak = Gray

                    edges.push({
                        type: 'scatter',
                        mode: 'lines',
                        x: [nodes[sourceNode].x, nodes[targetNode].x],
                        y: [nodes[sourceNode].y, nodes[targetNode].y],
                        line: {
                            color: lineColor,
                            width: lineWidth
                        },
                        opacity: 0.8,
                        hoverinfo: 'text',
                        hovertext: `${edge.relationship || 'Related'}<br>Strength: ${edge.strength ? edge.strength.toFixed(2) : 'N/A'}`,
                        showlegend: false
                    });
                }
            });
        }

        // Enhanced layout configuration with better containment
        const layout = {
            showlegend: false,
            hovermode: 'closest',
            margin: { t: 50, l: 50, r: 50, b: 50 }, // More generous margins
            xaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                range: [-4.5, 4.5], // Slightly wider range for better spacing
                fixedrange: false, // Prevent x-axis zoom
                //constrain: 'domain', // Constrain to the plot area
            },
            yaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                range: [-4.5, 4.5], // Slightly wider range for better spacing
                scaleanchor: 'x', // Keep aspect ratio square
                scaleratio: 1,
                fixedrange: false, // Prevent y-axis zoom
                //constrain: 'domain', // Constrain to the plot area
            },
            plot_bgcolor: '#f1f5f9',
            paper_bgcolor: '#ffffff',
            autosize: true,
            dragmode: 'pan',
            modebar: {
                remove: ['zoomIn', 'zoomOut', 'resetScale', 'select2d', 'lasso2d'],
                orientation: 'v'
            },
            shapes: [], // Will be used for edges to ensure they stay within bounds
            annotations: [] // Will be used for node labels to ensure they stay within bounds
        };

        // Create the interactive plot
        const plotDiv = document.createElement('div');
        plotDiv.style.width = '100%';
        plotDiv.style.height = '1200px';
        graphContainer.appendChild(plotDiv);

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: [
                'select2d', 'lasso2d', 'hoverClosestCartesian', 'hoverCompareCartesian'
            ],
            displaylogo: false,
            doubleClick: 'reset',
            scrollZoom: true // Enable scroll zoom
        };

        Plotly.newPlot(plotDiv, [...edges, ...nodes], layout, config);

        // Add event listener for node click events
        plotDiv.on('plotly_click', (data) => {
            const point = data.points[0];
            if (point && point.data.type === 'scatter' && point.data.mode.includes('markers')) {
                // Highlight connected nodes and edges
                this.highlightConnections(plotDiv, point.data.name, edges, nodes);
            }
        });
    }

    highlightConnections(plotDiv, nodeName, edges, nodes) {
        // Find connected nodes
        const connectedNodes = new Set();
        edges.forEach(edge => {
            const xCoords = edge.x;
            const sourceNode = nodes.find(n => n.x === xCoords[0]);
            const targetNode = nodes.find(n => n.x === xCoords[1]);

            if (sourceNode && targetNode) {
                if (sourceNode.name === nodeName) {
                    connectedNodes.add(targetNode.name);
                } else if (targetNode.name === nodeName) {
                    connectedNodes.add(sourceNode.name);
                }
            }
        });

        // Update node markers
        const updatedNodes = nodes.map(node => {
            const isSelected = node.name === nodeName;
            const isConnected = connectedNodes.has(node.name);

            return {
                marker: {
                    ...node.marker,
                    size: isSelected ? 40 : isConnected ? 35 : 30,
                    opacity: isSelected || isConnected ? 1 : 0.4
                }
            };
        });

        // Update edge styling
        const updatedEdges = edges.map(edge => {
            const xCoords = edge.x;
            const sourceNode = nodes.find(n => n.x === xCoords[0]);
            const targetNode = nodes.find(n => n.x === xCoords[1]);

            const isConnected = (sourceNode && targetNode) &&
                (sourceNode.name === nodeName || targetNode.name === nodeName);

            return {
                line: {
                    ...edge.line,
                    opacity: isConnected ? 1 : 0.2
                }
            };
        });

        // Apply updates
        Plotly.update(plotDiv,
            {
                'marker': updatedNodes.map(n => n.marker),
                'line': updatedEdges.map(e => e.line)
            }
        );
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