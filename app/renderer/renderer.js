// Add these methods to your DigitalTwinApp class in renderer.js

    // Enhanced Knowledge Graph Display with Visuals
    displayKnowledgeGraph(data) {
        const graphContainer = document.getElementById('knowledgeGraph');
        if (!graphContainer) return;

        if (!data || !data.nodes || !Array.isArray(data.nodes) || data.nodes.length === 0) {
            graphContainer.innerHTML = '<div class="loading-message">No knowledge graph data available. Run pattern analysis first to build the graph.</div>';
            return;
        }

        // Create a visual network graph using D3-like visualization
        let html = '<div class="graph-display">';
        
        // Graph Statistics Summary
        html += `
            <div class="graph-stats">
                <div class="graph-stat">
                    <span class="stat-number">${data.nodes.length}</span>
                    <span class="stat-label">Entities</span>
                </div>
                <div class="graph-stat">
                    <span class="stat-number">${data.edges ? data.edges.length : 0}</span>
                    <span class="stat-label">Relationships</span>
                </div>
            </div>
        `;
        
        // Visual Network Display
        html += '<div class="network-container">';
        html += '<svg id="networkSVG" width="100%" height="300"></svg>';
        html += '</div>';
        
        // Entity Details
        html += '<div class="entities-panel">';
        html += '<h4>📊 Data Entities</h4>';
        html += '<div class="entity-grid">';
        
        data.nodes.forEach((node, index) => {
            const nodeType = node.type || 'entity';
            const nodeColor = this.getNodeColor(nodeType);
            
            html += `
                <div class="entity-card" style="border-left-color: ${nodeColor}">
                    <div class="entity-header">
                        <span class="entity-dot" style="background: ${nodeColor}"></span>
                        <strong>${this.escapeHtml(node.label || node.id || 'Unknown')}</strong>
                    </div>
                    <div class="entity-details">
                        <span class="entity-type">${this.escapeHtml(nodeType)}</span>
                        ${node.data_type ? `<span class="entity-datatype">${this.escapeHtml(node.data_type)}</span>` : ''}
                    </div>
                </div>
            `;
        });
        
        html += '</div></div>';

        // Relationships Panel
        if (data.edges && Array.isArray(data.edges) && data.edges.length > 0) {
            html += '<div class="relationships-panel">';
            html += '<h4>🔗 Relationships</h4>';
            html += '<div class="relationships-list">';
            
            data.edges.forEach(edge => {
                const strength = typeof edge.strength === 'number' ? edge.strength : 0;
                const strengthClass = strength > 0.8 ? 'strength-high' : strength > 0.5 ? 'strength-medium' : 'strength-low';
                const strengthLabel = strength > 0.8 ? 'Strong' : strength > 0.5 ? 'Medium' : 'Weak';
                
                html += `
                    <div class="relationship-item ${strengthClass}">
                        <div class="relationship-connection">
                            <span class="node-name">${this.escapeHtml(edge.source || '')}</span>
                            <span class="relationship-arrow">→</span>
                            <span class="node-name">${this.escapeHtml(edge.target || '')}</span>
                        </div>
                        <div class="relationship-meta">
                            <span class="relationship-type">${this.escapeHtml(edge.relationship || 'connected')}</span>
                            <span class="relationship-strength">${strengthLabel} (${strength.toFixed(2)})</span>
                        </div>
                    </div>
                `;
            });
            
            html += '</div></div>';
        }
        
        html += '</div>';
        graphContainer.innerHTML = html;
        
        // Draw the network visualization
        this.drawNetworkGraph(data);
        
        // Add graph styles
        this.addGraphStyles();
    }

    // Draw Network Graph using SVG
    drawNetworkGraph(data) {
        const svg = document.getElementById('networkSVG');
        if (!svg || !data.nodes) return;
        
        const svgRect = svg.getBoundingClientRect();
        const width = svgRect.width || 600;
        const height = 300;
        
        // Clear existing content
        svg.innerHTML = '';
        
        // Create nodes with circular layout
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) / 3;
        
        const nodes = data.nodes.map((node, i) => {
            const angle = (2 * Math.PI * i) / data.nodes.length;
            return {
                ...node,
                x: centerX + radius * Math.cos(angle),
                y: centerY + radius * Math.sin(angle)
            };
        });
        
        // Draw edges first (so they appear behind nodes)
        if (data.edges) {
            data.edges.forEach(edge => {
                const sourceNode = nodes.find(n => n.id === edge.source || n.label === edge.source);
                const targetNode = nodes.find(n => n.id === edge.target || n.label === edge.target);
                
                if (sourceNode && targetNode) {
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', sourceNode.x);
                    line.setAttribute('y1', sourceNode.y);
                    line.setAttribute('x2', targetNode.x);
                    line.setAttribute('y2', targetNode.y);
                    line.setAttribute('stroke', '#60a5fa');
                    line.setAttribute('stroke-width', Math.max(1, (edge.strength || 0.5) * 3));
                    line.setAttribute('stroke-opacity', '0.6');
                    svg.appendChild(line);
                    
                    // Add relationship label
                    const labelX = (sourceNode.x + targetNode.x) / 2;
                    const labelY = (sourceNode.y + targetNode.y) / 2;
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', labelX);
                    text.setAttribute('y', labelY);
                    text.setAttribute('text-anchor', 'middle');
                    text.setAttribute('fill', '#94a3b8');
                    text.setAttribute('font-size', '10');
                    text.textContent = edge.relationship || 'connected';
                    svg.appendChild(text);
                }
            });
        }
        
        // Draw nodes
        nodes.forEach(node => {
            const nodeColor = this.getNodeColor(node.type);
            
            // Node circle
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', node.x);
            circle.setAttribute('cy', node.y);
            circle.setAttribute('r', 20);
            circle.setAttribute('fill', nodeColor);
            circle.setAttribute('stroke', '#ffffff');
            circle.setAttribute('stroke-width', '2');
            circle.style.cursor = 'pointer';
            svg.appendChild(circle);
            
            // Node label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', node.x);
            text.setAttribute('y', node.y + 35);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('fill', '#e2e8f0');
            text.setAttribute('font-size', '12');
            text.setAttribute('font-weight', 'bold');
            const label = (node.label || node.id || 'Unknown');
            text.textContent = label.length > 10 ? label.substring(0, 10) + '...' : label;
            svg.appendChild(text);
        });
    }

    // Get color for different node types
    getNodeColor(nodeType) {
        const colors = {
            'attribute': '#3b82f6',    // Blue for data attributes
            'pattern': '#a855f7',      // Purple for patterns
            'cluster': '#f59e0b',      // Orange for clusters
            'correlation': '#ef4444',   // Red for correlations
            'entity': '#22c55e',       // Green for generic entities
            'unknown': '#6b7280'       // Gray for unknown
        };
        return colors[nodeType] || colors['unknown'];
    }

    // Enhanced Simulation Results with What-If Controls
    displayModelResults(result) {
        const resultsDiv = document.getElementById('modelResults');
        if (!resultsDiv) return;

        let html = '<div class="model-results enhanced-model-results">';
        html += `<h4>🤖 AI Model Successfully Built!</h4>`;
        html += `
            <div class="model-summary">
                <div class="model-metric">
                    <span class="metric-label">Target Variable</span>
                    <span class="metric-value">${this.escapeHtml(result.target_column)}</span>
                </div>
                <div class="model-metric">
                    <span class="metric-label">Model Accuracy (MAE)</span>
                    <span class="metric-value">${result.mae.toFixed(4)}</span>
                </div>
                <div class="model-metric">
                    <span class="metric-label">Features Used</span>
                    <span class="metric-value">${result.features ? result.features.length : 0}</span>
                </div>
            </div>
        `;
        
        if (result.features && Array.isArray(result.features)) {
            html += `<p><strong>Input Features:</strong> ${result.features.map(f => this.escapeHtml(f)).join(', ')}</p>`;
        }
        
        if (result.feature_importance) {
            html += '<div class="feature-importance-chart">';
            html += '<h5>📊 Feature Importance:</h5>';
            html += '<div class="importance-bars">';
            
            const sortedFeatures = Object.entries(result.feature_importance)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5);
                
            sortedFeatures.forEach(([feature, importance]) => {
                const percentage = (importance * 100);
                html += `
                    <div class="importance-bar-container">
                        <div class="importance-label">
                            <span class="feature-name">${this.escapeHtml(feature)}</span>
                            <span class="importance-value">${percentage.toFixed(1)}%</span>
                        </div>
                        <div class="importance-bar">
                            <div class="importance-fill" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            });
            html += '</div></div>';
        }
        
        html += '<p class="model-info">💡 <em>Lower MAE indicates better model accuracy</em></p>';
        html += '</div>';
        
        resultsDiv.innerHTML = html;
        
        // Generate What-If Controls
        this.generateWhatIfControls(result);
    }

    // Generate Interactive What-If Analysis Controls
    generateWhatIfControls(modelResult) {
        const controlsDiv = document.getElementById('whatIfControls');
        if (!controlsDiv || !modelResult.features) return;

        let html = '<div class="what-if-panel">';
        html += '<h4>🎭 Interactive What-If Analysis</h4>';
        html += '<p>Adjust feature values to see how they impact predictions:</p>';
        html += '<div class="what-if-controls-grid">';

        // Create sliders for top features based on importance
        const features = modelResult.features.slice(0, 4); // Limit to top 4 features
        
        features.forEach((feature, index) => {
            html += `
                <div class="what-if-control">
                    <label class="control-label">${this.escapeHtml(feature)}</label>
                    <div class="slider-container">
                        <input type="range" 
                               id="whatif_${feature}" 
                               class="what-if-slider" 
                               min="-2" 
                               max="2" 
                               step="0.1" 
                               value="0"
                               data-feature="${feature}">
                        <div class="slider-value">
                            <span id="value_${feature}">0.0</span>
                        </div>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        html += '<button class="what-if-button" id="runWhatIfBtn">🔮 Run What-If Analysis</button>';
        html += '</div>';

        controlsDiv.innerHTML = html;

        // Add event listeners
        features.forEach(feature => {
            const slider = document.getElementById(`whatif_${feature}`);
            const valueDisplay = document.getElementById(`value_${feature}`);
            
            if (slider && valueDisplay) {
                slider.addEventListener('input', (e) => {
                    valueDisplay.textContent = parseFloat(e.target.value).toFixed(1);
                });
            }
        });

        // What-if analysis button
        const whatIfButton = document.getElementById('runWhatIfBtn');
        if (whatIfButton) {
            whatIfButton.addEventListener('click', () => this.runWhatIfAnalysis());
        }

        this.addSimulationStyles();
    }

    // Run What-If Analysis
    async runWhatIfAnalysis() {
        const sliders = document.querySelectorAll('.what-if-slider');
        const changes = {};
        
        sliders.forEach(slider => {
            const feature = slider.dataset.feature;
            const value = parseFloat(slider.value);
            if (value !== 0) {
                changes[feature] = value;
            }
        });

        if (Object.keys(changes).length === 0) {
            this.showNotification('Adjust some feature values first', 'warning');
            return;
        }

        this.showLoading('Running what-if analysis...');

        try {
            const targetSelect = document.getElementById('targetColumn');
            const targetColumn = targetSelect ? targetSelect.value : '';

            const response = await fetch(`${this.serverUrl}/api/simulation/whatif`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    target_column: targetColumn,
                    feature_changes: changes
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayWhatIfResults(result, changes);
            this.addActivity('What-If Analysis', `Analyzed ${Object.keys(changes).length} feature changes`);

        } catch (error) {
            console.error('What-if analysis failed:', error);
            this.showNotification(`What-if analysis failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    // Display What-If Results
    displayWhatIfResults(result, changes) {
        const resultsDiv = document.getElementById('scenarioResults');
        if (!resultsDiv) return;

        let html = '<div class="what-if-results">';
        html += '<h4>🎭 What-If Analysis Results</h4>';
        
        if (result.success !== false) {
            const impact = result.impact || 0;
            const percentChange = result.percent_change || 0;
            const impactClass = impact > 0 ? 'positive-impact' : impact < 0 ? 'negative-impact' : 'neutral-impact';
            
            html += `
                <div class="scenario-comparison">
                    <div class="scenario-metric">
                        <span class="scenario-label">Baseline Prediction</span>
                        <span class="scenario-value">${(result.baseline_prediction || 0).toFixed(3)}</span>
                    </div>
                    <div class="scenario-metric">
                        <span class="scenario-label">Modified Prediction</span>
                        <span class="scenario-value">${(result.modified_prediction || 0).toFixed(3)}</span>
                    </div>
                    <div class="scenario-metric ${impactClass}">
                        <span class="scenario-label">Impact</span>
                        <span class="scenario-value">
                            ${impact >= 0 ? '+' : ''}${impact.toFixed(3)} (${percentChange.toFixed(1)}%)
                        </span>
                    </div>
                </div>
            `;
            
            html += '<div class="changes-applied">';
            html += '<h5>📝 Changes Applied:</h5>';
            html += '<ul>';
            Object.entries(changes).forEach(([feature, value]) => {
                const changeClass = value > 0 ? 'increase' : 'decrease';
                html += `<li class="${changeClass}"><strong>${this.escapeHtml(feature)}:</strong> ${value > 0 ? '+' : ''}${value.toFixed(1)}</li>`;
            });
            html += '</ul></div>';
        } else {
            html += `<p class="error-message">Analysis failed: ${result.error || 'Unknown error'}</p>`;
        }
        
        html += '</div>';
        resultsDiv.innerHTML = html;
    }