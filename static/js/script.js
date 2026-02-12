/**
 * Cancer Subtype Predictor - Logic Handler
 * Wrapped in DOMContentLoaded to ensure elements exist before selection.
 */
document.addEventListener('DOMContentLoaded', () => {
    // --- Element Selectors ---
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const fileInfo = document.getElementById('fileInfo');
    const uploadBtn = document.getElementById('uploadBtn');
    const loading = document.getElementById('loading');
    const resultsContainer = document.getElementById('resultsContainer');

    let selectedFile = null;

    // --- Global Functions (Attached to window for HTML onclick access) ---

    /**
     * Resets the upload interface to its initial state.
     */
    window.resetUpload = function() {
        selectedFile = null;
        if (fileInput) fileInput.value = '';
        if (uploadArea) uploadArea.style.display = 'block';
        if (fileInfo) fileInfo.style.display = 'none';
        if (resultsContainer) resultsContainer.style.display = 'none';
    };

    /**
     * Triggers a demo prediction using pre-loaded server data.
     */
    window.runDemo = function() {
        console.log('Demo initiated...');
        prepareForLoading();
        
        fetch('/demo')
            .then(response => {
                if (!response.ok) throw new Error('Network response was not ok');
                return response.json();
            })
            .then(data => handleServerResponse(data))
            .catch(error => handleUIError(error));
    };

    // --- Internal Helper Functions ---

    function handleFile(file) {
        if (!file) return;
        
        const validExtensions = ['.csv', '.tsv', '.txt'];
        const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        
        if (!hasValidExtension) {
            alert('Please upload a CSV, TSV, or TXT file.');
            return;
        }
        
        if (file.size > 16 * 1024 * 1024) {
            alert('File size exceeds 16MB limit.');
            return;
        }
        
        selectedFile = file;
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = formatFileSize(file.size);
        
        uploadArea.style.display = 'none';
        fileInfo.style.display = 'block';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function prepareForLoading() {
        loading.style.display = 'block';
        resultsContainer.style.display = 'none';
        uploadArea.style.display = 'none';
        fileInfo.style.display = 'none';
    }

    function handleServerResponse(data) {
        if (data.error) {
            throw new Error(data.error);
        }
        displayResults(data);
    }

    function handleUIError(error) {
        console.error('Application Error:', error);
        alert('Error: ' + error.message);
        loading.style.display = 'none';
        uploadArea.style.display = 'block';
    }

    /**
     * Renders Plotly charts and populates result fields.
     */
    function displayResults(data) {
        loading.style.display = 'none';
        resultsContainer.style.display = 'block';
        
        // Update Text Content
        document.getElementById('clusterName').textContent = data.cluster_name;
        const badge = document.getElementById('riskBadge');
        badge.textContent = `${data.emoji} ${data.risk_level}`;
        badge.style.backgroundColor = data.color;
        badge.style.color = 'white';
        
        document.getElementById('clusterID').textContent = data.cluster_id;
        document.getElementById('riskLevel').textContent = data.risk_level;
        document.getElementById('mortalityRate').textContent = data.mortality_rate;
        document.getElementById('confidence').textContent = data.confidence;
        document.getElementById('similarPatients').textContent = data.n_similar_patients.toLocaleString();
        
        // Dynamic styling
        document.querySelector('.main-result').style.borderLeftColor = data.color;
        
        // Render Charts via Plotly
        try {
            const config = { responsive: true, displayModeBar: false };
            Plotly.newPlot('gaugeChart', JSON.parse(data.plots.gauge).data, JSON.parse(data.plots.gauge).layout, config);
            Plotly.newPlot('comparisonChart', JSON.parse(data.plots.comparison).data, JSON.parse(data.plots.comparison).layout, config);
            Plotly.newPlot('distributionChart', JSON.parse(data.plots.distribution).data, JSON.parse(data.plots.distribution).layout, config);
        } catch (err) {
            console.error('Plotly rendering failed:', err);
        }
        
        // Generate Clinical Interpretation
        document.getElementById('interpretation').innerHTML = generateInterpretation(data);
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    function generateInterpretation(data) {
        const conf = parseFloat(data.confidence);
        let confText = conf > 80 ? 'High confidence match.' : (conf > 60 ? 'Moderate confidence.' : 'Lower confidence; clinical review recommended.');
        
        let recs = data.risk_level === 'Good Prognosis' 
            ? '<li>Standard protocols may be effective.</li><li>Regular monitoring.</li>'
            : (data.risk_level === 'Intermediate Risk' 
                ? '<li>Personalized strategies advised.</li><li>Targeted therapy discussion.</li>'
                : '<li><strong>Aggressive treatment recommended.</strong></li><li>Clinical trial consideration.</li>');

        return `
            <p><strong>📊 Summary:</strong> Patient classified as <strong>${data.risk_level}</strong>.</p>
            <p><strong>🎯 Confidence:</strong> ${data.confidence}% - ${confText}</p>
            <p><strong>💊 Recommendations:</strong><ul>${recs}</ul></p>
            <p><small>⚠️ Research use only. Consult a professional.</small></p>
        `;
    }

    // --- Event Listeners ---
    if (fileInput) {
        fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
    }

    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => {
            if (!selectedFile) return;
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            fileInfo.style.display = 'none';
            loading.style.display = 'block';
            
            fetch('/predict', { method: 'POST', body: formData })
                .then(res => res.json())
                .then(data => handleServerResponse(data))
                .catch(err => handleUIError(err));
        });
    }
});