// frontend/comparison.js

// This file is used by nb_vs_svc.html and nb_vs_dt.html

const SERVER_URL = 'http://localhost:5000';

document.addEventListener('DOMContentLoaded', () => {
    // 1. Get data from localStorage
    const resultsDataJSON = localStorage.getItem('analysisResults');
    const resultsContainer = document.getElementById('results-container');
    const errorMessageDiv = document.getElementById('error-message');

    if (!resultsDataJSON) {
        // Show error if no data is found
        errorMessageDiv.classList.remove('hidden');
        return;
    }

    // 2. We have data, so show the results container
    resultsContainer.classList.remove('hidden');
    const analysisData = JSON.parse(resultsDataJSON);

    // 3. Check which page we're on and render the correct comparison
    const path = window.location.pathname;

    if (path.includes('nb_vs_svc.html')) {
        renderComparisonPage("Naive Bayes", "SVC", analysisData);
    } else if (path.includes('nb_vs_dt.html')) {
        renderComparisonPage("Naive Bayes", "Decision Tree", analysisData);
    }
});


// --- Helper Functions (Copied from main script.js) ---

function renderComparisonPage(model1Name, model2Name, analysisData) {
    const model1Data = analysisData.models[model1Name];
    const model2Data = analysisData.models[model2Name];
    const resultsContainer = document.getElementById('results-container');

    // Check if models exist in the data
    if (!model1Data || !model2Data) {
        resultsContainer.innerHTML = "<p>Error: Model data not found. Please re-run analysis.</p>";
        return;
    }

    // Fill page with HTML
    resultsContainer.innerHTML += `
        <div class="comparison-grid">
            
            <div class="comparison-column">
                <h3>${model1Name}</h3>
                <div class="metrics-summary">
                    ${renderMetricBox("Accuracy", model1Data.accuracy, true)}
                    ${renderMetricBox("Precision", model1Data.classification_report['weighted avg'].precision)}
                    ${renderMetricBox("Recall", model1Data.classification_report['weighted avg'].recall)}
                    ${renderMetricBox("F1-Score", model1Data.classification_report['weighted avg']['f1-score'])}
                </div>
                <div class="card">
                    <h3>Classification Report</h3>
                    ${createReportTable(model1Data.classification_report)}
                </div>
                <div class="card">
                    <h3>Confusion Matrix</h3>
                    <img src="${SERVER_URL}${model1Data.plots.confusion_matrix}?t=${Date.now()}" alt="Confusion Matrix for ${model1Name}">
                </div>
            </div>

            <div class="comparison-column">
                <h3>${model2Name}</h3>
                <div class="metrics-summary">
                    ${renderMetricBox("Accuracy", model2Data.accuracy, true)}
                    ${renderMetricBox("Precision", model2Data.classification_report['weighted avg'].precision)}
                    ${renderMetricBox("Recall", model2Data.classification_report['weighted avg'].recall)}
                    ${renderMetricBox("F1-Score", model2Data.classification_report['weighted avg']['f1-score'])}
                </div>
                <div class="card">
                    <h3>Classification Report</h3>
                    ${createReportTable(model2Data.classification_report)}
                </div>
                <div class="card">
                    <h3>Confusion Matrix</h3>
                    <img src="${SERVER_URL}${model2Data.plots.confusion_matrix}?t=${Date.now()}" alt="Confusion Matrix for ${model2Name}">
                </div>
            </div>
        </div>
    `;
}

function renderMetricBox(label, value, isPercent = false) {
    const displayValue = isPercent 
        ? (value * 100).toFixed(2) + '%' 
        : value.toFixed(3);
    
    return `
        <div class="metric-box">
            <div class="label">${label}</div>
            <div class="value">${displayValue}</div>
        </div>
    `;
}

function createReportTable(report) {
    let tableHTML = '<table class="report-table"><thead><tr><th>Class/Metric</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead><tbody>';
    for(const key in report) {
        if (typeof report[key] === 'object') {
            const row = report[key];
            tableHTML += `
                <tr>
                    <td>${key}</td>
                    <td>${row.precision ? row.precision.toFixed(3) : 'N/A'}</td>
                    <td>${row.recall ? row.recall.toFixed(3) : 'N/A'}</td>
                    <td>${row['f1-score'] ? row['f1-score'].toFixed(3) : 'N/A'}</td>
                    <td>${row.support ? row.support : ''}</td>
                </tr>
            `;
        } else { 
            tableHTML += `
                <tr>
                    <td>${key}</td>
                    <td colspan="3">${report[key].toFixed(3)}</td>
                    <td>${report['weighted avg'].support}</td>
                </tr>
            `;
        }
    }
    tableHTML += '</tbody></table>';
    return tableHTML;
}