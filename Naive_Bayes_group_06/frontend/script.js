document.addEventListener("DOMContentLoaded", () => {
  // --- Element References ---
  const fileInput = document.getElementById("file-input");
  const fileLabelText = document.getElementById("file-label-text");
  const analyzeBtn = document.getElementById("analyze-btn");
  const loader = document.getElementById("loader");
  const appNav = document.getElementById("app-nav");
  const uploadPage = document.getElementById("upload-page");
  const resultsContainer = document.getElementById("results-container");
  const errorMessageDiv = document.getElementById("error-message");
  const preloadedSelect = document.getElementById("preloaded-select"); // NEW

  // const SERVER_URL = "http://localhost:5000";
  const SERVER_URL = window.location.origin;

  //const SERVER_URL = window.location.origin;
  let analysisData = null;
  let accuracyChartInstance;

  // --- Event Listeners ---
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      fileLabelText.textContent = fileInput.files[0].name;
      preloadedSelect.value = ""; // Clear dropdown
    } else {
      fileLabelText.textContent = "Click to select a .CSV dataset";
    }
    checkAnalysisButtonState(); // Use new function
  });

  preloadedSelect.addEventListener("change", () => {
    if (preloadedSelect.value) {
      // Clear file input
      fileInput.value = null;
      fileLabelText.textContent = "Click to select a .CSV dataset";
    }
    checkAnalysisButtonState(); // Use new function
  });

  analyzeBtn.addEventListener("click", handleAnalysis);

  // --- Core Functions ---
  function checkAnalysisButtonState() {
    if (fileInput.files.length > 0 || preloadedSelect.value) {
      analyzeBtn.disabled = false;
    } else {
      analyzeBtn.disabled = true;
    }
  }

  async function handleAnalysis() {
    setUIState("loading");

    let fetchUrl;
    let fetchOptions;

    // Decide which analysis to run
    if (fileInput.files.length > 0) {
      // 1. UPLOADED FILE logic
      console.log("Running analysis on uploaded file...");
      const formData = new FormData();
      formData.append("dataset", fileInput.files[0]);

      fetchUrl = `${SERVER_URL}/analyze`;
      fetchOptions = { method: "POST", body: formData };
    } else if (preloadedSelect.value) {
      // 2. PRELOADED FILE logic
      console.log(
        `Running analysis on pre-loaded file: ${preloadedSelect.value}`
      );
      fetchUrl = `${SERVER_URL}/analyze-preloaded`; // NEW ENDPOINT
      fetchOptions = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ datasetName: preloadedSelect.value }),
      };
    } else {
      showError("Please select a dataset file first.");
      setUIState("initial");
      return;
    }

    try {
      // This part is now generic and works for both requests
      const response = await fetch(fetchUrl, fetchOptions);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.details ||
            errorData.error ||
            `Server responded with status: ${response.status}`
        );
      }

      analysisData = await response.json();
      localStorage.setItem("analysisResults", JSON.stringify(analysisData));
      setupUIForResults();
      renderAllPages();
      showPage("dashboard");
    } catch (error) {
      console.error("Analysis failed:", error);
      let userMessage = "An unexpected error occurred.";
      if (error.message.includes("Failed to fetch")) {
        userMessage =
          "Could not connect to the server. Please ensure the backend server is running.";
      } else {
        userMessage = `Analysis Failed: ${error.message}`;
      }
      showError(userMessage);
      setUIState("initial");
    }
  }

  function setUIState(state) {
    loader.classList.add("hidden");
    errorMessageDiv.classList.add("hidden");
    analyzeBtn.disabled = false;
    fileInput.disabled = false;

    if (state === "loading") {
      loader.classList.remove("hidden");
      analyzeBtn.disabled = true;
      fileInput.disabled = true;
    } else if (state === "results") {
      uploadPage.classList.add("hidden");
      appNav.classList.remove("hidden");
    } else {
      uploadPage.classList.remove("hidden");
      appNav.classList.add("hidden");
    }
  }

  function setupUIForResults() {
    resultsContainer.innerHTML = "";
    const oldNavLinks = appNav.querySelectorAll(".model-link");
    oldNavLinks.forEach((link) => link.remove());

    const dashboardPage = document.createElement("section");
    dashboardPage.id = "dashboard-page";
    dashboardPage.className = "page";
    resultsContainer.appendChild(dashboardPage);
    // createComparisonPage("Naive Bayes", "SVC");
    // createComparisonPage("Naive Bayes", "Decision Tree");
    const dashboardLink = appNav.querySelector('a[href="#dashboard"]');
    const newDashboardLink = dashboardLink.cloneNode(true);
    dashboardLink.parentNode.replaceChild(newDashboardLink, dashboardLink);
    newDashboardLink.addEventListener("click", (e) => {
      e.preventDefault();
      showPage("dashboard");
    });

    Object.keys(analysisData.models).forEach((modelName) => {
      const link = document.createElement("a");
      const pageId = modelName.replace(/\s+/g, "-");
      link.href = `#${pageId}`;
      link.textContent = modelName;
      link.className = "nav-link model-link";
      appNav.appendChild(link);
      link.addEventListener("click", (e) => {
        e.preventDefault();
        showPage(pageId);
      });
      const modelPage = document.createElement("section");
      modelPage.id = `${pageId}-page`;
      modelPage.className = "page hidden";
      resultsContainer.appendChild(modelPage);
    });

    setUIState("results");
  }

  // --- NEW FUNCTION to create comparison pages ---
  // function createComparisonPage(model1Name, model2Name) {
  //     // Create ID, e.g., "nb-vs-svc"
  //     const id1 = model1Name.split(' ').map(s => s[0]).join('').toLowerCase();
  //     const id2 = model2Name.split(' ').map(s => s[0]).join('').toLowerCase();
  //     const pageId = `${id1}-vs-${id2}`; // e.g., nb-vs-svc
  //     const pageTitle = `${model1Name} vs ${model2Name}`;

  //     // 1. Create the page section
  //     const page = document.createElement('section');
  //     page.id = `${pageId}-page`;
  //     page.className = 'page hidden';
  //     resultsContainer.appendChild(page);

  //     // 2. Add event listener to the link
  //     const link = appNav.querySelector(`a[href="#${pageId}"]`);
  //     if (link) {
  //         link.addEventListener('click', (e) => {
  //             e.preventDefault();
  //             showPage(pageId);
  //         });
  //     }
  // }

  // --- NEW FUNCTION to render a comparison page ---
  // function renderComparisonPage(model1Name, model2Name) {
  //     const model1Data = analysisData.models[model1Name];
  //     const model2Data = analysisData.models[model2Name];

  //     // Find page ID
  //     const id1 = model1Name.split(' ').map(s => s[0]).join('').toLowerCase();
  //     const id2 = model2Name.split(' ').map(s => s[0]).join('').toLowerCase();
  //     const pageId = `${id1}-vs-${id2}`;
  //     const pageTitle = `${model1Name} vs ${model2Name}`;

  //     const page = document.getElementById(`${pageId}-page`);
  //     if (!page) return; // Don't do anything if page wasn't created

  //     // Fill page with HTML
  //     page.innerHTML = `
  //         <div class="page-header"><h2>${pageTitle} Comparison</h2></div>
  //         <div class="comparison-grid">

  //             <!-- === MODEL 1 COLUMN === -->
  //             <div class="comparison-column">
  //                 <h3>${model1Name}</h3>
  //                 <div class="metrics-summary">
  //                     ${renderMetricBox("Accuracy", model1Data.accuracy, true)}
  //                     ${renderMetricBox("Precision", model1Data.classification_report['weighted avg'].precision)}
  //                     ${renderMetricBox("Recall", model1Data.classification_report['weighted avg'].recall)}
  //                     ${renderMetricBox("F1-Score", model1Data.classification_report['weighted avg']['f1-score'])}
  //                 </div>
  //                 <div class="card">
  //                     <h3>Classification Report</h3>
  //                     ${createReportTable(model1Data.classification_report)}
  //                 </div>
  //                 <div class="card">
  //                     <h3>Confusion Matrix</h3>
  //                     <img src="${SERVER_URL}${model1Data.plots.confusion_matrix}?t=${Date.now()}" alt="Confusion Matrix for ${model1Name}">
  //                 </div>
  //             </div>

  //             <!-- === MODEL 2 COLUMN === -->
  //             <div class="comparison-column">
  //                 <h3>${model2Name}</h3>
  //                 <div class="metrics-summary">
  //                     ${renderMetricBox("Accuracy", model2Data.accuracy, true)}
  //                     ${renderMetricBox("Precision", model2Data.classification_report['weighted avg'].precision)}
  //                     ${renderMetricBox("Recall", model2Data.classification_report['weighted avg'].recall)}
  //                     ${renderMetricBox("F1-Score", model2Data.classification_report['weighted avg']['f1-score'])}
  //                 </div>
  //                 <div class="card">
  //                     <h3>Classification Report</h3>
  //                     ${createReportTable(model2Data.classification_report)}
  //                 </div>
  //                 <div class="card">
  //                     <h3>Confusion Matrix</h3>
  //                     <img src="${SERVER_URL}${model2Data.plots.confusion_matrix}?t=${Date.now()}" alt="Confusion Matrix for ${model2Name}">
  //                 </div>
  //             </div>
  //         </div>
  //     `;
  // }

  // --- NEW HELPER FUNCTION to render a single metric box ---
  // function renderMetricBox(label, value, isPercent = false) {
  //     const displayValue = isPercent
  //         ? (value * 100).toFixed(2) + '%'
  //         : value.toFixed(3);

  //     return `
  //         <div class="metric-box">
  //             <div class="label">${label}</div>
  //             <div class="value">${displayValue}</div>
  //         </div>
  //     `;
  // }

  function renderAllPages() {
    renderDashboardPage();
    Object.entries(analysisData.models).forEach(([modelName, modelData]) => {
      renderDetailedModelPage(modelName, modelData);
    });
    // renderComparisonPage("Naive Bayes", "SVC");
    // renderComparisonPage("Naive Bayes", "Decision Tree");
  }

  // ... in script.js

  function renderDashboardPage() {
    const page = document.getElementById("dashboard-page");

    // --- (FIX) Check if the ROC plot exists in the JSON ---
    const rocPlotHtml = analysisData.comparison_plots.roc_curves
      ? `
            <div class="card">
                <h3>ROC Curve Comparison</h3>
                <img src="${SERVER_URL}${
          analysisData.comparison_plots.roc_curves
        }?t=${Date.now()}" alt="ROC Curve Comparison">
            </div>
            `
      : `
            <div classG="card">
                <h3>ROC Curve Comparison</h3>
                <p>ROC curves are only generated for binary classification problems (not multiclass).</p>
            </div>
            `;

    // --- (MODIFIED) Use the new rocPlotHtml variable ---
    page.innerHTML = `
            <div class="page-header"><h2>Comparison Dashboard</h2></div>
            <div class="grid-container">
                <div class="card">
                    <h3>Model Accuracy</h3>
                    <div class="chart-container">
                        <canvas id="accuracy-chart"></canvas>
                    </div>
                </div>
                ${rocPlotHtml}
            </div>
        `;
    renderAccuracyChart(analysisData.accuracies);
  }

  // ... (rest of script.js)

  function renderDetailedModelPage(name, data) {
    const pageId = name.replace(/\s+/g, "-");
    const page = document.getElementById(`${pageId}-page`);
    const report = data.classification_report;
    const weightedAvg = report["weighted avg"];

    page.innerHTML = `
            <div class="page-header"><h2>${name} - Detailed Report</h2></div>
            <div class="metrics-summary">
                <div class="metric-box"><div class="label">Accuracy</div><div class="value">${(
                  data.accuracy * 100
                ).toFixed(2)}%</div></div>
                <div class="metric-box"><div class="label">Precision</div><div class="value">${weightedAvg.precision.toFixed(
                  3
                )}</div></div>
                <div class="metric-box"><div class="label">Recall</div><div class="value">${weightedAvg.recall.toFixed(
                  3
                )}</div></div>
                <div class="metric-box"><div class="label">F1-Score</div><div class="value">${weightedAvg[
                  "f1-score"
                ].toFixed(3)}</div></div>
            </div>
            <div class="grid-container">
                <div class="card">
                    <h3>Classification Report</h3>
                    ${createReportTable(report)}
                </div>
                <div class="card">
                    <h3>Confusion Matrix</h3>
                    <img src="${SERVER_URL}${
      data.plots.confusion_matrix
    }?t=${Date.now()}" alt="Confusion Matrix for ${name}">
                </div>
            </div>
        `;
  }

  // Replace your complex showPage with this original, simple one
  function showPage(pageId) {
    document
      .querySelectorAll(".page")
      .forEach((p) => p.classList.add("hidden"));

    // This is the page ID, e.g., "dashboard-page" or "Naive-Bayes-page"
    const targetPageId = pageId.endsWith("-page") ? pageId : `${pageId}-page`;

    const targetPage = document.getElementById(targetPageId);
    if (targetPage) {
      targetPage.classList.remove("hidden");
    }

    // This is the link href, e.g., "#dashboard" or "#Naive Bayes"
    const linkHref = pageId.endsWith("-page")
      ? pageId.replace("-page", "")
      : pageId;

    document
      .querySelectorAll(".nav-link")
      .forEach((link) => link.classList.remove("active"));
    const activeLink = document.querySelector(`.nav-link[href="#${linkHref}"]`);
    if (activeLink) {
      activeLink.classList.add("active");
    }
  }

  function createReportTable(report) {
    let tableHTML =
      '<table class="report-table"><thead><tr><th>Metric</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead><tbody>';
    for (const key in report) {
      if (typeof report[key] === "object") {
        const row = report[key];
        tableHTML += `
                    <tr>
                        <td>${key}</td>
                        <td>${
                          row.precision ? row.precision.toFixed(3) : "N/A"
                        }</td>
                        <td>${row.recall ? row.recall.toFixed(3) : "N/A"}</td>
                        <td>${
                          row["f1-score"] ? row["f1-score"].toFixed(3) : "N/A"
                        }</td>
                        <td>${row.support}</td>
                    </tr>
                `;
      }
    }
    tableHTML += "</tbody></table>";
    return tableHTML;
  }

  function renderAccuracyChart(accuracies) {
    if (accuracyChartInstance) accuracyChartInstance.destroy();
    const ctx = document.getElementById("accuracy-chart").getContext("2d");
    accuracyChartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: Object.keys(accuracies),
        datasets: [
          {
            label: "Accuracy",
            data: Object.values(accuracies),
            backgroundColor: "rgba(0, 123, 255, 0.6)",
            borderColor: "rgba(0, 123, 255, 1)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, max: 1.0 } },
        plugins: { legend: { display: false } },
      },
    });
  }

  function showError(message) {
    errorMessageDiv.textContent = message;
    errorMessageDiv.classList.remove("hidden");
  }
});
