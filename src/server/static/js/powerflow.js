/**
 * Japan Power Grid Map - Power flow UI and MATPOWER export (Tab 2).
 */

let isRunning = false;

// ── Power Flow Region Dropdown (Tab 2) ──

async function initPfRegionDropdown() {
    // Wait for regionsData from grid_map.js, or fetch independently
    let regions = regionsData;
    if (!regions || regions.length === 0) {
        try {
            const res = await fetch("/api/regions");
            if (res.ok) regions = await res.json();
        } catch (e) {
            console.error("Failed to load regions for PF dropdown:", e);
            return;
        }
    }

    const sel = document.getElementById("pf-region");
    if (!sel || !regions) return;

    for (const r of regions) {
        const opt = document.createElement("option");
        opt.value = r.id;
        opt.textContent = `${r.name_en} (${r.name_ja})`;
        sel.appendChild(opt);
    }
}

function getSelectedPfRegion() {
    const sel = document.getElementById("pf-region");
    return sel ? sel.value : null;
}

// ── Run Power Flow ──

async function runPowerFlow() {
    if (isRunning) return;

    const region = getSelectedPfRegion();
    if (!region) {
        alert("Select a region first.");
        return;
    }

    const mode = document.getElementById("pf-mode").value;
    const loadFactorInput = document.getElementById("pf-load-factor").value;
    const loadFactor = loadFactorInput ? parseFloat(loadFactorInput) : null;

    isRunning = true;
    const btn = document.getElementById("btn-run-pf");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Running...';
    setStatus(`Running ${mode.toUpperCase()} power flow on ${region}...`);
    showResultsSection(false);
    clearResults();

    try {
        const res = await fetch("/api/powerflow/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ region, mode, load_factor: loadFactor }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Power flow failed");
        }

        const summary = await res.json();
        displayResults(summary);
        showResultsSection(true);

        // Load result GeoJSON and show on map
        const [busRes, lineRes] = await Promise.all([
            fetch(`/api/powerflow/results/${region}/buses`),
            fetch(`/api/powerflow/results/${region}/lines`),
        ]);

        let busData = null, lineData = null;
        if (busRes.ok) busData = await busRes.json();
        if (lineRes.ok) lineData = await lineRes.json();

        // Also load the region on the map so PF results overlay correctly
        await loadRegion(region);
        showPowerFlowResults(busData, lineData);

        const status = summary.converged
            ? `Power flow converged (${summary.mode.toUpperCase()})`
            : `Power flow did NOT converge`;
        setStatus(status);

    } catch (err) {
        setStatus(`Error: ${err.message}`);
        displayError(err.message);
        showResultsSection(true);
    } finally {
        isRunning = false;
        btn.disabled = false;
        btn.textContent = "Run Power Flow";
    }
}

// ── Display Results ──

function showResultsSection(show) {
    const section = document.getElementById("pf-results-section");
    if (section) section.style.display = show ? "block" : "none";
}

function displayResults(summary) {
    const container = document.getElementById("pf-results-content");
    if (!container) return;

    const convergedClass = summary.converged ? "success" : "fail";
    const convergedText = summary.converged ? "Yes" : "No";

    container.innerHTML = `
        <div class="result-grid">
            <div class="result-item">
                <div class="label">Converged</div>
                <div class="value ${convergedClass}">${convergedText}</div>
            </div>
            <div class="result-item">
                <div class="label">Mode</div>
                <div class="value">${summary.mode.toUpperCase()}</div>
            </div>
            <div class="result-item">
                <div class="label">Total Load</div>
                <div class="value">${summary.total_load_mw} MW</div>
            </div>
            <div class="result-item">
                <div class="label">Losses</div>
                <div class="value">${summary.total_loss_mw} MW</div>
            </div>
            <div class="result-item">
                <div class="label">Max Loading</div>
                <div class="value">${summary.max_line_loading_pct}%</div>
            </div>
            <div class="result-item">
                <div class="label">Active Buses</div>
                <div class="value">${summary.buses}</div>
            </div>
            <div class="result-item">
                <div class="label">Active Lines</div>
                <div class="value">${summary.lines}</div>
            </div>
        </div>
        ${summary.warnings && summary.warnings.length > 0 ? `
            <div style="margin-top:8px;font-size:0.7rem;color:#e67e22;">
                ${summary.warnings.length} warning(s)
            </div>
        ` : ""}
    `;
}

function displayError(msg) {
    const container = document.getElementById("pf-results-content");
    if (!container) return;
    container.innerHTML = `<div style="color:#e74c3c;font-size:0.85rem;padding:8px;">${msg}</div>`;
}

function clearResults() {
    const container = document.getElementById("pf-results-content");
    if (container) container.innerHTML = "";
}

// ── MATPOWER Export ──

async function exportMatpower() {
    const region = getSelectedPfRegion();
    if (!region) {
        alert("Select a region first.");
        return;
    }

    setStatus(`Exporting MATPOWER for ${region}...`);

    try {
        const res = await fetch("/api/matpower/export", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ region }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Export failed");
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${region}.mat`;
        a.click();
        URL.revokeObjectURL(url);

        setStatus(`MATPOWER export complete: ${region}.mat`);
    } catch (err) {
        setStatus(`Export error: ${err.message}`);
    }
}

// ── Init (called from DOMContentLoaded in grid_map.js) ──

document.addEventListener("DOMContentLoaded", function () {
    initPfRegionDropdown();

    const btnRun = document.getElementById("btn-run-pf");
    if (btnRun) btnRun.addEventListener("click", runPowerFlow);

    const btnExport = document.getElementById("btn-export-mat");
    if (btnExport) btnExport.addEventListener("click", exportMatpower);
});
