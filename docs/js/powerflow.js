/**
 * Japan Power Grid Map - Power flow stub for static GitHub Pages.
 *
 * Power flow requires a local FastAPI server. This stub disables
 * the UI controls and shows an informational message.
 */

document.addEventListener("DOMContentLoaded", function () {
    // All power flow controls are disabled in index.html.
    // Show a message in the results section.
    var section = document.getElementById("pf-results-section");
    var content = document.getElementById("pf-results-content");
    if (section && content) {
        section.style.display = "block";
        content.innerHTML =
            '<div class="pf-info">' +
            "Power flow analysis is not available on the static site.<br>" +
            "To run power flow, start the local server:" +
            '<pre style="margin:8px 0;padding:8px;background:#16213e;border-radius:4px;font-size:0.75rem;color:#e94560;">' +
            "pip install -r requirements.txt\n" +
            "uvicorn src.server.app:app --reload" +
            "</pre>" +
            "Then open <a href='http://localhost:8000' style='color:#e94560;'>http://localhost:8000</a>" +
            "</div>";
    }
});
