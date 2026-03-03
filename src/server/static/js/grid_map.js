/**
 * Japan Power Grid Map - Leaflet initialization, tab control, and layer management.
 */

// Voltage color scheme (matches scripts/interactive_grid_maps.py)
const VOLTAGE_COLORS = {
    500: "#d62728",
    275: "#ff7f0e",
    220: "#e377c2",
    187: "#9467bd",
    154: "#2ca02c",
    132: "#17becf",
    110: "#1f77b4",
    77:  "#bcbd22",
    66:  "#8c564b",
    0:   "#cccccc",
};

// Loading percent color scale
const LOADING_COLORS = [
    [0,   "#2ecc71"],
    [30,  "#27ae60"],
    [50,  "#f39c12"],
    [70,  "#e67e22"],
    [85,  "#e74c3c"],
    [100, "#c0392b"],
];

function voltageKvToBracket(kv) {
    if (kv >= 500) return 500;
    if (kv >= 275) return 275;
    if (kv >= 220) return 220;
    if (kv >= 187) return 187;
    if (kv >= 154) return 154;
    if (kv >= 132) return 132;
    if (kv >= 110) return 110;
    if (kv >= 77) return 77;
    if (kv >= 66) return 66;
    return 0;
}

function voltageColor(kv) {
    return VOLTAGE_COLORS[voltageKvToBracket(kv)] || "#cccccc";
}

function loadingColor(pct) {
    if (pct == null || isNaN(pct)) return "#cccccc";
    let color = LOADING_COLORS[0][1];
    for (const [threshold, c] of LOADING_COLORS) {
        if (pct >= threshold) color = c;
    }
    return color;
}

function voltageLineWeight(kv) {
    if (kv >= 500) return 3;
    if (kv >= 275) return 2.5;
    if (kv >= 154) return 2;
    return 1.5;
}

// ── Map state ──

let map;
let substationLayer = null;
let lineLayer = null;
let pfBusLayer = null;
let pfLineLayer = null;
let currentRegion = null;
let regionsData = [];  // cached /api/regions response

function initMap() {
    map = L.map("map", {
        center: [36.5, 137.0],
        zoom: 6,
        zoomControl: true,
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        maxZoom: 19,
    }).addTo(map);

    addVoltageLegend();
}

function addVoltageLegend() {
    const legend = L.control({ position: "bottomright" });
    legend.onAdd = function () {
        const div = L.DomUtil.create("div", "legend");
        div.innerHTML = "<h4>Voltage Class</h4>";
        const voltages = [500, 275, 220, 187, 154, 132, 110, 77, 66];
        for (const kv of voltages) {
            const color = VOLTAGE_COLORS[kv];
            div.innerHTML +=
                `<div class="legend-item">` +
                `<span class="legend-color" style="background:${color}"></span>` +
                `${kv} kV</div>`;
        }
        return div;
    };
    legend.addTo(map);
}

// ── Tab switching ──

function initTabs() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", function () {
            const tabId = this.dataset.tab;
            // Deactivate all tabs
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
            // Activate selected
            this.classList.add("active");
            const panel = document.getElementById(tabId);
            if (panel) panel.classList.add("active");
        });
    });
}

// ── Data loading (Tab 1: Grid Map) ──

async function loadAllRegions(minKv) {
    if (minKv === undefined) {
        const sel = document.getElementById("min-kv");
        minKv = sel ? parseFloat(sel.value) : 275;
    }
    currentRegion = "all";
    setStatus("Loading all regions...");
    clearLayers();

    try {
        const [subRes, lineRes] = await Promise.all([
            fetch(`/api/geojson/all/substations?min_kv=${minKv}`),
            fetch(`/api/geojson/all/lines?min_kv=${minKv}`),
        ]);

        if (!subRes.ok || !lineRes.ok) {
            setStatus("Failed to load all regions");
            return;
        }

        const subData = await subRes.json();
        const lineData = await lineRes.json();

        lineLayer = L.geoJSON(lineData, {
            style: function (feature) {
                const kv = feature.properties._voltage_kv || 0;
                return {
                    color: voltageColor(kv),
                    weight: voltageLineWeight(kv),
                    opacity: 0.7,
                };
            },
            onEachFeature: function (feature, layer) {
                const p = feature.properties;
                const kv = p._voltage_kv ? `${p._voltage_kv} kV` : "Unknown";
                layer.bindPopup(
                    `<b>${p._display_name || "Unnamed"}</b><br>` +
                    `Voltage: ${kv}<br>` +
                    `Region: ${p._region_ja || ""}`
                );
            },
        }).addTo(map);

        substationLayer = L.geoJSON(subData, {
            pointToLayer: function (feature, latlng) {
                const kv = feature.properties._voltage_kv || 0;
                const bracket = voltageKvToBracket(kv);
                const radius = bracket >= 500 ? 5 : bracket >= 275 ? 4 : 3;
                return L.circleMarker(latlng, {
                    radius: radius,
                    fillColor: voltageColor(kv),
                    color: "#fff",
                    weight: 0.5,
                    fillOpacity: 0.8,
                });
            },
            onEachFeature: function (feature, layer) {
                const p = feature.properties;
                const kv = p._voltage_kv ? `${p._voltage_kv} kV` : "Unknown";
                layer.bindPopup(
                    `<b>${p._display_name || "Unnamed"}</b><br>` +
                    `Voltage: ${kv}<br>` +
                    `Region: ${p._region_ja || ""}`
                );
            },
        }).addTo(map);

        map.setView([36.5, 137.0], 6);
        const subCount = subData.features ? subData.features.length : 0;
        const lineCount = lineData.features ? lineData.features.length : 0;
        const kvLabel = minKv > 0 ? ` (${minKv} kV+)` : "";
        setStatus(`All regions${kvLabel}: ${subCount} substations, ${lineCount} lines`);
    } catch (err) {
        setStatus(`Error: ${err.message}`);
    }
}

async function loadRegion(region) {
    currentRegion = region;
    setStatus(`Loading ${region}...`);
    clearLayers();

    try {
        const [subRes, lineRes] = await Promise.all([
            fetch(`/api/geojson/${region}/substations`),
            fetch(`/api/geojson/${region}/lines`),
        ]);

        if (!subRes.ok || !lineRes.ok) {
            setStatus(`Failed to load ${region}`);
            return;
        }

        const subData = await subRes.json();
        const lineData = await lineRes.json();

        lineLayer = L.geoJSON(lineData, {
            style: function (feature) {
                const kv = feature.properties._voltage_kv || 0;
                return {
                    color: voltageColor(kv),
                    weight: voltageLineWeight(kv),
                    opacity: 0.8,
                };
            },
            onEachFeature: function (feature, layer) {
                const p = feature.properties;
                const kv = p._voltage_kv ? `${p._voltage_kv} kV` : "Unknown";
                const name = p._display_name || p.name || "Unnamed";
                layer.bindPopup(
                    `<b>${name}</b><br>` +
                    `Voltage: ${kv}<br>` +
                    `Region: ${p._region_ja || p._region || ""}<br>` +
                    `Cables: ${p.cables || "N/A"} | Circuits: ${p.circuits || "N/A"}`
                );
            },
        }).addTo(map);

        substationLayer = L.geoJSON(subData, {
            pointToLayer: function (feature, latlng) {
                const kv = feature.properties._voltage_kv || 0;
                const bracket = voltageKvToBracket(kv);
                const radius = bracket >= 500 ? 6 : bracket >= 275 ? 5 : bracket >= 154 ? 4 : 3;
                return L.circleMarker(latlng, {
                    radius: radius,
                    fillColor: voltageColor(kv),
                    color: "#fff",
                    weight: 1,
                    fillOpacity: 0.85,
                });
            },
            onEachFeature: function (feature, layer) {
                const p = feature.properties;
                const kv = p._voltage_kv ? `${p._voltage_kv} kV` : "Unknown";
                const name = p._display_name || p.name || "Unnamed";
                layer.bindPopup(
                    `<b>${name}</b><br>` +
                    `Voltage: ${kv}<br>` +
                    `Region: ${p._region_ja || p._region || ""}<br>` +
                    `Operator: ${p.operator || "N/A"}`
                );
            },
        }).addTo(map);

        const bounds = lineLayer.getBounds();
        if (bounds.isValid()) {
            map.fitBounds(bounds, { padding: [30, 30] });
        }

        const subCount = subData.features ? subData.features.length : 0;
        const lineCount = lineData.features ? lineData.features.length : 0;
        setStatus(`${region}: ${subCount} substations, ${lineCount} lines`);
    } catch (err) {
        setStatus(`Error: ${err.message}`);
    }
}

// ── Power flow results visualization ──

function showPowerFlowResults(busData, lineData) {
    if (pfBusLayer) { map.removeLayer(pfBusLayer); pfBusLayer = null; }
    if (pfLineLayer) { map.removeLayer(pfLineLayer); pfLineLayer = null; }

    if (lineData && lineData.features && lineData.features.length > 0) {
        pfLineLayer = L.geoJSON(lineData, {
            style: function (feature) {
                const loading = feature.properties.loading_percent || 0;
                return {
                    color: loadingColor(loading),
                    weight: 3,
                    opacity: 0.9,
                };
            },
            onEachFeature: function (feature, layer) {
                const p = feature.properties;
                layer.bindPopup(
                    `<b>${p.name || "Line"}</b><br>` +
                    `Loading: ${p.loading_percent}%<br>` +
                    `Voltage: ${p.voltage_kv} kV<br>` +
                    `P: ${p.p_from_mw} MW<br>` +
                    `Length: ${p.length_km} km`
                );
            },
        }).addTo(map);
    }

    if (busData && busData.features && busData.features.length > 0) {
        pfBusLayer = L.geoJSON(busData, {
            pointToLayer: function (feature, latlng) {
                const vm = feature.properties.vm_pu || 1.0;
                let color;
                if (vm >= 0.97 && vm <= 1.03) color = "#2ecc71";
                else if (vm >= 0.95 && vm <= 1.05) color = "#f39c12";
                else color = "#e74c3c";
                return L.circleMarker(latlng, {
                    radius: 5,
                    fillColor: color,
                    color: "#fff",
                    weight: 1,
                    fillOpacity: 0.9,
                });
            },
            onEachFeature: function (feature, layer) {
                const p = feature.properties;
                layer.bindPopup(
                    `<b>${p.name || "Bus"}</b><br>` +
                    `V: ${p.vm_pu} pu<br>` +
                    `P: ${p.p_mw} MW<br>` +
                    `Rated: ${p.voltage_kv} kV`
                );
            },
        }).addTo(map);
    }
}

function clearPowerFlowLayers() {
    if (pfBusLayer) { map.removeLayer(pfBusLayer); pfBusLayer = null; }
    if (pfLineLayer) { map.removeLayer(pfLineLayer); pfLineLayer = null; }
}

function clearLayers() {
    if (substationLayer) { map.removeLayer(substationLayer); substationLayer = null; }
    if (lineLayer) { map.removeLayer(lineLayer); lineLayer = null; }
    clearPowerFlowLayers();
}

// ── Region list (Tab 1) ──

async function initRegionList() {
    try {
        const res = await fetch("/api/regions");
        if (!res.ok) return;
        regionsData = await res.json();

        const list = document.getElementById("region-list");
        if (!list) return;

        // "All Regions" button
        const allBtn = document.createElement("button");
        allBtn.className = "region-btn active";
        allBtn.innerHTML = `<span>All Regions (全国)</span><span class="count"></span>`;
        allBtn.onclick = function () {
            setActiveRegionBtn(null);
            loadAllRegions();
        };
        list.appendChild(allBtn);

        // Individual regions
        for (const r of regionsData) {
            const btn = document.createElement("button");
            btn.className = "region-btn";
            btn.dataset.region = r.id;
            btn.innerHTML =
                `<span>${r.name_en} (${r.name_ja})</span>` +
                `<span class="count">${r.substations + r.lines}</span>`;
            btn.onclick = function () {
                setActiveRegionBtn(r.id);
                loadRegion(r.id);
            };
            list.appendChild(btn);
        }
    } catch (err) {
        console.error("Failed to load regions:", err);
    }
}

function setActiveRegionBtn(region) {
    document.querySelectorAll("#region-list .region-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.region === region);
    });
    if (!region) {
        const first = document.querySelector("#region-list .region-btn:first-child");
        if (first) first.classList.add("active");
    }
}

// ── Status ──

function setStatus(msg) {
    const el = document.getElementById("status-text");
    if (el) el.textContent = msg;
}

// ── Init ──

document.addEventListener("DOMContentLoaded", function () {
    initMap();
    initTabs();
    initRegionList();

    // Bind voltage filter
    const minKvSelect = document.getElementById("min-kv");
    if (minKvSelect) {
        minKvSelect.addEventListener("change", function () {
            if (currentRegion === "all") {
                loadAllRegions(parseFloat(this.value));
            }
        });
    }

    // Load all regions (275 kV+) by default
    loadAllRegions(275);
});
