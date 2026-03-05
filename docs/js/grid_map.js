/**
 * Japan Power Grid Map - Static GitHub Pages version.
 *
 * Features:
 * - Layer visibility checkboxes (lines, substations, plants)
 * - Region filtering: click a region to show only its features
 * - Voltage tier files (275kv, 154kv, all) + client-side filter
 * - Plant category filter (utility / +IPP / all)
 */

// Voltage color scheme
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

function voltageLineWeight(kv) {
    if (kv >= 500) return 3;
    if (kv >= 275) return 2.5;
    if (kv >= 154) return 2;
    return 1.5;
}

// ── Voltage tier file mapping ──

function voltageTier(minKv) {
    if (minKv >= 500) return { suffix: "275kv", clientFilter: 500 };
    if (minKv >= 275) return { suffix: "275kv", clientFilter: 0 };
    if (minKv >= 154) return { suffix: "154kv", clientFilter: 0 };
    if (minKv >= 110) return { suffix: "all",   clientFilter: 110 };
    if (minKv >= 66)  return { suffix: "all",   clientFilter: 66 };
    return                    { suffix: "all",   clientFilter: 0 };
}

// Fuel type colors for power plants
const FUEL_COLORS = {
    nuclear: "#ff0000", coal: "#444444", gas: "#ff8800",
    oil: "#884400", hydro: "#0088ff", pumped_hydro: "#0044aa",
    wind: "#00cc88", solar: "#ffdd00", geothermal: "#cc4488",
    biomass: "#668833", waste: "#996633", tidal: "#006688",
    battery: "#aa00ff", unknown: "#999999",
};

// ── Map state ──

var map;
var substationLayer = null;
var lineLayer = null;
var plantLayer = null;
var regionsData = [];

// Raw cached GeoJSON (before region filter)
var rawSubData = null;
var rawLineData = null;
var rawPlantData = null;

// Enriched data caches
var enrichedSubData = null;    // substations.geojson
var enrichedGenData = null;    // generators.geojson

// Current state
var selectedRegion = "all";  // "all" or region id
var plantFilter = "utility";
var layerVisible = { lines: true, subs: true, plants: true };

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
    var legend = L.control({ position: "bottomright" });
    legend.onAdd = function () {
        var div = L.DomUtil.create("div", "legend");
        div.innerHTML = "<h4>Voltage Class</h4>";
        var voltages = [500, 275, 220, 187, 154, 132, 110, 77, 66];
        for (var i = 0; i < voltages.length; i++) {
            var kv = voltages[i];
            div.innerHTML +=
                '<div class="legend-item">' +
                '<span class="legend-color" style="background:' + VOLTAGE_COLORS[kv] + '"></span>' +
                kv + ' kV</div>';
        }
        div.innerHTML += "<h4 style='margin-top:8px'>Power Plants</h4>";
        var fuelEntries = [
            ["nuclear","Nuclear"],["coal","Coal"],["gas","Gas"],["oil","Oil"],
            ["hydro","Hydro"],["wind","Wind"],["solar","Solar"],
            ["geothermal","Geothermal"],["biomass","Biomass"]
        ];
        for (var j = 0; j < fuelEntries.length; j++) {
            var key = fuelEntries[j][0], label = fuelEntries[j][1];
            div.innerHTML +=
                '<div class="legend-item">' +
                '<span class="legend-color" style="background:' + FUEL_COLORS[key] + ';border-radius:50%"></span>' +
                label + '</div>';
        }
        return div;
    };
    legend.addTo(map);
}

// ── Tab switching ──

function initTabs() {
    document.querySelectorAll(".tab-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
            var tabId = this.dataset.tab;
            document.querySelectorAll(".tab-btn").forEach(function (b) { b.classList.remove("active"); });
            document.querySelectorAll(".tab-panel").forEach(function (p) { p.classList.remove("active"); });
            this.classList.add("active");
            var panel = document.getElementById(tabId);
            if (panel) panel.classList.add("active");
        });
    });
}

// ── Filtering helpers ──

function filterByVoltage(geojson, minKv) {
    if (!minKv || minKv <= 0) return geojson;
    var filtered = geojson.features.filter(function (f) {
        var kv = f.properties._voltage_kv;
        return kv != null && kv >= minKv;
    });
    return { type: "FeatureCollection", features: filtered };
}

function filterByRegion(geojson, region) {
    if (!region || region === "all") return geojson;
    var filtered = geojson.features.filter(function (f) {
        return f.properties._region === region;
    });
    return { type: "FeatureCollection", features: filtered };
}

// ── Layer rendering ──

function renderLayers() {
    clearLayers();

    if (!rawSubData || !rawLineData) return;

    // Filter by region
    var subData = filterByRegion(rawSubData, selectedRegion);
    var lineData = filterByRegion(rawLineData, selectedRegion);

    // Lines
    if (layerVisible.lines) {
        lineLayer = L.geoJSON(lineData, {
            style: function (feature) {
                var kv = feature.properties._voltage_kv || 0;
                return {
                    color: voltageColor(kv),
                    weight: voltageLineWeight(kv),
                    opacity: 0.7,
                };
            },
            onEachFeature: function (feature, layer) {
                var p = feature.properties;
                var kv = p._voltage_kv ? p._voltage_kv + " kV" : "Unknown";
                layer.bindPopup(
                    "<b>" + (p._display_name || "Unnamed") + "</b><br>" +
                    "Voltage: " + kv + "<br>" +
                    "Region: " + (p._region_ja || "")
                );
            },
        }).addTo(map);
    }

    // Substations
    if (layerVisible.subs) {
        substationLayer = L.geoJSON(subData, {
            pointToLayer: function (feature, latlng) {
                var kv = feature.properties._voltage_kv || 0;
                var bracket = voltageKvToBracket(kv);
                var radius = bracket >= 500 ? 5 : bracket >= 275 ? 4 : 3;
                return L.circleMarker(latlng, {
                    radius: radius,
                    fillColor: voltageColor(kv),
                    color: "#fff",
                    weight: 0.5,
                    fillOpacity: 0.8,
                });
            },
            onEachFeature: function (feature, layer) {
                var p = feature.properties;
                var coords = feature.geometry.coordinates;
                var enriched = (coords && coords.length >= 2) ? lookupEnrichedSub(coords[0], coords[1]) : null;
                if (enriched) {
                    layer.bindPopup(buildSubPopup(enriched), { maxWidth: 350 });
                } else {
                    var kv = p._voltage_kv ? p._voltage_kv + " kV" : "Unknown";
                    layer.bindPopup(
                        "<b>" + (p._display_name || "Unnamed") + "</b><br>" +
                        "Voltage: " + kv + "<br>" +
                        "Region: " + (p._region_ja || "")
                    );
                }
            },
        }).addTo(map);
    }

    // Plants
    if (layerVisible.plants && rawPlantData) {
        var plantData = filterByRegion(rawPlantData, selectedRegion);
        plantLayer = L.geoJSON(plantData, {
            pointToLayer: function (feature, latlng) {
                var p = feature.properties;
                var fuel = p.fuel_type || "unknown";
                var color = FUEL_COLORS[fuel] || FUEL_COLORS.unknown;
                var mw = p.capacity_mw || 0;
                var radius = mw >= 1000 ? 7 : mw >= 100 ? 5 : mw > 0 ? 4 : 3;
                return L.circleMarker(latlng, {
                    radius: radius,
                    fillColor: color,
                    color: "#000",
                    weight: 1,
                    fillOpacity: 0.85,
                });
            },
            onEachFeature: function (feature, layer) {
                var p = feature.properties;
                var coords = feature.geometry.coordinates;
                var enriched = (coords && coords.length >= 2) ? lookupEnrichedGen(coords[0], coords[1]) : null;
                if (enriched) {
                    layer.bindPopup(buildGenPopup(enriched), { maxWidth: 350 });
                } else {
                    var cap = p.capacity_mw ? p.capacity_mw + " MW" : "N/A";
                    layer.bindPopup(
                        "<b>" + (p._display_name || "Unnamed") + "</b><br>" +
                        "Fuel: " + (p.fuel_type || "unknown") + "<br>" +
                        "Capacity: " + cap + "<br>" +
                        "Region: " + (p._region_ja || "")
                    );
                }
            },
        }).addTo(map);
    }

    updateStatus();
}

function updateStatus() {
    var subCount = substationLayer ? substationLayer.getLayers().length : 0;
    var lineCount = lineLayer ? lineLayer.getLayers().length : 0;
    var plantCount = plantLayer ? plantLayer.getLayers().length : 0;

    var regionLabel = "All regions";
    if (selectedRegion !== "all") {
        var r = regionsData.find(function (rd) { return rd.id === selectedRegion; });
        if (r) regionLabel = r.name_en + " (" + r.name_ja + ")";
    }

    var minKv = parseFloat((document.getElementById("min-kv") || {}).value || 275);
    var kvLabel = minKv > 0 ? " [" + minKv + " kV+]" : "";

    setStatus(regionLabel + kvLabel + ": " + subCount + " subs, " + lineCount + " lines, " + plantCount + " plants");
}

// ── Data loading ──

async function loadData(minKv) {
    if (minKv === undefined) {
        var sel = document.getElementById("min-kv");
        minKv = sel ? parseFloat(sel.value) : 275;
    }

    setStatus("Loading...");
    clearLayers();

    var tier = voltageTier(minKv);
    var linesFile = "./data/lines_" + tier.suffix + ".geojson";
    var subsFile  = "./data/subs_"  + tier.suffix + ".geojson";

    try {
        var responses = await Promise.all([
            fetch(subsFile),
            fetch(linesFile),
        ]);

        if (!responses[0].ok || !responses[1].ok) {
            setStatus("Failed to load data");
            return;
        }

        rawSubData  = await responses[0].json();
        rawLineData = await responses[1].json();

        // Apply client-side voltage filter if needed
        if (tier.clientFilter > 0) {
            rawSubData  = filterByVoltage(rawSubData, tier.clientFilter);
            rawLineData = filterByVoltage(rawLineData, tier.clientFilter);
        }

        // Load plants
        await loadPlantData();

        renderLayers();

        // Zoom
        if (selectedRegion === "all") {
            map.setView([36.5, 137.0], 6);
        } else {
            zoomToRegion(selectedRegion);
        }

    } catch (err) {
        setStatus("Error: " + err.message);
    }
}

async function loadPlantData() {
    rawPlantData = null;

    try {
        var plantData;
        if (plantFilter === "utility") {
            var res = await fetch("./data/plants_utility.geojson");
            if (res.ok) plantData = await res.json();
        } else if (plantFilter === "ipp") {
            var responses = await Promise.all([
                fetch("./data/plants_utility.geojson"),
                fetch("./data/plants_ipp.geojson"),
            ]);
            if (responses[0].ok && responses[1].ok) {
                var d1 = await responses[0].json();
                var d2 = await responses[1].json();
                plantData = {
                    type: "FeatureCollection",
                    features: d1.features.concat(d2.features),
                };
            }
        } else {
            var res2 = await fetch("./data/plants_all.geojson");
            if (res2.ok) plantData = await res2.json();
        }
        rawPlantData = plantData || null;
    } catch (err) {
        console.error("Failed to load plants:", err);
    }
}

function zoomToRegion(region) {
    var r = regionsData.find(function (rd) { return rd.id === region; });
    if (!r || !r.bounding_box) return;
    var bb = r.bounding_box;
    map.fitBounds([
        [bb.lat_min, bb.lon_min],
        [bb.lat_max, bb.lon_max],
    ], { padding: [30, 30] });
}

function selectRegion(region) {
    selectedRegion = region || "all";
    setActiveRegionBtn(region);
    renderLayers();

    if (selectedRegion === "all") {
        map.setView([36.5, 137.0], 6);
    } else {
        zoomToRegion(selectedRegion);
    }
}

function clearLayers() {
    if (substationLayer) { map.removeLayer(substationLayer); substationLayer = null; }
    if (lineLayer) { map.removeLayer(lineLayer); lineLayer = null; }
    if (plantLayer) { map.removeLayer(plantLayer); plantLayer = null; }
}

// ── Enriched data loaders ──

async function loadEnrichedData() {
    try {
        var res = await fetch("./data/substations.geojson");
        if (res.ok) enrichedSubData = await res.json();
    } catch (e) { console.warn("No enriched substations:", e); }
    try {
        var res2 = await fetch("./data/generators.geojson");
        if (res2.ok) enrichedGenData = await res2.json();
    } catch (e) { console.warn("No enriched generators:", e); }
}

// ── Detail popup builders ──

function fmtNum(n) {
    if (n == null) return "-";
    if (typeof n === "number" && n >= 1000) return n.toLocaleString();
    return String(n);
}

function buildSubPopup(p) {
    var html = '<div class="detail-popup">';
    html += '<h3>' + (p.name || p._display_name || "(unnamed)") + '</h3>';
    var meta = [];
    if (p.operator) meta.push(p.operator);
    if (p.region_ja) meta.push(p.region_ja);
    if (meta.length) html += '<div class="popup-meta">' + meta.join(" | ") + '</div>';

    // Electrical
    html += '<div class="popup-section"><div class="popup-section-title">Electrical</div><table>';
    var kvStr = p.voltage_kv != null ? p.voltage_kv + " kV" : "Unknown";
    if (p.voltage_source && p.voltage_source !== "osm") {
        kvStr += ' <span style="color:#e94560;font-size:0.68rem">(' + p.voltage_source.replace(/_/g," ") + ')</span>';
    }
    html += '<tr><td>Voltage</td><td>' + kvStr + '</td></tr>';
    if (p.voltage_label) html += '<tr><td>Class</td><td>' + p.voltage_label + '</td></tr>';
    if (p.frequency_hz) html += '<tr><td>Frequency</td><td>' + p.frequency_hz + ' Hz</td></tr>';
    if (p.rating) html += '<tr><td>Rating</td><td>' + p.rating + '</td></tr>';
    html += '</table></div>';

    // Classification
    html += '<div class="popup-section"><div class="popup-section-title">Classification</div><table>';
    if (p.category_ja) html += '<tr><td>Type</td><td>' + p.category_ja + '</td></tr>';
    if (p.substation_type) html += '<tr><td>OSM Type</td><td>' + p.substation_type + '</td></tr>';
    if (p.gas_insulated != null) html += '<tr><td>GIS</td><td>' + (p.gas_insulated ? "Yes" : "No") + '</td></tr>';
    html += '</table></div>';

    // Operator
    if (p.operator || p.operator_en) {
        html += '<div class="popup-section"><div class="popup-section-title">Operator</div><table>';
        if (p.operator) html += '<tr><td>Name</td><td>' + p.operator + '</td></tr>';
        if (p.operator_en) html += '<tr><td>English</td><td>' + p.operator_en + '</td></tr>';
        html += '</table></div>';
    }

    // Reference
    if (p.ref || p.addr_city || p.website) {
        html += '<div class="popup-section"><div class="popup-section-title">Reference</div><table>';
        if (p.ref) html += '<tr><td>Ref</td><td>' + p.ref + '</td></tr>';
        if (p.addr_city) html += '<tr><td>City</td><td>' + p.addr_city + '</td></tr>';
        if (p.website) html += '<tr><td>Web</td><td><a href="' + p.website + '" target="_blank" style="color:#3498db">Link</a></td></tr>';
        html += '</table></div>';
    }
    html += '</div>';
    return html;
}

function buildGenPopup(p) {
    var fuel = (p.fuel_type_ja || "") + " / " + (p.fuel_type_en || "");
    var html = '<div class="detail-popup">';
    html += '<h3>' + (p.name || "Unknown") + '</h3>';
    var meta = [];
    if (p.operator) meta.push(p.operator);
    if (p.region) meta.push(p.region);
    if (meta.length) html += '<div class="popup-meta">' + meta.join(" | ") + '</div>';

    html += '<div class="popup-section"><div class="popup-section-title">Basic</div><table>';
    html += '<tr><td>Fuel</td><td>' + fuel + '</td></tr>';
    html += '<tr><td>Capacity</td><td>' + fmtNum(p.capacity_mw) + ' MW</td></tr>';
    html += '<tr><td>P_min</td><td>' + fmtNum(p.p_min_mw) + ' MW</td></tr>';
    html += '<tr><td>Dispatchable</td><td>' + (p.dispatchable ? "Yes" : "No") + '</td></tr>';
    html += '</table></div>';

    html += '<div class="popup-section"><div class="popup-section-title">Ramp & Timing</div><table>';
    html += '<tr><td>Ramp Up</td><td>' + fmtNum(p.ramp_up_mw_per_h) + ' MW/h</td></tr>';
    html += '<tr><td>Ramp Down</td><td>' + fmtNum(p.ramp_down_mw_per_h) + ' MW/h</td></tr>';
    html += '<tr><td>Min Up Time</td><td>' + fmtNum(p.min_up_time_h) + ' h</td></tr>';
    html += '<tr><td>Min Down Time</td><td>' + fmtNum(p.min_down_time_h) + ' h</td></tr>';
    html += '<tr><td>Startup Time</td><td>' + fmtNum(p.startup_time_h) + ' h</td></tr>';
    html += '<tr><td>Shutdown Time</td><td>' + fmtNum(p.shutdown_time_h) + ' h</td></tr>';
    html += '</table></div>';

    html += '<div class="popup-section"><div class="popup-section-title">Costs (JPY)</div><table>';
    html += '<tr><td>Startup</td><td>' + fmtNum(p.startup_cost_jpy) + '</td></tr>';
    html += '<tr><td>Shutdown</td><td>' + fmtNum(p.shutdown_cost_jpy) + '</td></tr>';
    html += '<tr><td>Fuel</td><td>' + fmtNum(p.fuel_cost_per_mwh_jpy) + ' /MWh</td></tr>';
    html += '</table></div>';

    html += '<div class="popup-section"><div class="popup-section-title">Efficiency & Emissions</div><table>';
    html += '<tr><td>Heat Rate</td><td>' + fmtNum(p.heat_rate_kj_per_kwh) + ' kJ/kWh</td></tr>';
    html += '<tr><td>CO2</td><td>' + fmtNum(p.co2_intensity_kg_per_mwh) + ' kg/MWh</td></tr>';
    if (p.capacity_factor != null) html += '<tr><td>Capacity Factor</td><td>' + (p.capacity_factor * 100).toFixed(0) + '%</td></tr>';
    html += '</table></div>';

    html += '<div class="popup-section"><div class="popup-section-title">Reliability & Lifecycle</div><table>';
    if (p.planned_outage_rate != null) html += '<tr><td>Planned Outage</td><td>' + (p.planned_outage_rate * 100).toFixed(1) + '%</td></tr>';
    if (p.forced_outage_rate != null) html += '<tr><td>Forced Outage</td><td>' + (p.forced_outage_rate * 100).toFixed(1) + '%</td></tr>';
    html += '<tr><td>Lifetime</td><td>' + fmtNum(p.typical_lifetime_years) + ' years</td></tr>';
    html += '<tr><td>Construction</td><td>' + fmtNum(p.typical_construction_years) + ' years</td></tr>';
    html += '</table></div>';

    html += '</div>';
    return html;
}

// ── Enriched data lookup by coordinates ──

var _enrichedSubIndex = null;
var _enrichedGenIndex = null;

function buildSpatialIndex(geojson) {
    if (!geojson || !geojson.features) return {};
    var idx = {};
    for (var i = 0; i < geojson.features.length; i++) {
        var f = geojson.features[i];
        var c = f.geometry.coordinates;
        // Key: rounded lon,lat for fast lookup
        var key = c[0].toFixed(4) + "," + c[1].toFixed(4);
        idx[key] = f.properties;
    }
    return idx;
}

function lookupEnrichedSub(lon, lat) {
    if (!_enrichedSubIndex && enrichedSubData) {
        _enrichedSubIndex = buildSpatialIndex(enrichedSubData);
    }
    if (!_enrichedSubIndex) return null;
    var key = lon.toFixed(4) + "," + lat.toFixed(4);
    return _enrichedSubIndex[key] || null;
}

function lookupEnrichedGen(lon, lat) {
    if (!_enrichedGenIndex && enrichedGenData) {
        _enrichedGenIndex = buildSpatialIndex(enrichedGenData);
    }
    if (!_enrichedGenIndex) return null;
    var key = lon.toFixed(4) + "," + lat.toFixed(4);
    return _enrichedGenIndex[key] || null;
}

// ── Region list ──

async function initRegionList() {
    try {
        var res = await fetch("./data/regions.json");
        if (!res.ok) return;
        regionsData = await res.json();

        var list = document.getElementById("region-list");
        if (!list) return;

        // "All Regions" button
        var allBtn = document.createElement("button");
        allBtn.className = "region-btn active";
        allBtn.innerHTML = '<span>All Regions (\u5168\u56fd)</span><span class="count"></span>';
        allBtn.onclick = function () { selectRegion(null); };
        list.appendChild(allBtn);

        // Individual regions
        for (var i = 0; i < regionsData.length; i++) {
            (function (r) {
                var btn = document.createElement("button");
                btn.className = "region-btn";
                btn.dataset.region = r.id;
                btn.innerHTML =
                    "<span>" + r.name_en + " (" + r.name_ja + ")</span>" +
                    '<span class="count">' + (r.substations + r.lines) + "</span>";
                btn.onclick = function () { selectRegion(r.id); };
                list.appendChild(btn);
            })(regionsData[i]);
        }
    } catch (err) {
        console.error("Failed to load regions:", err);
    }
}

function setActiveRegionBtn(region) {
    document.querySelectorAll("#region-list .region-btn").forEach(function (btn) {
        btn.classList.toggle("active", btn.dataset.region === region);
    });
    if (!region) {
        var first = document.querySelector("#region-list .region-btn:first-child");
        if (first) first.classList.add("active");
    }
}

// ── Status ──

function setStatus(msg) {
    var el = document.getElementById("status-text");
    if (el) el.textContent = msg;
}

// ── Init ──

document.addEventListener("DOMContentLoaded", function () {
    initMap();
    initTabs();
    initRegionList();

    // Layer visibility checkboxes
    var cbLines = document.getElementById("layer-lines");
    var cbSubs = document.getElementById("layer-subs");
    var cbPlants = document.getElementById("layer-plants");

    if (cbLines) cbLines.addEventListener("change", function () {
        layerVisible.lines = this.checked;
        renderLayers();
    });
    if (cbSubs) cbSubs.addEventListener("change", function () {
        layerVisible.subs = this.checked;
        renderLayers();
    });
    if (cbPlants) cbPlants.addEventListener("change", function () {
        layerVisible.plants = this.checked;
        renderLayers();
    });

    // Voltage filter
    var minKvSelect = document.getElementById("min-kv");
    if (minKvSelect) {
        minKvSelect.addEventListener("change", function () {
            loadData(parseFloat(this.value));
        });
    }

    // Plant filter dropdown
    var plantFilterSelect = document.getElementById("plant-filter");
    if (plantFilterSelect) {
        plantFilterSelect.addEventListener("change", function () {
            plantFilter = this.value;
            loadPlantData().then(function () { renderLayers(); });
        });
    }

    // Load enriched data, then load map data
    loadEnrichedData().then(function () {
        loadData(275);
    });
});
