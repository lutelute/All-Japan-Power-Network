/**
 * Japan Power Grid Map - Static GitHub Pages version.
 *
 * Differences from the server version:
 * - Fetches pre-built GeoJSON from ./data/ instead of /api/
 * - Voltage filtering: 3-tier files (275kv, 154kv, all) + client-side filter
 * - Region click zooms to bounding_box (no individual region GeoJSON)
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
// Map dropdown values to the pre-built file suffix and optional client filter

function voltageTier(minKv) {
    // Returns { suffix, clientFilter } where clientFilter is the kV threshold
    // to apply on the client side (0 means no additional filtering needed).
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

let map;
let substationLayer = null;
let lineLayer = null;
let plantLayer = null;
let currentRegion = null;
let regionsData = [];  // cached regions.json response
let plantsVisible = true;

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
                '<div class="legend-item">' +
                '<span class="legend-color" style="background:' + color + '"></span>' +
                kv + ' kV</div>';
        }
        div.innerHTML += "<h4 style='margin-top:8px'>Power Plants</h4>";
        var fuelEntries = [
            ["nuclear","Nuclear"],["coal","Coal"],["gas","Gas"],["oil","Oil"],
            ["hydro","Hydro"],["wind","Wind"],["solar","Solar"],
            ["geothermal","Geothermal"],["biomass","Biomass"]
        ];
        for (var i = 0; i < fuelEntries.length; i++) {
            var key = fuelEntries[i][0], label = fuelEntries[i][1];
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

// ── Data loading (static files) ──

function filterFeatures(geojson, minKv) {
    if (!minKv || minKv <= 0) return geojson;
    var filtered = geojson.features.filter(function (f) {
        var kv = f.properties._voltage_kv;
        return kv != null && kv >= minKv;
    });
    return { type: "FeatureCollection", features: filtered };
}

async function loadAllRegions(minKv) {
    if (minKv === undefined) {
        var sel = document.getElementById("min-kv");
        minKv = sel ? parseFloat(sel.value) : 275;
    }
    currentRegion = "all";
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

        var subData  = await responses[0].json();
        var lineData = await responses[1].json();

        // Apply client-side voltage filter if needed
        if (tier.clientFilter > 0) {
            subData  = filterFeatures(subData, tier.clientFilter);
            lineData = filterFeatures(lineData, tier.clientFilter);
        }

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
                var kv = p._voltage_kv ? p._voltage_kv + " kV" : "Unknown";
                layer.bindPopup(
                    "<b>" + (p._display_name || "Unnamed") + "</b><br>" +
                    "Voltage: " + kv + "<br>" +
                    "Region: " + (p._region_ja || "")
                );
            },
        }).addTo(map);

        // Load power plants
        await loadPlants();

        map.setView([36.5, 137.0], 6);
        var subCount = subData.features ? subData.features.length : 0;
        var lineCount = lineData.features ? lineData.features.length : 0;
        var plantCount = plantLayer ? plantLayer.getLayers().length : 0;
        var kvLabel = minKv > 0 ? " (" + minKv + " kV+)" : "";
        setStatus("All regions" + kvLabel + ": " + subCount + " substations, " + lineCount + " lines, " + plantCount + " plants");
    } catch (err) {
        setStatus("Error: " + err.message);
    }
}

function loadRegion(region) {
    // In static mode, zoom to bounding box instead of loading separate data
    var r = regionsData.find(function (rd) { return rd.id === region; });
    if (!r || !r.bounding_box) return;

    var bb = r.bounding_box;
    map.fitBounds([
        [bb.lat_min, bb.lon_min],
        [bb.lat_max, bb.lon_max],
    ], { padding: [30, 30] });

    setStatus(r.name_en + " (" + r.name_ja + "): zoom to region");
}

async function loadPlants() {
    if (plantLayer) { map.removeLayer(plantLayer); plantLayer = null; }
    if (!plantsVisible) return;
    try {
        var res = await fetch("./data/plants_all.geojson");
        if (!res.ok) return;
        var plantData = await res.json();
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
                var cap = p.capacity_mw ? p.capacity_mw + " MW" : "N/A";
                layer.bindPopup(
                    "<b>" + (p._display_name || "Unnamed") + "</b><br>" +
                    "Fuel: " + (p.fuel_type || "unknown") + "<br>" +
                    "Capacity: " + cap + "<br>" +
                    "Region: " + (p._region_ja || "")
                );
            },
        }).addTo(map);
    } catch (err) {
        console.error("Failed to load plants:", err);
    }
}

function clearLayers() {
    if (substationLayer) { map.removeLayer(substationLayer); substationLayer = null; }
    if (lineLayer) { map.removeLayer(lineLayer); lineLayer = null; }
    if (plantLayer) { map.removeLayer(plantLayer); plantLayer = null; }
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
        allBtn.onclick = function () {
            setActiveRegionBtn(null);
            loadAllRegions();
        };
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
                btn.onclick = function () {
                    setActiveRegionBtn(r.id);
                    loadRegion(r.id);
                };
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

    // Bind voltage filter
    var minKvSelect = document.getElementById("min-kv");
    if (minKvSelect) {
        minKvSelect.addEventListener("change", function () {
            loadAllRegions(parseFloat(this.value));
        });
    }

    // Bind plants toggle
    var showPlantsCheckbox = document.getElementById("show-plants");
    if (showPlantsCheckbox) {
        showPlantsCheckbox.addEventListener("change", function () {
            plantsVisible = this.checked;
            if (plantsVisible) {
                loadPlants();
            } else if (plantLayer) {
                map.removeLayer(plantLayer);
                plantLayer = null;
            }
        });
    }

    // Load all regions (275 kV+) by default
    loadAllRegions(275);
});
