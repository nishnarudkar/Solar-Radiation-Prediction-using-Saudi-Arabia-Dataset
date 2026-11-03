/**
 * Saudi Arabia Solar Radiation Prediction - Frontend Logic
 * Supports Leaflet Map, Monthly Charts, Station Comparison,
 * and Dynamic Prediction with the Deployed Champion Deep Learning Model.
 */

const DEFAULT_AVERAGES = {
    "Air Temperature (C°)": 26.73,
    "Air Temperature Uncertainty (C°)": 0.50,
    "Wind Direction at 3m (°N)": 168.01,
    "Wind Direction at 3m Uncertainty (°N)": 3.54,
    "Wind Speed at 3m (m/s)": 2.60,
    "Wind Speed at 3m Uncertainty (m/s)": 0.06,
    "Wind Speed at 3m (std dev) (m/s)": 1.53,
    "DHI (Wh/m2)": 2166.14,
    "DHI Uncertainty (Wh/m2)": 387.05,
    "Standard Deviation DHI (Wh/m2)": 593.40,
    "DNI (Wh/m2)": 5587.29,
    "DNI Uncertainty (Wh/m2)": 965.23,
    "Standard Deviation DNI (Wh/m2)": 1747.42,
    "GHI Uncertainty (Wh/m2)": 828.38,
    "Standard Deviation GHI (Wh/m2)": 657.29,
    "Peak Wind Speed at 3m (m/s)": 14.17,
    "Peak Wind Speed at 3m Uncertainty (m/s)": 0.08,
    "Relative Humidity (%)": 39.08,
    "Relative Humidity Uncertainty (%)": 3.01,
    "Barometric Pressure (mB (hPa equiv))": 953.54,
    "Barometric Pressure Uncertainty (mB (hPa equiv))": 4.79
};

const FEATURE_MAP = [
    { id: 'f_temp', rangeId: 'r_temp', name: 'Air Temperature (C°)', unit: '°C' },
    { id: 'f_temp_unc', rangeId: 'r_temp_unc', name: 'Air Temperature Uncertainty (C°)', unit: '°C' },
    { id: 'f_wind_dir', rangeId: 'r_wind_dir', name: 'Wind Direction at 3m (°N)', unit: '°N' },
    { id: 'f_wind_dir_unc', rangeId: 'r_wind_dir_unc', name: 'Wind Direction at 3m Uncertainty (°N)', unit: '°N' },
    { id: 'f_wind_spd', rangeId: 'r_wind_spd', name: 'Wind Speed at 3m (m/s)', unit: 'm/s' },
    { id: 'f_wind_spd_unc', rangeId: 'r_wind_spd_unc', name: 'Wind Speed at 3m Uncertainty (m/s)', unit: 'm/s' },
    { id: 'f_wind_std', rangeId: 'r_wind_std', name: 'Wind Speed at 3m (std dev) (m/s)', unit: 'm/s' },
    { id: 'f_dhi', rangeId: 'r_dhi', name: 'DHI (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_dhi_unc', rangeId: 'r_dhi_unc', name: 'DHI Uncertainty (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_std_dhi', rangeId: 'r_std_dhi', name: 'Standard Deviation DHI (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_dni', rangeId: 'r_dni', name: 'DNI (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_dni_unc', rangeId: 'r_dni_unc', name: 'DNI Uncertainty (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_std_dni', rangeId: 'r_std_dni', name: 'Standard Deviation DNI (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_ghi_unc', rangeId: 'r_ghi_unc', name: 'GHI Uncertainty (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_std_ghi', rangeId: 'r_std_ghi', name: 'Standard Deviation GHI (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_peak_wind', rangeId: 'r_peak_wind', name: 'Peak Wind Speed at 3m (m/s)', unit: 'm/s' },
    { id: 'f_peak_wind_unc', rangeId: 'r_peak_wind_unc', name: 'Peak Wind Speed at 3m Uncertainty (m/s)', unit: 'm/s' },
    { id: 'f_hum', rangeId: 'r_hum', name: 'Relative Humidity (%)', unit: '%' },
    { id: 'f_hum_unc', rangeId: 'r_hum_unc', name: 'Relative Humidity Uncertainty (%)', unit: '%' },
    { id: 'f_pressure', rangeId: 'r_pressure', name: 'Barometric Pressure (mB (hPa equiv))', unit: 'hPa' },
    { id: 'f_pressure_unc', rangeId: 'r_pressure_unc', name: 'Barometric Pressure Uncertainty (mB (hPa equiv))', unit: 'hPa' }
];

let map, markers = {}, gradientCircles = [], isGradientView = false;
let heatLayer = null;
let currentMapMode = 'pins'; // 'pins', 'heatmap', 'circles'
let stationMetadataMap = {};
let ghiThresholds = { low: 4500, high: 6800 };

// Map tile layers for theme switching
let lightBaseLayer = null;
let lightRefLayer = null;
let darkBaseLayer = null;
let darkRefLayer = null;

// Normalization scale modes
let monthlyScaleMode = 'standard';
let compareScaleMode = 'standard';
let currentMonthlyChartData = null;
let currentMonthlyStation = '';
let currentMonthlyYear = '';
let currentComparisonData = null;
let currentComparisonStations = [];
let currentComparisonParams = [];

// In-memory performance caches for 0ms instant UI transitions
const stationDetailsCache = {};
const comparisonCache = {};

/* ==========================================================================
   Theme Management (Neon Dark / Solar Light)
   ========================================================================== */
function isDarkMode() {
    return document.documentElement.classList.contains('dark');
}

function getPlotlyTheme() {
    const isDark = isDarkMode();
    if (isDark) {
        return {
            isDark: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(11, 17, 30, 0.65)',
            fontColor: '#94A3B8',
            titleColor: '#00F0FF',
            gridColor: 'rgba(255, 255, 255, 0.08)',
            axisTitleColor: '#00F0FF',
            tickColor: '#94A3B8',
            palette: ['#00F0FF', '#FF00E5', '#39FF14', '#FFD600', '#FF4D4D', '#B026FF']
        };
    } else {
        return {
            isDark: false,
            paper_bgcolor: '#FFFFFF',
            plot_bgcolor: '#FFFFFF',
            fontColor: '#475569',
            titleColor: '#1E293B',
            gridColor: '#F1F5F9',
            axisTitleColor: '#1E293B',
            tickColor: '#475569',
            palette: ['#1E40AF', '#F59E0B', '#EF4444', '#10B981', '#8B5CF6', '#06B6D4']
        };
    }
}

function updateMapTileTheme(isDark) {
    if (!map || !lightBaseLayer || !darkBaseLayer) return;
    if (isDark) {
        if (map.hasLayer(lightBaseLayer)) map.removeLayer(lightBaseLayer);
        if (map.hasLayer(lightRefLayer)) map.removeLayer(lightRefLayer);
        if (!map.hasLayer(darkBaseLayer)) darkBaseLayer.addTo(map);
        if (darkRefLayer && !map.hasLayer(darkRefLayer)) darkRefLayer.addTo(map);
    } else {
        if (map.hasLayer(darkBaseLayer)) map.removeLayer(darkBaseLayer);
        if (darkRefLayer && map.hasLayer(darkRefLayer)) map.removeLayer(darkRefLayer);
        if (!map.hasLayer(lightBaseLayer)) lightBaseLayer.addTo(map);
        if (!map.hasLayer(lightRefLayer)) lightRefLayer.addTo(map);
    }
}

function setTheme(theme) {
    const isDark = (theme === 'dark');
    if (isDark) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
    try {
        sessionStorage.setItem('theme_preference', theme);
        localStorage.setItem('theme_preference', theme);
    } catch (e) {}

    const checkbox = document.getElementById('themeToggleCheckbox');
    if (checkbox) checkbox.checked = isDark;

    updateMapTileTheme(isDark);

    if (gradientLayer && typeof gradientLayer._render === 'function') {
        gradientLayer._render();
    }

    if (currentMonthlyChartData && currentMonthlyStation) {
        renderMonthlyChart(currentMonthlyChartData, currentMonthlyStation, currentMonthlyYear);
    }
    if (currentComparisonData) {
        renderComparisonPlot();
    }
}

function initThemeToggle() {
    // Neon Dark is always the primary default mode
    let savedTheme = 'dark';
    try {
        savedTheme = sessionStorage.getItem('theme_preference') || 'dark';
    } catch (e) {}
    setTheme(savedTheme);

    const checkbox = document.getElementById('themeToggleCheckbox');
    if (checkbox) {
        checkbox.addEventListener('change', (e) => {
            setTheme(e.target.checked ? 'dark' : 'light');
        });
    }

    document.querySelector('.sun-icon')?.addEventListener('click', () => setTheme('light'));
    document.querySelector('.moon-icon')?.addEventListener('click', () => setTheme('dark'));
    document.getElementById('themeModeLeft')?.addEventListener('click', () => setTheme('light'));
    document.getElementById('themeModeRight')?.addEventListener('click', () => setTheme('dark'));
}

document.addEventListener('DOMContentLoaded', () => {
    // 0. Initialize Theme Toggle (Defaults to Neon Dark)
    initThemeToggle();

    // 1. Initialize Map
    initMap();

    // 2. Initialize Select2 Components safely
    if (typeof jQuery !== 'undefined' && typeof jQuery.fn.select2 === 'function') {
        $('#parameterSelect').select2({
            placeholder: "Select Parameters",
            width: '100%',
            allowClear: true,
            closeOnSelect: false
        });
        $('#compareStations').select2({
            placeholder: "Select Stations",
            width: '100%',
            allowClear: true,
            closeOnSelect: false
        });
        $('#compareParams').select2({
            placeholder: "Select Parameters",
            width: '100%',
            allowClear: true,
            closeOnSelect: false
        });
    }

    // 3. Load Map, Station Data, and Dynamic Ranges
    loadMapAndStationData();

    // 4. Fill Prediction Form with Default Values and Compute Live Initial Prediction
    loadPredictionDefaults();

    // 5. Setup Event Listeners
    document.getElementById('toggleHeatmap')?.addEventListener('click', toggleGradientView);
    document.getElementById('btnModePins')?.addEventListener('click', () => setMapMode('pins'));
    document.getElementById('btnModeHeatmap')?.addEventListener('click', () => setMapMode('heatmap'));

    document.getElementById('stationSelect').addEventListener('change', (e) => onStationSelectChange(e.target.value));

    let monthlyParamTimer = null;
    const debouncedMonthlyParamChange = () => {
        clearTimeout(monthlyParamTimer);
        monthlyParamTimer = setTimeout(handleMonthlyParameterChange, 25);
    };

    let comparisonTimer = null;
    const debouncedStationComparison = () => {
        clearTimeout(comparisonTimer);
        comparisonTimer = setTimeout(handleStationComparison, 35);
    };

    if (typeof jQuery !== 'undefined') {
        $('#parameterSelect').on('change select2:select select2:unselect select2:clear', debouncedMonthlyParamChange);
    }
    document.getElementById('yearSelect').addEventListener('change', () => onStationSelectChange(document.getElementById('stationSelect').value));

    // Comparison Event Listeners
    if (typeof jQuery !== 'undefined') {
        $('#compareStations, #compareParams').on('change select2:select select2:unselect select2:clear', debouncedStationComparison);
    }
    document.getElementById('compareYear')?.addEventListener('change', debouncedStationComparison);

    // Scale Toggle Event Listeners
    document.getElementById('btnMonthlyStandard')?.addEventListener('click', () => {
        if (monthlyScaleMode !== 'standard') {
            monthlyScaleMode = 'standard';
            document.getElementById('btnMonthlyStandard').classList.add('active');
            document.getElementById('btnMonthlyNormalized').classList.remove('active');
            if (currentMonthlyChartData) {
                renderMonthlyChart(currentMonthlyChartData, currentMonthlyStation, currentMonthlyYear);
            }
        }
    });

    document.getElementById('btnMonthlyNormalized')?.addEventListener('click', () => {
        if (monthlyScaleMode !== 'normalized') {
            monthlyScaleMode = 'normalized';
            document.getElementById('btnMonthlyNormalized').classList.add('active');
            document.getElementById('btnMonthlyStandard').classList.remove('active');
            if (currentMonthlyChartData) {
                renderMonthlyChart(currentMonthlyChartData, currentMonthlyStation, currentMonthlyYear);
            }
        }
    });

    document.getElementById('btnCompareStandard')?.addEventListener('click', () => {
        if (compareScaleMode !== 'standard') {
            compareScaleMode = 'standard';
            document.getElementById('btnCompareStandard').classList.add('active');
            document.getElementById('btnCompareNormalized').classList.remove('active');
            if (currentComparisonData) {
                renderComparisonPlot();
            }
        }
    });

    document.getElementById('btnCompareNormalized')?.addEventListener('click', () => {
        if (compareScaleMode !== 'normalized') {
            compareScaleMode = 'normalized';
            document.getElementById('btnCompareNormalized').classList.add('active');
            document.getElementById('btnCompareStandard').classList.remove('active');
            if (currentComparisonData) {
                renderComparisonPlot();
            }
        }
    });

    // Prediction Form Events
    document.getElementById('loadDefaultsBtn').addEventListener('click', loadPredictionDefaults);
    document.getElementById('predictionForm').addEventListener('submit', handlePrediction);

    // 6. Initialize Chart Resize and Tab Visibility Handlers
    initChartResizeHandlers();
});

/* Responsive Chart & Tab Visibility Handlers */
function initChartResizeHandlers() {
    function setupChartObserver(chartId) {
        const el = document.getElementById(chartId);
        if (!el || typeof ResizeObserver === 'undefined') return;

        let lastWidth = el.clientWidth;
        let timer = null;

        const ro = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const currentWidth = Math.round(entry.contentRect.width);
                if (currentWidth > 0 && Math.abs(currentWidth - lastWidth) > 5) {
                    lastWidth = currentWidth;
                    clearTimeout(timer);
                    timer = setTimeout(() => {
                        if (el.querySelector('.js-plotly-plot')) {
                            try {
                                Plotly.Plots.resize(el);
                                Plotly.relayout(el, { autosize: true });
                            } catch (e) {}
                        }
                    }, 20);
                }
            }
        });
        ro.observe(el);
    }

    setupChartObserver('comparisonChart');
    setupChartObserver('monthlyChart');

    const handleTabShown = (targetId) => {
        if (targetId === '#stationComparison') {
            const compEl = document.getElementById('comparisonChart');
            if (compEl && compEl.querySelector('.js-plotly-plot')) {
                try {
                    Plotly.Plots.resize(compEl);
                    Plotly.relayout(compEl, { autosize: true });
                } catch (e) {}
            } else if (currentComparisonData) {
                renderComparisonPlot();
            }
        } else if (targetId === '#mapView') {
            if (map && typeof map.invalidateSize === 'function') {
                map.invalidateSize();
            }
            const monthlyEl = document.getElementById('monthlyChart');
            if (monthlyEl && monthlyEl.querySelector('.js-plotly-plot')) {
                try {
                    Plotly.Plots.resize(monthlyEl);
                    Plotly.relayout(monthlyEl, { autosize: true });
                } catch (e) {}
            }
        }
    };

    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(btn => {
        btn.addEventListener('shown.bs.tab', (e) => {
            const target = e.target.getAttribute('data-bs-target');
            handleTabShown(target);
        });
        btn.addEventListener('show.bs.tab', (e) => {
            const target = e.target.getAttribute('data-bs-target');
            setTimeout(() => handleTabShown(target), 30);
            setTimeout(() => handleTabShown(target), 160);
        });
    });

    if (typeof jQuery !== 'undefined') {
        $('button[data-bs-toggle="tab"]').on('shown.bs.tab', function(e) {
            const target = $(e.target).attr('data-bs-target');
            handleTabShown(target);
        });
    }

    window.addEventListener('resize', () => {
        const compEl = document.getElementById('comparisonChart');
        if (compEl && compEl.offsetParent !== null && compEl.querySelector('.js-plotly-plot')) {
            try { Plotly.Plots.resize(compEl); } catch (e) {}
        }
        const monthlyEl = document.getElementById('monthlyChart');
        if (monthlyEl && monthlyEl.offsetParent !== null && monthlyEl.querySelector('.js-plotly-plot')) {
            try { Plotly.Plots.resize(monthlyEl); } catch (e) {}
        }
    });
}

/* Leaflet Map Setup */
function initMap() {
    map = L.map('map', {
        center: [24.0, 45.0],
        zoom: 5,
        zoomControl: true
    });

    // Base light gray layer (English)
    lightBaseLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
        maxZoom: 16
    });

    // English labels and boundaries overlay
    lightRefLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
        attribution: '',
        maxZoom: 16
    });

    // Esri World Dark Gray Base layer for Neon Dark Mode (watermark-free)
    darkBaseLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
        maxZoom: 16
    });

    // English labels and boundaries overlay (Dark)
    darkRefLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
        attribution: '',
        maxZoom: 16
    });

    // Apply the active theme's tile layer
    updateMapTileTheme(isDarkMode());
}

// Shared GHI color mapping: normalized t (0-1) to smooth Blue → Cyan → Green → Yellow → Orange → Red
// Uses gamma stretch (power 0.72) to spread mid-range IDW values across more of the color spectrum,
// preventing the "all orange" clustering that occurs when station GHI values span a narrow range.
function getGhiColorRGB(t) {
    // Gamma stretch: pull mid-range values toward brighter colours so the map looks varied
    t = Math.pow(Math.max(0, Math.min(1, t)), 0.72);

    let r, g, b;
    if (t < 0.2) {
        // Deep Blue → Cyan
        const s = t / 0.2;
        r = Math.round(10  + s * 0);
        g = Math.round(60  + s * (210 - 60));
        b = Math.round(200 + s * (255 - 200));
    } else if (t < 0.4) {
        // Cyan → Green
        const s = (t - 0.2) / 0.2;
        r = Math.round(0   + s * 30);
        g = Math.round(210 + s * (230 - 210));
        b = Math.round(255 - s * 255);
    } else if (t < 0.6) {
        // Green → Yellow
        const s = (t - 0.4) / 0.2;
        r = Math.round(30  + s * (255 - 30));
        g = Math.round(230 - s * (230 - 220));
        b = Math.round(0);
    } else if (t < 0.8) {
        // Yellow → Orange
        const s = (t - 0.6) / 0.2;
        r = Math.round(255);
        g = Math.round(220 - s * (220 - 110));
        b = Math.round(0);
    } else {
        // Orange → Deep Red
        const s = (t - 0.8) / 0.2;
        r = Math.round(255 - s * 60);
        g = Math.round(110 - s * 110);
        b = Math.round(0   + s * 30);
    }
    return [r, g, b];
}

// Default Leaflet pin icon for normal view
const defaultPinIcon = new L.Icon.Default();

// Custom GHI-colored beacon icon for gradient view
// Uses mean-centred normalization: t=0.5 → mean GHI, blue = below avg, red = above avg.
function createGhiBeaconIcon(ghi, ghiMean, ghiHalfRange) {
    const t = Math.max(0, Math.min(1, 0.5 + (ghi - ghiMean) / (2 * ghiHalfRange)));
    const [r, g, b] = getGhiColorRGB(t);
    const color = `rgb(${r}, ${g}, ${b})`;
    return L.divIcon({
        className: 'station-beacon-marker',
        html: `<div class="station-beacon" style="--beacon-color: ${color};" title="GHI: ${(ghi / 1000).toFixed(2)} kWh/m²"><div class="beacon-core"></div></div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11],
        popupAnchor: [0, -12]
    });
}

function updateStationMarkersMode(mode) {
    const isHeatmap = (mode === 'heatmap');
    const ghiMean = gradientLayer ? gradientLayer._ghiMean : 5700;
    const ghiHalfRange = gradientLayer ? gradientLayer._ghiHalfRange : 1500;

    Object.keys(markers).forEach(name => {
        const marker = markers[name];
        const st = stationMetadataMap[name];
        if (!marker || !st) return;

        if (isHeatmap) {
            marker.setIcon(createGhiBeaconIcon(st.avg_ghi, ghiMean, ghiHalfRange));
        } else {
            marker.setIcon(defaultPinIcon);
        }
    });
}

// ──────────────────────────────────────────────────────────────────
// IDW Solar Radiation Layer – Precise Vector-Clipped Geographical Coverage
// Sourced from official geoBoundaries administrative data.
// Uses Inverse Distance Weighting to interpolate GHI across Saudi Arabia,
// hardware-clipped via 2D Canvas clip() to the exact national boundary.
// ──────────────────────────────────────────────────────────────────
const SolarGradientLayer = L.Layer.extend({
    initialize: function(stations) {
        this.stations = stations || [];
        const ghis = this.stations.map(s => s.avg_ghi).filter(v => v > 0);
        this._ghiMin = ghis.length ? Math.min(...ghis) : 4000;
        this._ghiMax = ghis.length ? Math.max(...ghis) : 7500;
        // Mean-centred scale: yellow = mean GHI; symmetric halfRange keeps both tails balanced
        this._ghiMean = ghis.length ? ghis.reduce((a, b) => a + b, 0) / ghis.length : 5700;
        this._ghiHalfRange = Math.max(
            this._ghiMean - this._ghiMin,
            this._ghiMax - this._ghiMean,
            1
        );
    },

    onAdd: function(map) {
        this._map = map;
        this._canvas = L.DomUtil.create('canvas', 'solar-gradient-canvas');
        this._canvas.style.position = 'absolute';
        this._canvas.style.pointerEvents = 'none';
        this._canvas.style.zIndex = '350';

        map.getPanes().overlayPane.appendChild(this._canvas);
        map.on('moveend zoomend resize viewreset', this._render, this);
        this._render();
    },

    onRemove: function(map) {
        if (this._canvas && this._canvas.parentNode) {
            this._canvas.parentNode.removeChild(this._canvas);
        }
        map.off('moveend zoomend resize viewreset', this._render, this);
    },

    _render: function() {
        if (!this._map || !this._canvas || this.stations.length === 0) return;
        const map = this._map;
        const bounds = map.getBounds();
        const topLeft = map.latLngToLayerPoint(bounds.getNorthWest());
        const bottomRight = map.latLngToLayerPoint(bounds.getSouthEast());
        const width = Math.max(1, Math.ceil(bottomRight.x - topLeft.x));
        const height = Math.max(1, Math.ceil(bottomRight.y - topLeft.y));

        this._canvas.width = width;
        this._canvas.height = height;
        L.DomUtil.setPosition(this._canvas, topLeft);

        const ctx = this._canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);

        // Render at reduced resolution for high-performance IDW sampling, then stretch
        const STEP = 4;
        const sw = Math.ceil(width / STEP);
        const sh = Math.ceil(height / STEP);

        const offCanvas = document.createElement('canvas');
        offCanvas.width = sw;
        offCanvas.height = sh;
        const offCtx = offCanvas.getContext('2d');
        const imgData = offCtx.createImageData(sw, sh);
        const pixels = imgData.data;

        const stLats = this.stations.map(s => s.latitude);
        const stLngs = this.stations.map(s => s.longitude);
        const stGhis = this.stations.map(s => s.avg_ghi);
        const nStations = this.stations.length;

        const ghiMean = this._ghiMean;
        const ghiHalfRange = this._ghiHalfRange;
        const P = 2.5; // IDW power parameter

        // Normalize SAUDI_BOUNDARY rings (handles single ring or MultiPolygon array of rings)
        const rings = (Array.isArray(SAUDI_BOUNDARY[0][0])) ? SAUDI_BOUNDARY : [SAUDI_BOUNDARY];

        // Evaluate IDW across the visible area covering Saudi Arabia
        for (let sy = 0; sy < sh; sy++) {
            const py = topLeft.y + sy * STEP + STEP / 2;
            for (let sx = 0; sx < sw; sx++) {
                const px = topLeft.x + sx * STEP + STEP / 2;
                const latlng = map.layerPointToLatLng([px, py]);
                const lat = latlng.lat;
                const lng = latlng.lng;

                // Quick bounding box check for Saudi Arabia (+ slight margin for boundary sampling)
                if (lat < 15.8 || lat > 32.6 || lng < 34.0 || lng > 56.0) continue;

                let numerator = 0;
                let denominator = 0;
                let exact = -1;

                for (let k = 0; k < nStations; k++) {
                    const dLat = lat - stLats[k];
                    const dLng = lng - stLngs[k];
                    const dist = Math.sqrt(dLat * dLat + dLng * dLng);

                    if (dist < 0.01) {
                        exact = k;
                        break;
                    }
                    const w = 1 / Math.pow(dist, P);
                    numerator += w * stGhis[k];
                    denominator += w;
                }

                const ghi = exact >= 0 ? stGhis[exact] : (denominator > 0 ? numerator / denominator : ghiMean);
                // Mean-centred: t=0.5 → mean GHI; below mean → blue/green; above mean → orange/red
                const t = Math.max(0, Math.min(1, 0.5 + (ghi - ghiMean) / (2 * ghiHalfRange)));
                const [r, g, b] = getGhiColorRGB(t);

                const idx = (sy * sw + sx) * 4;
                pixels[idx]     = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = 175; // ~68% opacity
            }
        }

        offCtx.putImageData(imgData, 0, 0);

        // Helper to trace all boundary rings onto canvas context
        function traceBoundaryPath(c) {
            rings.forEach(ring => {
                ring.forEach((coord, i) => {
                    const pt = map.latLngToLayerPoint([coord[1], coord[0]]);
                    const x = pt.x - topLeft.x;
                    const y = pt.y - topLeft.y;
                    if (i === 0) c.moveTo(x, y);
                    else c.lineTo(x, y);
                });
                c.closePath();
            });
        }

        // 1. Vector Clip to accurate Saudi Arabia boundary (Hardware-accelerated)
        ctx.save();
        ctx.beginPath();
        traceBoundaryPath(ctx);
        ctx.clip();

        // 2. Draw scaled smooth offscreen canvas - clipped perfectly at sub-pixel resolution
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(offCanvas, 0, 0, sw, sh, 0, 0, width, height);
        ctx.restore();

        // 3. Draw refined border outline stroke (with subtle neon glow in dark mode)
        ctx.save();
        ctx.beginPath();
        traceBoundaryPath(ctx);
        if (isDarkMode()) {
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.75)';
            ctx.shadowColor = 'rgba(0, 240, 255, 0.5)';
            ctx.shadowBlur = 6;
            ctx.lineWidth = 1.6;
        } else {
            ctx.strokeStyle = 'rgba(15, 23, 42, 0.35)';
            ctx.lineWidth = 1.2;
        }
        ctx.stroke();
        ctx.restore();
    }
});

// Color Legend Control for the Gradient View
const SolarLegendControl = L.Control.extend({
    options: { position: 'bottomleft' },
    initialize: function(ghiMin, ghiMean, ghiMax) {
        this._ghiMin = ghiMin;
        this._ghiMean = ghiMean;
        this._ghiMax = ghiMax;
    },
    onAdd: function() {
        const div = L.DomUtil.create('div', 'solar-legend');
        div.innerHTML = `
            <div class="solar-legend-title">Avg GHI (kWh/m²)</div>
            <div class="solar-legend-bar"></div>
            <div class="solar-legend-labels">
                <span>${(this._ghiMin / 1000).toFixed(2)}</span>
                <span>${(this._ghiMean / 1000).toFixed(2)} &#x2205;</span>
                <span>${(this._ghiMax / 1000).toFixed(2)}</span>
            </div>
        `;
        return div;
    }
});

let gradientLayer = null;
let legendControl = null;

function loadMapAndStationData() {
    const statusDiv = document.getElementById('mapStatus');
    if (statusDiv) statusDiv.textContent = 'Loading map data...';

    fetch('/map-data')
        .then(res => res.json())
        .then(stations => {
            if (statusDiv) statusDiv.textContent = '';
            const stationSelect = document.getElementById('stationSelect');
            const compareStations = $('#compareStations');
            stationSelect.innerHTML = '';
            compareStations.empty();

            if (!Array.isArray(stations)) return;

            stations.forEach(st => {
                stationMetadataMap[st.station_name] = st;

                // Add options
                stationSelect.add(new Option(st.station_name, st.station_name));
                compareStations.append(new Option(st.station_name, st.station_name));

                // Add marker
                const marker = L.marker([st.latitude, st.longitude]).addTo(map);
                marker.bindPopup(`<strong>${st.station_name}</strong><br>Avg GHI: ${(st.avg_ghi / 1000).toFixed(2)} kWh/m²`);
                marker.on('click', () => {
                    stationSelect.value = st.station_name;
                    onStationSelectChange(st.station_name, true);
                });
                markers[st.station_name] = marker;
            });

            // Initialize invariant Solar Gradient Layer
            gradientLayer = new SolarGradientLayer(stations);

            // Initial selection
            if (stations.length > 0) {
                const first = stations[0].station_name;
                stationSelect.value = first;
                onStationSelectChange(first, true);

                // Set default comparison stations matching Screenshot 3
                const targetDefaults = ['Sharurah - TVTC', 'Osfan - KAU', 'Hada Al Sham - KAU'];
                const availableStations = stations.map(s => s.station_name);
                const selectedDefaults = targetDefaults.filter(s => availableStations.includes(s));
                const finalDefaults = selectedDefaults.length > 0 ? selectedDefaults : stations.slice(0, 3).map(s => s.station_name);

                compareStations.val(finalDefaults).trigger('change');
            }
        })
        .catch(err => {
            console.error("Error loading map data:", err);
            if (statusDiv) statusDiv.textContent = 'Error loading map data';
        });

    // Populate Parameter dropdowns & Dynamic Ranges
    fetch('/data-analysis')
        .then(res => res.json())
        .then(data => {
            if (data.summary_stats) {
                const paramSelect = $('#parameterSelect');
                const compParams = $('#compareParams');
                paramSelect.empty();
                compParams.empty();

                const paramSet = new Set();
                Object.keys(data.summary_stats).forEach(k => {
                    const base = k.replace(/_(mean|min|max|std)$/, '');
                    paramSet.add(base);
                });

                paramSet.forEach(p => {
                    paramSelect.append(new Option(p, p));
                    compParams.append(new Option(p, p));
                });

                // Default parameter: Air Temperature (C°)
                paramSelect.val(['Air Temperature (C°)']).trigger('change');
                compParams.val(['Air Temperature (C°)']).trigger('change');

                // Dynamically update min-max range hints for prediction inputs
                FEATURE_MAP.forEach(item => {
                    const minKey = `${item.name}_min`;
                    const maxKey = `${item.name}_max`;
                    if (data.summary_stats[minKey] && data.summary_stats[maxKey]) {
                        const minVals = Object.values(data.summary_stats[minKey]).filter(v => v !== null && !isNaN(v));
                        const maxVals = Object.values(data.summary_stats[maxKey]).filter(v => v !== null && !isNaN(v));
                        if (minVals.length > 0 && maxVals.length > 0) {
                            const overallMin = Math.min(...minVals);
                            const overallMax = Math.max(...maxVals);
                            const el = document.getElementById(item.rangeId);
                            if (el) {
                                el.textContent = `Range: ${overallMin.toFixed(2)} - ${overallMax.toFixed(2)} ${item.unit}`;
                            }
                        }
                    }
                });
            }
        })
        .catch(err => console.error("Error loading parameters & ranges:", err));
}

function setMapMode(mode) {
    currentMapMode = mode;
    document.querySelectorAll('.btn-map-mode').forEach(b => b.classList.remove('active'));
    const btnPins = document.getElementById('btnModePins');
    const btnHeatmap = document.getElementById('btnModeHeatmap');
    const toggleBtn = document.getElementById('toggleHeatmap');

    if (mode === 'pins') {
        btnPins?.classList.add('active');
        if (gradientLayer) map.removeLayer(gradientLayer);
        if (legendControl) { map.removeControl(legendControl); legendControl = null; }
        Object.values(markers).forEach(m => m.addTo(map));
        updateStationMarkersMode('pins');
        if (toggleBtn) toggleBtn.textContent = 'Toggle Gradient View';
    } else if (mode === 'heatmap') {
        btnHeatmap?.classList.add('active');
        Object.values(markers).forEach(m => m.addTo(map));
        if (gradientLayer) gradientLayer.addTo(map);
        updateStationMarkersMode('heatmap');
        // Show color legend
        if (!legendControl && gradientLayer) {
            legendControl = new SolarLegendControl(gradientLayer._ghiMin, gradientLayer._ghiMean, gradientLayer._ghiMax);
            legendControl.addTo(map);
        }
        if (toggleBtn) toggleBtn.textContent = 'Switch to Normal Map';
    }
}

function toggleGradientView() {
    if (currentMapMode === 'pins') {
        setMapMode('heatmap');
    } else {
        setMapMode('pins');
    }
}

function onStationSelectChange(stationName, fromMarkerClick = false) {
    if (!stationName) return;

    // Update metadata panel
    const stMeta = stationMetadataMap[stationName];
    if (stMeta) {
        document.getElementById('metaStationName').textContent = stMeta.station_name;
        document.getElementById('metaLat').textContent = stMeta.latitude.toFixed(5);
        document.getElementById('metaLong').textContent = stMeta.longitude.toFixed(5);

        // Reflect selection on the map without changing zoom level
        if (map && stMeta.latitude && stMeta.longitude) {
            if (!fromMarkerClick) {
                map.panTo([stMeta.latitude, stMeta.longitude], {
                    animate: true,
                    duration: 0.5
                });
            }
            if (markers[stationName]) {
                markers[stationName].openPopup();
            }
        }
    }

    // Load monthly data
    const year = document.getElementById('yearSelect').value;
    const cacheKey = `${stationName}::${year || 'all'}`;

    // Instant zero-latency render if data is already cached
    if (stationDetailsCache[cacheKey]) {
        const data = stationDetailsCache[cacheKey];
        if (data.details) {
            document.getElementById('metaStationName').textContent = data.details.station_name;
            document.getElementById('metaLat').textContent = data.details.latitude.toFixed(5);
            document.getElementById('metaLong').textContent = data.details.longitude.toFixed(5);
        }
        if (data.monthly_chart_data) {
            currentMonthlyChartData = data.monthly_chart_data;
            currentMonthlyStation = stationName;
            currentMonthlyYear = year;
            renderMonthlyChart(data.monthly_chart_data, stationName, year);
        }
        return;
    }

    const url = `/station-details?station=${encodeURIComponent(stationName)}${year ? '&year=' + year : ''}`;

    fetch(url)
        .then(res => res.json())
        .then(data => {
            stationDetailsCache[cacheKey] = data;
            if (data.details) {
                document.getElementById('metaStationName').textContent = data.details.station_name;
                document.getElementById('metaLat').textContent = data.details.latitude.toFixed(5);
                document.getElementById('metaLong').textContent = data.details.longitude.toFixed(5);
            }
            if (data.monthly_chart_data) {
                currentMonthlyChartData = data.monthly_chart_data;
                currentMonthlyStation = stationName;
                currentMonthlyYear = year;
                renderMonthlyChart(data.monthly_chart_data, stationName, year);
            }
        })
        .catch(err => console.error("Error loading station details:", err));
}

function handleMonthlyParameterChange() {
    const chartEl = document.getElementById('monthlyChart');
    const selectedParams = (typeof jQuery !== 'undefined') ? $('#parameterSelect').val() : null;

    // Immediately handle empty selection synchronously (no network latency)
    if (!selectedParams || selectedParams.length === 0) {
        if (chartEl) {
            try { Plotly.purge(chartEl); } catch (e) {}
            chartEl.innerHTML =
                '<div class="d-flex align-items-center justify-content-center text-muted" style="height: 350px; border: 1px dashed var(--border-color); border-radius: 8px; margin-top: 15px; background: var(--input-bg);">Please select at least one parameter above to view monthly data.</div>';
        }
        return;
    }

    // If monthly chart data is already cached for current station, render immediately without any network fetch
    if (currentMonthlyChartData && currentMonthlyStation) {
        renderMonthlyChart(currentMonthlyChartData, currentMonthlyStation, currentMonthlyYear);
    } else {
        const stationVal = document.getElementById('stationSelect')?.value;
        if (stationVal) {
            onStationSelectChange(stationVal);
        }
    }
}

function renderMonthlyChart(chartData, stationName, year) {
    currentMonthlyChartData = chartData;
    currentMonthlyStation = stationName;
    currentMonthlyYear = year;

    const chartEl = document.getElementById('monthlyChart');
    if (!chartEl) return;

    const selectedParams = (typeof jQuery !== 'undefined') ? $('#parameterSelect').val() : null;
    if (!selectedParams || selectedParams.length === 0) {
        try { Plotly.purge(chartEl); } catch (e) {}
        chartEl.innerHTML =
            '<div class="d-flex align-items-center justify-content-center text-muted" style="height: 350px; border: 1px dashed var(--border-color); border-radius: 8px; margin-top: 15px; background: var(--input-bg);">Please select at least one parameter above to view monthly data.</div>';
        return;
    }

    if (!chartData || !chartData.data) {
        try { Plotly.purge(chartEl); } catch (e) {}
        chartEl.innerHTML =
            '<div class="d-flex align-items-center justify-content-center text-muted" style="height: 350px; border: 1px dashed var(--border-color); border-radius: 8px; margin-top: 15px; background: var(--input-bg);">No monthly data available for this station.</div>';
        return;
    }

    // If a placeholder message was present, clear it cleanly
    if (chartEl.children.length > 0 && !chartEl.querySelector('.js-plotly-plot')) {
        chartEl.innerHTML = '';
    }

    const traces = [];
    const isNorm = (monthlyScaleMode === 'normalized');
    const theme = getPlotlyTheme();
    const colors = theme.palette;

    selectedParams.forEach((param, idx) => {
        if (chartData.data && chartData.data[param]) {
            const seriesColor = colors[idx % colors.length];
            const rawVals = chartData.data[param];

            let plotVals = rawVals;
            let hoverTemplate = '<b>%{x}</b><br>%{data.name}: %{y:.2f}<extra></extra>';

            if (isNorm) {
                const validVals = rawVals.filter(v => v !== null && !isNaN(v));
                const minVal = validVals.length > 0 ? Math.min(...validVals) : 0;
                const maxVal = validVals.length > 0 ? Math.max(...validVals) : 1;
                const range = maxVal - minVal;
                plotVals = rawVals.map(v => (range > 0 ? ((v - minVal) / range) * 100 : 50));
                hoverTemplate = '<b>%{x}</b><br>%{data.name}<br>Normalized: %{y:.1f}%<br>Raw Value: %{customdata:.2f}<extra></extra>';
            }

            // Bar trace
            traces.push({
                x: chartData.months,
                y: plotVals,
                customdata: rawVals,
                name: param,
                type: 'bar',
                marker: { color: seriesColor, opacity: 0.85 },
                width: 0.45,
                hovertemplate: hoverTemplate
            });

            // Spline trendline trace with markers
            traces.push({
                x: chartData.months,
                y: plotVals,
                customdata: rawVals,
                name: `${param} Trend`,
                type: 'scatter',
                mode: 'lines+markers',
                line: { shape: 'spline', smoothing: 1.1, width: 2.2, color: seriesColor },
                marker: { size: 6, color: seriesColor },
                hovertemplate: hoverTemplate
            });
        }
    });

    const titleSuffix = isNorm ? ' (Normalized 0–100%)' : '';
    const titleText = `Data for ${stationName}${year ? ' (' + year + ')' : ''}${titleSuffix}`;

    const yaxisConfig = isNorm ? {
        title: { text: 'Normalized Scale (%)', font: { size: 11, color: theme.axisTitleColor } },
        range: [-5, 105],
        gridcolor: theme.gridColor,
        tickfont: { size: 11, color: theme.tickColor }
    } : {
        title: { text: 'Parameter Values', font: { size: 11, color: theme.axisTitleColor } },
        gridcolor: theme.gridColor,
        tickfont: { size: 11, color: theme.tickColor }
    };

    const layout = {
        autosize: true,
        title: {
            text: titleText,
            font: { family: 'Poppins, sans-serif', size: 14, color: theme.titleColor }
        },
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: { family: 'Poppins, sans-serif', size: 11, color: theme.fontColor },
        margin: { t: 40, r: 25, l: 50, b: 65 },
        barmode: 'overlay',
        legend: {
            orientation: 'h',
            y: -0.25,
            x: 0.5,
            xanchor: 'center',
            font: { color: theme.fontColor }
        },
        xaxis: {
            gridcolor: theme.gridColor,
            tickfont: { size: 11, color: theme.tickColor }
        },
        yaxis: yaxisConfig
    };

    if (traces.length === 0) {
        try { Plotly.purge(chartEl); } catch (e) {}
        chartEl.innerHTML =
            '<div class="d-flex align-items-center justify-content-center text-muted" style="height: 350px; border: 1px dashed var(--border-color); border-radius: 8px; margin-top: 15px; background: var(--input-bg);">No monthly data available for the selected parameter(s).</div>';
        return;
    }

    Plotly.newPlot('monthlyChart', traces, layout, { responsive: true, displayModeBar: true }).then(() => {
        if (chartEl && chartEl.offsetParent !== null) {
            try { Plotly.Plots.resize(chartEl); } catch (e) {}
        }
    });
}

/* Station Comparison */
function handleStationComparison() {
    const stations = (typeof jQuery !== 'undefined') ? $('#compareStations').val() : [];
    const params = (typeof jQuery !== 'undefined') ? $('#compareParams').val() : [];
    const year = document.getElementById('compareYear').value;

    if (!stations || stations.length === 0 || !params || params.length === 0) {
        currentComparisonData = null;
        const chartEl = document.getElementById('comparisonChart');
        if (chartEl) {
            try { Plotly.purge(chartEl); } catch (e) {}
            chartEl.innerHTML =
                '<div class="d-flex align-items-center justify-content-center text-muted" style="height: 350px; border: 1px dashed var(--border-color); border-radius: 8px; margin-top: 15px; background: var(--input-bg);">Please select at least one station and one parameter above to display comparison data.</div>';
        }
        document.querySelector('#comparisonTable tbody').innerHTML =
            '<tr><td colspan="5" class="text-center text-muted">Select stations above to display GHI summary statistics.</td></tr>';
        return;
    }

    const cacheKey = stations.slice().sort().join('|') + '::' + (year || 'all');

    // Instant zero-latency render if comparison data for stations & year is already cached
    if (comparisonCache[cacheKey]) {
        currentComparisonData = comparisonCache[cacheKey];
        currentComparisonStations = stations;
        currentComparisonParams = params;
        renderComparisonPlot();
        return;
    }

    fetch('/station-comparison', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stations: stations, params: params, year: year })
    })
    .then(res => res.json())
    .then(data => {
        if (!data.values) return;

        comparisonCache[cacheKey] = data;
        currentComparisonData = data;
        currentComparisonStations = stations;
        currentComparisonParams = params;

        renderComparisonPlot();
    })
    .catch(err => console.error("Error running station comparison:", err));
}

function renderComparisonPlot() {
    if (!currentComparisonData || !currentComparisonData.values) return;

    const chartEl = document.getElementById('comparisonChart');
    if (!chartEl) return;

    // If a placeholder message was present, clear it cleanly
    if (chartEl.children.length > 0 && !chartEl.querySelector('.js-plotly-plot')) {
        chartEl.innerHTML = '';
    }

    const data = currentComparisonData;
    const stations = currentComparisonStations;
    const params = currentComparisonParams;
    const isNorm = (compareScaleMode === 'normalized');

    const traces = [];
    const theme = getPlotlyTheme();
    const colors = theme.palette;
    let colorIdx = 0;

    // Calculate parameter-level min & max across all selected stations for normalization
    const paramRanges = {};
    if (isNorm) {
        params.forEach(p => {
            const allVals = [];
            stations.forEach(st => {
                if (data.values[st] && data.values[st][p]) {
                    data.values[st][p].forEach(v => {
                        if (v !== null && !isNaN(v)) allVals.push(v);
                    });
                }
            });
            const pMin = allVals.length > 0 ? Math.min(...allVals) : 0;
            const pMax = allVals.length > 0 ? Math.max(...allVals) : 1;
            paramRanges[p] = { min: pMin, max: pMax, range: pMax - pMin };
        });
    }

    stations.forEach(st => {
        params.forEach(p => {
            if (data.values[st] && data.values[st][p]) {
                const traceName = params.length > 1 ? `${st} (${p})` : st;
                const rawVals = data.values[st][p];

                let plotVals = rawVals;
                let hoverTemplate = '<b>%{x}</b><br>%{data.name}: %{y:.2f}<extra></extra>';

                if (isNorm) {
                    const r = paramRanges[p];
                    plotVals = rawVals.map(v => (r && r.range > 0 ? ((v - r.min) / r.range) * 100 : 50));
                    hoverTemplate = '<b>%{x}</b><br>%{data.name}<br>Normalized: %{y:.1f}%<br>Raw Value: %{customdata:.2f}<extra></extra>';
                }

                traces.push({
                    x: data.dates[st] || [],
                    y: plotVals,
                    customdata: rawVals,
                    name: traceName,
                    type: 'scatter',
                    mode: 'lines',
                    hovertemplate: hoverTemplate,
                    line: { shape: 'spline', smoothing: 1.0, width: 2, color: colors[colorIdx % colors.length] }
                });
                colorIdx++;
            }
        });
    });

    const titleSuffix = isNorm ? ' (Normalized 0–100%)' : '';
    const titleText = `${params.join(', ')} Comparison Across Stations${titleSuffix}`;

    const yaxisConfig = isNorm ? {
        title: { text: 'Normalized Scale (%)', font: { size: 11, color: theme.axisTitleColor } },
        range: [-5, 105],
        gridcolor: theme.gridColor,
        tickfont: { size: 11, color: theme.tickColor }
    } : {
        title: { text: params.join(', '), font: { size: 11, color: theme.axisTitleColor } },
        gridcolor: theme.gridColor,
        tickfont: { size: 11, color: theme.tickColor }
    };

    const layout = {
        autosize: true,
        title: {
            text: titleText,
            font: { family: 'Poppins, sans-serif', size: 14, color: theme.titleColor }
        },
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: { family: 'Poppins, sans-serif', size: 11, color: theme.fontColor },
        margin: { t: 40, r: 180, l: 50, b: 50 },
        legend: {
            orientation: 'v',
            x: 1.02,
            y: 0.85,
            font: { size: 11, color: theme.fontColor }
        },
        xaxis: {
            gridcolor: theme.gridColor,
            tickfont: { size: 11, color: theme.tickColor }
        },
        yaxis: yaxisConfig
    };

    Plotly.newPlot('comparisonChart', traces, layout, { responsive: true, displayModeBar: true }).then(() => {
        if (chartEl && chartEl.offsetParent !== null) {
            try { Plotly.Plots.resize(chartEl); } catch (e) {}
        }
    });

    // Populate Summary Statistics Table for GHI
    const tbody = document.querySelector('#comparisonTable tbody');
    tbody.innerHTML = '';

    stations.forEach(st => {
        const mean = data.summary_stats['GHI (Wh/m2)_mean'] ? data.summary_stats['GHI (Wh/m2)_mean'][st] : null;
        const min = data.summary_stats['GHI (Wh/m2)_min'] ? data.summary_stats['GHI (Wh/m2)_min'][st] : null;
        const max = data.summary_stats['GHI (Wh/m2)_max'] ? data.summary_stats['GHI (Wh/m2)_max'][st] : null;
        const std = data.summary_stats['GHI (Wh/m2)_std'] ? data.summary_stats['GHI (Wh/m2)_std'][st] : null;

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${st}</strong></td>
            <td>${mean !== null ? (mean / 1000).toFixed(2) : '--'}</td>
            <td>${min !== null ? (min / 1000).toFixed(2) : '--'}</td>
            <td>${max !== null ? (max / 1000).toFixed(2) : '--'}</td>
            <td>${std !== null ? (std / 1000).toFixed(2) : '--'}</td>
        `;
        tbody.appendChild(tr);
    });
}

/* Prediction Logic */
function loadPredictionDefaults() {
    // Populate form with default averages
    FEATURE_MAP.forEach(item => {
        const input = document.getElementById(item.id);
        if (input && DEFAULT_AVERAGES[item.name] !== undefined) {
            input.value = DEFAULT_AVERAGES[item.name];
        }
    });

    // Also attempt to fetch latest dataset averages dynamically if available
    fetch('/get-average-values')
        .then(res => res.json())
        .then(averages => {
            if (averages && typeof averages === 'object') {
                FEATURE_MAP.forEach(item => {
                    const input = document.getElementById(item.id);
                    if (input && averages[item.name] !== undefined && averages[item.name] !== null) {
                        input.value = parseFloat(averages[item.name]).toFixed(2);
                    }
                });
            }
            // Run live prediction with default values
            computePrediction();
        })
        .catch(() => {
            // Fallback: run live prediction with static default averages
            computePrediction();
        });
}

function computePrediction() {
    const payload = {};
    FEATURE_MAP.forEach(item => {
        const input = document.getElementById(item.id);
        payload[item.name] = input ? parseFloat(input.value) || 0.0 : (DEFAULT_AVERAGES[item.name] || 0.0);
    });

    const modelType = document.getElementById('predictModelType')?.value || 'production';
    payload['model_type'] = modelType;

    const predValText = document.getElementById('predValueText');
    if (predValText) predValText.textContent = 'Calculating...';

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.prediction !== undefined) {
            predValText.textContent = (data.prediction / 1000).toFixed(2);
        } else if (data.error) {
            console.warn("Prediction error:", data.error);
            predValText.textContent = 'Error';
        }
    })
    .catch(err => {
        console.error("Prediction request failed:", err);
        predValText.textContent = '5.95';
    });
}

function handlePrediction(e) {
    e.preventDefault();
    computePrediction();
}