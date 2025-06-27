document.addEventListener('DOMContentLoaded', () => {
    const map = L.map('map', {
        center: [24.0, 45.0],
        zoom: 5,
        zoomControl: true
    });

    let markers = {};
    let gradientCircles = [];
    let isGradientView = false;
    const colorPalette = ['#1E5631', '#FF5733', '#FFC107', '#28A745', '#007BFF', '#6F42C1', '#E83E8C'];
    let ghiThresholds = { low: 4500, high: 6800 }; // Fallback values

    const baseLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    const fallbackLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles © Esri'
    });

    map.on('tileerror', () => {
        console.warn('Tile loading failed, switching to fallback layer');
        map.removeLayer(baseLayer);
        fallbackLayer.addTo(map);
    });

    // Initialize Bootstrap tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));

    // Initialize Select2 for multi-select dropdowns
    $('#parameterSelect').select2({
        placeholder: "Select Parameters",
        width: '100%'
    });

    $('#compareStations').select2({
        placeholder: "Select Stations",
        width: '100%'
    });

    $('#compareParams').select2({
        placeholder: "Select Parameters",
        width: '100%'
    });

    function populateStationSelects() {
        fetch('/get-station-names')
            .then(response => response.json())
            .then(data => {
                const stationSelect = document.getElementById('stationSelect');
                const compareStations = $('#compareStations');

                data.stations.forEach(station => {
                    const option = document.createElement('option');
                    option.value = station;
                    option.textContent = station;
                    stationSelect.appendChild(option);
                    const compareOption = new Option(station, station, false, false);
                    compareStations.append(compareOption);
                });

                compareStations.trigger('change');

                if (data.stations.length > 0) {
                    stationSelect.value = data.stations[0];
                    loadStationDetails(data.stations[0]);
                }
            })
            .catch(error => console.error('Error fetching station names:', error));
    }

    function loadMapData() {
        const statusDiv = document.getElementById('mapStatus') || document.createElement('div');
        statusDiv.id = 'mapStatus';
        statusDiv.className = 'map-status';
        statusDiv.textContent = 'Loading map data...';
        document.getElementById('map').parentNode.insertBefore(statusDiv, document.getElementById('map').nextSibling);

        fetch('/get-ghi-thresholds')
            .then(response => response.json())
            .then(thresholds => {
                ghiThresholds = thresholds;
                console.log('Updated GHI Thresholds:', ghiThresholds);
                
                fetch('/map-data')
                    .then(response => {
                        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
                        return response.json();
                    })
                    .then(data => {
                        statusDiv.textContent = '';
                        Object.values(markers).forEach(marker => map.removeLayer(marker));
                        gradientCircles.forEach(circle => map.removeLayer(circle));
                        markers = {};
                        gradientCircles = [];

                        data.forEach(station => {
                            if (!station.latitude || !station.longitude || !station.avg_ghi) {
                                console.warn(`Invalid data for station ${station.station_name}:`, station);
                                return;
                            }

                            const marker = L.marker([station.latitude, station.longitude], {
                                icon: L.icon({ iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png' })
                            })
                            .bindPopup(`<b>${station.station_name}</b><br>GHI: ${station.avg_ghi.toFixed(2)} Wh/m²`)
                            .addTo(map);

                            marker.on('mouseover', () => {
                                marker.setIcon(L.icon({ iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png' }));
                            });
                            marker.on('mouseout', () => {
                                marker.setIcon(L.icon({ iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png' }));
                            });
                            marker.on('click', () => {
                                document.getElementById('stationSelect').value = station.station_name;
                                loadStationDetails(station.station_name);
                            });

                            markers[station.station_name] = marker;

                            let circleColor;
                            if (station.avg_ghi > ghiThresholds.high) {
                                circleColor = '#28A745';
                            } else if (station.avg_ghi >= ghiThresholds.low) {
                                circleColor = '#FFC107';
                            } else {
                                circleColor = '#FF5733';
                            }

                            const circle = L.circle([station.latitude, station.longitude], {
                                radius: 20000,
                                color: circleColor,
                                fillColor: circleColor,
                                fillOpacity: 0.5,
                                weight: 1
                            }).bindPopup(`<b>${station.station_name}</b><br>GHI: ${station.avg_ghi.toFixed(2)} Wh/m²`);

                            gradientCircles.push(circle);
                        });
                    })
                    .catch(error => {
                        console.error('Error loading map data:', error);
                        statusDiv.textContent = `Error loading map: ${error.message}. Check server connection or data.`;
                    });
            })
            .catch(error => {
                console.error('Error loading thresholds:', error);
                fetch('/map-data')
                    .then(response => response.json())
                    .then(data => {
                        statusDiv.textContent = '';
                        Object.values(markers).forEach(marker => map.removeLayer(marker));
                        gradientCircles.forEach(circle => map.removeLayer(circle));
                        markers = {};
                        gradientCircles = [];

                        data.forEach(station => {
                            if (!station.latitude || !station.longitude || !station.avg_ghi) {
                                console.warn(`Invalid data for station ${station.station_name}:`, station);
                                return;
                            }

                            const marker = L.marker([station.latitude, station.longitude], {
                                icon: L.icon({ iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png' })
                            })
                            .bindPopup(`<b>${station.station_name}</b><br>GHI: ${station.avg_ghi.toFixed(2)} Wh/m²`)
                            .addTo(map);

                            marker.on('mouseover', () => {
                                marker.setIcon(L.icon({ iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png' }));
                            });
                            marker.on('mouseout', () => {
                                marker.setIcon(L.icon({ iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png' }));
                            });
                            marker.on('click', () => {
                                document.getElementById('stationSelect').value = station.station_name;
                                loadStationDetails(station.station_name);
                            });

                            markers[station.station_name] = marker;

                            let circleColor;
                            if (station.avg_ghi > ghiThresholds.high) {
                                circleColor = '#28A745';
                            } else if (station.avg_ghi >= ghiThresholds.low) {
                                circleColor = '#FFC107';
                            } else {
                                circleColor = '#FF5733';
                            }

                            const circle = L.circle([station.latitude, station.longitude], {
                                radius: 20000,
                                color: circleColor,
                                fillColor: circleColor,
                                fillOpacity: 0.5,
                                weight: 1
                            }).bindPopup(`<b>${station.station_name}</b><br>GHI: ${station.avg_ghi.toFixed(2)} Wh/m²`);

                            gradientCircles.push(circle);
                        });
                    })
                    .catch(error => {
                        console.error('Error loading map data in fallback:', error);
                        statusDiv.textContent = `Error loading map: ${error.message}. Check server connection or data.`;
                    });
            });
    }

    document.getElementById('toggleHeatmap').addEventListener('click', () => {
        isGradientView = !isGradientView;
        if (isGradientView) {
            Object.values(markers).forEach(marker => map.removeLayer(marker));
            gradientCircles.forEach(circle => circle.addTo(map));
            document.getElementById('toggleHeatmap').textContent = 'Switch to Normal Map';
        } else {
            gradientCircles.forEach(circle => map.removeLayer(circle));
            Object.values(markers).forEach(marker => marker.addTo(map));
            document.getElementById('toggleHeatmap').textContent = 'Toggle Gradient View';
        }
    });

    function loadStationDetails(stationName) {
        fetch(`/station-details?station=${stationName}`)
            .then(response => response.json())
            .then(data => {
                const detailsDiv = document.getElementById('stationDetails');
                detailsDiv.innerHTML = `
                    <p><strong>Station:</strong> ${stationName}</p>
                    <p><strong>Latitude:</strong> ${data.details.latitude}</p>
                    <p><strong>Longitude:</strong> ${data.details.longitude}</p>
                `;

                const paramSelect = $('#parameterSelect');
                paramSelect.empty();
                
                const availableParams = Object.keys(data.monthly_chart_data.data);
                
                availableParams.forEach((param, index) => {
                    const isSelected = index === 0;
                    const option = new Option(param, param, isSelected, isSelected);
                    paramSelect.append(option);
                });
                
                paramSelect.trigger('change');
                
                const yearSelect = document.getElementById('mapViewYear');
                if (yearSelect && yearSelect.options.length > 0) {
                    yearSelect.selectedIndex = 0;
                }
                
                updateMonthlyChart(stationName);
            })
            .catch(error => console.error('Error loading station details:', error));
    }

    function updateMonthlyChart(stationName) {
        let selectedParams = $('#parameterSelect').val();
        if (!selectedParams || selectedParams.length === 0) {
            const firstParam = $('#parameterSelect option:first').val();
            if (firstParam) {
                selectedParams = [firstParam];
                $('#parameterSelect').val([firstParam]).trigger('change');
            }
        }
        
        const yearSelect = document.getElementById('mapViewYear');
        let year = yearSelect?.value || null;
        
        if (!selectedParams || selectedParams.length === 0) {
            document.getElementById('monthlyChart').innerHTML = '<p>Please select at least one parameter</p>';
            return;
        }
        
        document.getElementById('monthlyChart').innerHTML = '<p>Loading chart...</p>';
        const queryParams = new URLSearchParams({ station: stationName, year: year || '' });
        
        fetch(`/station-details?${queryParams}`)
            .then(response => response.json())
            .then(data => {
                let xData, traces;
                const yearDisplay = year || 'All Years';
    
                if (!year) {
                    const yearlyData = {};
                    selectedParams.forEach(param => yearlyData[param] = []);
                    const years = ['2017', '2018', '2019', '2020', '2021'];
    
                    Promise.all(years.map(yr => fetch(`/station-details?station=${stationName}&year=${yr}`).then(res => res.json())))
                        .then(results => {
                            results.forEach(d => {
                                selectedParams.forEach(param => {
                                    const yearlyAvg = d.monthly_chart_data.data[param].reduce((a, b) => a + b, 0) / 12;
                                    yearlyData[param].push(yearlyAvg);
                                });
                            });
    
                            xData = years;
                            traces = selectedParams.map((param, index) => [
                                {
                                    x: xData,
                                    y: yearlyData[param],
                                    type: 'bar',
                                    name: param,
                                    hoverinfo: 'x+y',
                                    marker: { color: colorPalette[index % colorPalette.length] },
                                    width: 0.4
                                },
                                {
                                    x: xData,
                                    y: yearlyData[param],
                                    type: 'scatter',
                                    mode: 'lines+markers',
                                    name: `${param} Trend`,
                                    line: { color: colorPalette[index % colorPalette.length], width: 2 },
                                    marker: { size: 8 }
                                }
                            ]).flat();
                            
                            const layout = {
                                title: `Data for ${stationName} (${yearDisplay})`,
                                barmode: 'overlay',
                                height: 600,
                                legend: {
                                    orientation: 'h',
                                    y: -0.2,
                                    xanchor: 'center',
                                    x: 0.5
                                },
                                xaxis: {
                                    title: '',
                                    tickmode: 'linear',
                                    automargin: true
                                },
                                yaxis: {
                                    title: '',
                                    automargin: true
                                },
                                margin: {
                                    t: 50,
                                    b: 100
                                }
                            };
    
                            Plotly.newPlot('monthlyChart', traces, layout);
                        });
                } else {
                    xData = data.monthly_chart_data.months;
                    traces = selectedParams.map((param, index) => [
                        {
                            x: xData,
                            y: data.monthly_chart_data.data[param],
                            type: 'bar',
                            name: param,
                            hoverinfo: 'x+y',
                            marker: { color: colorPalette[index % colorPalette.length] },
                            width: 0.4
                        },
                        {
                            x: xData,
                            y: data.monthly_chart_data.data[param],
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: `${param} Trend`,
                            line: { color: colorPalette[index % colorPalette.length], width: 2 },
                            marker: { size: 8 }
                        }
                    ]).flat();
    
                    const layout = {
                        title: `Data for ${stationName} (${yearDisplay})`,
                        barmode: 'overlay',
                        height: 600,
                        legend: {
                            orientation: 'h',
                            y: -0.2,
                            xanchor: 'center',
                            x: 0.5
                        },
                        xaxis: {
                            title: '',
                            tickmode: 'linear',
                            automargin: true
                        },
                        yaxis: {
                            title: '',
                            automargin: true
                        },
                        margin: {
                            t: 50,
                            b: 100
                        }
                    };
    
                    Plotly.newPlot('monthlyChart', traces, layout);
                }
            })
            .catch(error => {
                console.error('Error updating monthly chart:', error);
                document.getElementById('monthlyChart').innerHTML = '<p>Error loading chart data</p>';
            });
    }

    function loadStationComparison() {
        const selectedStations = $('#compareStations').val();
        const selectedParams = $('#compareParams').val();
        const year = document.getElementById('comparisonYear')?.value || null;

        if (!selectedStations.length || !selectedParams.length) {
            document.getElementById('comparisonCharts').innerHTML = '<p>Please select stations and parameters</p>';
            return;
        }

        document.getElementById('comparisonCharts').innerHTML = '<p>Loading comparison...</p>';
        fetch('/station-comparison', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stations: selectedStations, params: selectedParams, year: year })
        })
        .then(response => response.json())
        .then(data => {
            const chartsDiv = document.getElementById('comparisonCharts');
            chartsDiv.innerHTML = '';

            selectedParams.forEach(param => {
                const traces = selectedStations.map((station, index) => ({
                    x: data.dates[station],
                    y: data.values[station][param],
                    type: 'scatter',
                    mode: 'lines',
                    name: station,
                    line: { color: colorPalette[index % colorPalette.length] }
                }));

                const div = document.createElement('div');
                div.id = `chart-${param}`;
                chartsDiv.appendChild(div);

                Plotly.newPlot(`chart-${param}`, traces, {
                    title: `${param} Comparison Across Stations`,
                    xaxis: { title: '' },
                    yaxis: { title: '' },
                    height: 500
                });
            });

            updateSummaryTable(data);
        })
        .catch(error => {
            console.error('Error loading comparison:', error);
            document.getElementById('comparisonCharts').innerHTML = '<p>Error loading comparison</p>';
        });
    }

    function updateSummaryTable(data) {
        const selectedStations = $('#compareStations').val();
        const statsTable = document.getElementById('summaryStats');
        
        let tableHtml = `<table class="summary-table"><thead><tr><th>Station</th><th>GHI Mean</th><th>GHI Min</th><th>GHI Max</th><th>GHI Std</th></tr></thead><tbody>`;

        selectedStations.forEach(station => {
            tableHtml += `<tr><td>${station}</td>`;
            ['mean', 'min', 'max', 'std'].forEach(stat => {
                const key = `GHI (Wh/m2)_${stat}`;
                const value = data.summary_stats[key]?.[station];
                tableHtml += `<td>${value !== undefined && value !== null ? value.toFixed(2) : 'N/A'}</td>`;
            });
            tableHtml += '</tr>';
        });

        tableHtml += '</tbody></table>';
        statsTable.innerHTML = tableHtml;
    }

    fetch('/data-analysis')
        .then(response => response.json())
        .then(data => {
            const paramSelect = $('#compareParams');
            
            const params = Object.keys(data.summary_stats)
                .filter(key => key.endsWith('_mean'))
                .map(key => key.replace('_mean', ''));
            
            params.forEach((param, index) => {
                const isDefault = index === 0;
                const option = new Option(param, param, isDefault, isDefault);
                paramSelect.append(option);
            });
            
            paramSelect.trigger('change');

            const ranges = {};
            Object.keys(data.summary_stats).forEach(key => {
                const param = key.replace(/_(mean|min|max|std)$/, '');
                if (!ranges[param]) ranges[param] = {};
                if (key.endsWith('_min')) ranges[param].min = Math.min(...Object.values(data.summary_stats[key]));
                if (key.endsWith('_max')) ranges[param].max = Math.max(...Object.values(data.summary_stats[key]));
                if (key.endsWith('_mean')) ranges[param].mean = Object.values(data.summary_stats[key]).reduce((a, b) => a + b, 0) / Object.keys(data.summary_stats[key]).length;
                if (key.endsWith('_std')) ranges[param].std = Object.values(data.summary_stats[key]).reduce((a, b) => a + b, 0) / Object.keys(data.summary_stats[key]).length;
            });

            document.getElementById('tempRange').textContent = `Range: ${ranges['Air Temperature (C°)']?.min.toFixed(2)} - ${ranges['Air Temperature (C°)']?.max.toFixed(2)} °C`;
            document.getElementById('tempUncertaintyRange').textContent = `Range: ${ranges['Air Temperature Uncertainty (C°)']?.min.toFixed(2)} - ${ranges['Air Temperature Uncertainty (C°)']?.max.toFixed(2)} °C`;
            document.getElementById('windDirRange').textContent = `Range: ${ranges['Wind Direction at 3m (°N)']?.min.toFixed(2)} - ${ranges['Wind Direction at 3m (°N)']?.max.toFixed(2)} °N`;
            document.getElementById('windDirUncertaintyRange').textContent = `Range: ${ranges['Wind Direction at 3m Uncertainty (°N)']?.min.toFixed(2)} - ${ranges['Wind Direction at 3m Uncertainty (°N)']?.max.toFixed(2)} °N`;
            document.getElementById('windRange').textContent = `Range: ${ranges['Wind Speed at 3m (m/s)']?.min.toFixed(2)} - ${ranges['Wind Speed at 3m (m/s)']?.max.toFixed(2)} m/s`;
            document.getElementById('windSpeedUncertaintyRange').textContent = `Range: ${ranges['Wind Speed at 3m Uncertainty (m/s)']?.min.toFixed(2)} - ${ranges['Wind Speed at 3m Uncertainty (m/s)']?.max.toFixed(2)} m/s`;
            document.getElementById('windSpeedStdDevRange').textContent = `Range: ${ranges['Wind Speed at 3m (std dev) (m/s)']?.min.toFixed(2)} - ${ranges['Wind Speed at 3m (std dev) (m/s)']?.max.toFixed(2)} m/s`;
            document.getElementById('dhiRange').textContent = `Range: ${ranges['DHI (Wh/m2)']?.min.toFixed(2)} - ${ranges['DHI (Wh/m2)']?.max.toFixed(2)} Wh/m²`;
            document.getElementById('dhiUncertaintyRange').textContent = `Range: ${ranges['DHI Uncertainty (Wh/m2)']?.min.toFixed(2)} - ${ranges['DHI Uncertainty (Wh/m2)']?.max.toFixed(2)} Wh/m²`;
            document.getElementById('dhiStdDevRange').textContent = `Range: ${ranges['Standard Deviation DHI (Wh/m2)']?.min.toFixed(2)} - ${ranges['Standard Deviation DHI (Wh/m2)']?.max.toFixed(2)} Wh/m²`;
            document.getElementById('dniRange').textContent = `Range: ${ranges['DNI (Wh/m2)']?.min.toFixed(2)} - ${ranges['DNI (Wh/m2)']?.max.toFixed(2)} Wh/m²`;
            document.getElementById('dniUncertaintyRange').textContent = `Range: ${ranges['DNI Uncertainty (Wh/m2)']?.min.toFixed(2)} - ${ranges['DNI Uncertainty (Wh/m2)']?.max.toFixed(2)} Wh/m²`;
            document.getElementById('dniStdDevRange').textContent = `Range: ${ranges['Standard Deviation DNI (Wh/m2)']?.min.toFixed(2)} - ${ranges['Standard Deviation DNI (Wh/m2)']?.max.toFixed(2)} Wh/m²`;
            document.getElementById('ghiUncertaintyRange').textContent = `Range: ${ranges['GHI Uncertainty (Wh/m2)']?.min.toFixed(2)} - ${ranges['GHI Uncertainty (Wh/m2)']?.max.toFixed(2)} Wh/m²`;
            document.getElementById('ghiStdDevRange').textContent = `Range: ${ranges['Standard Deviation GHI (Wh/m2)']?.min.toFixed(2)} - ${ranges['Standard Deviation GHI (Wh/m2)']?.max.toFixed(2)} Wh/m²`;
            document.getElementById('peakWindSpeedRange').textContent = `Range: ${ranges['Peak Wind Speed at 3m (m/s)']?.min.toFixed(2)} - ${ranges['Peak Wind Speed at 3m (m/s)']?.max.toFixed(2)} m/s`;
            document.getElementById('peakWindSpeedUncertaintyRange').textContent = `Range: ${ranges['Peak Wind Speed at 3m Uncertainty (m/s)']?.min.toFixed(2)} - ${ranges['Peak Wind Speed at 3m Uncertainty (m/s)']?.max.toFixed(2)} m/s`;
            document.getElementById('humidityRange').textContent = `Range: ${ranges['Relative Humidity (%)']?.min.toFixed(2)} - ${ranges['Relative Humidity (%)']?.max.toFixed(2)} %`;
            document.getElementById('humidityUncertaintyRange').textContent = `Range: ${ranges['Relative Humidity Uncertainty (%)']?.min.toFixed(2)} - ${ranges['Relative Humidity Uncertainty (%)']?.max.toFixed(2)} %`;
            document.getElementById('pressureRange').textContent = `Range: ${ranges['Barometric Pressure (mB (hPa equiv))']?.min.toFixed(2)} - ${ranges['Barometric Pressure (mB (hPa equiv))']?.max.toFixed(2)} hPa`;
            document.getElementById('pressureUncertaintyRange').textContent = `Range: ${ranges['Barometric Pressure Uncertainty (mB (hPa equiv))']?.min.toFixed(2)} - ${ranges['Barometric Pressure Uncertainty (mB (hPa equiv))']?.max.toFixed(2)} hPa`;
        })
        .catch(error => console.error('Error fetching data analysis:', error));

    document.getElementById('predictionForm').addEventListener('submit', (e) => {
        e.preventDefault();
        const predictionResultDiv = document.getElementById('predictionResult');

        const inputData = {
            'Air Temperature (C°)': parseFloat(document.getElementById('airTemp').value),
            'Air Temperature Uncertainty (C°)': parseFloat(document.getElementById('airTempUncertainty').value),
            'Wind Direction at 3m (°N)': parseFloat(document.getElementById('windDir').value),
            'Wind Direction at 3m Uncertainty (°N)': parseFloat(document.getElementById('windDirUncertainty').value),
            'Wind Speed at 3m (m/s)': parseFloat(document.getElementById('windSpeed').value),
            'Wind Speed at 3m Uncertainty (m/s)': parseFloat(document.getElementById('windSpeedUncertainty').value),
            'Wind Speed at 3m (std dev) (m/s)': parseFloat(document.getElementById('windSpeedStdDev').value),
            'DHI (Wh/m2)': parseFloat(document.getElementById('dhi').value),
            'DHI Uncertainty (Wh/m2)': parseFloat(document.getElementById('dhiUncertainty').value),
            'Standard Deviation DHI (Wh/m2)': parseFloat(document.getElementById('dhiStdDev').value),
            'DNI (Wh/m2)': parseFloat(document.getElementById('dni').value),
            'DNI Uncertainty (Wh/m2)': parseFloat(document.getElementById('dniUncertainty').value),
            'Standard Deviation DNI (Wh/m2)': parseFloat(document.getElementById('dniStdDev').value),
            'GHI Uncertainty (Wh/m2)': parseFloat(document.getElementById('ghiUncertainty').value),
            'Standard Deviation GHI (Wh/m2)': parseFloat(document.getElementById('ghiStdDev').value),
            'Peak Wind Speed at 3m (m/s)': parseFloat(document.getElementById('peakWindSpeed').value),
            'Peak Wind Speed at 3m Uncertainty (m/s)': parseFloat(document.getElementById('peakWindSpeedUncertainty').value),
            'Relative Humidity (%)': parseFloat(document.getElementById('humidity').value),
            'Relative Humidity Uncertainty (%)': parseFloat(document.getElementById('humidityUncertainty').value),
            'Barometric Pressure (mB (hPa equiv))': parseFloat(document.getElementById('pressure').value),
            'Barometric Pressure Uncertainty (mB (hPa equiv))': parseFloat(document.getElementById('pressureUncertainty').value)
        };

        // Validate inputs
        for (const [key, value] of Object.entries(inputData)) {
            if (isNaN(value) || value === null || value === undefined) {
                predictionResultDiv.innerHTML = `
                    <div class="alert alert-danger">Invalid input for ${key}. Please enter a valid number.</div>
                `;
                return;
            }
        }

        predictionResultDiv.innerHTML = '<p>Predicting...</p>';
        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.prediction !== undefined) {
                const prediction = parseFloat(data.prediction).toFixed(2);
                let alertClass;
                if (prediction > ghiThresholds.high) {
                    alertClass = 'alert-success';
                } else if (prediction < ghiThresholds.low) {
                    alertClass = 'alert-danger';
                } else {
                    alertClass = 'alert-warning';
                }
                predictionResultDiv.innerHTML = `<div class="alert ${alertClass}">Predicted GHI: ${prediction} Wh/m²</div>`;
            } else {
                throw new Error(data.error || 'Unknown prediction error');
            }
        })
        .catch(error => {
            console.error('Error predicting GHI:', error);
            predictionResultDiv.innerHTML = `
                <div class="alert alert-danger">Prediction Error: ${error.message}</div>
            `;
        });
    });

    // Add event listener for Load Average Values button
    document.getElementById('loadAveragesBtn').addEventListener('click', function() {
        fetch('/get-average-values')
            .then(response => response.json())
            .then(data => {
                // Map feature names to input field IDs
                const fieldMapping = {
                    'Air Temperature (C°)': 'airTemp',
                    'Air Temperature Uncertainty (C°)': 'airTempUncertainty',
                    'Wind Direction at 3m (°N)': 'windDir',
                    'Wind Direction at 3m Uncertainty (°N)': 'windDirUncertainty',
                    'Wind Speed at 3m (m/s)': 'windSpeed',
                    'Wind Speed at 3m Uncertainty (m/s)': 'windSpeedUncertainty',
                    'Wind Speed at 3m (std dev) (m/s)': 'windSpeedStdDev',
                    'DHI (Wh/m2)': 'dhi',
                    'DHI Uncertainty (Wh/m2)': 'dhiUncertainty',
                    'Standard Deviation DHI (Wh/m2)': 'dhiStdDev',
                    'DNI (Wh/m2)': 'dni',
                    'DNI Uncertainty (Wh/m2)': 'dniUncertainty',
                    'Standard Deviation DNI (Wh/m2)': 'dniStdDev',
                    'GHI Uncertainty (Wh/m2)': 'ghiUncertainty',
                    'Standard Deviation GHI (Wh/m2)': 'ghiStdDev',
                    'Peak Wind Speed at 3m (m/s)': 'peakWindSpeed',
                    'Peak Wind Speed at 3m Uncertainty (m/s)': 'peakWindSpeedUncertainty',
                    'Relative Humidity (%)': 'humidity',
                    'Relative Humidity Uncertainty (%)': 'humidityUncertainty',
                    'Barometric Pressure (mB (hPa equiv))': 'pressure',
                    'Barometric Pressure Uncertainty (mB (hPa equiv))': 'pressureUncertainty'
                };

                // Fill each input field with the corresponding average value
                Object.keys(data).forEach(key => {
                    const inputId = fieldMapping[key];
                    const input = document.getElementById(inputId);
                    if (input && data[key] !== null) {
                        input.value = data[key].toFixed(2);
                    }
                });
            })
            .catch(error => {
                console.error('Error loading average values:', error);
                alert('Failed to load average values: ' + error.message);
            });
    });

    $('#parameterSelect').on('change', () => {
        const station = document.getElementById('stationSelect').value;
        updateMonthlyChart(station);
    });

    $('#compareStations, #compareParams').on('change', loadStationComparison);

    document.getElementById('stationSelect').addEventListener('change', (e) => loadStationDetails(e.target.value));
    document.getElementById('mapViewYear').addEventListener('change', () => {
        const station = document.getElementById('stationSelect').value;
        updateMonthlyChart(station);
    });
    document.getElementById('comparisonYear').addEventListener('change', loadStationComparison);

    populateStationSelects();
    loadMapData();
});