"""
Plot interactive HTML map for the activated events.

- add detailed description later .
"""

from __future__ import annotations

import colorsys
import hashlib
import html
import ipaddress
import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import requests

from .settings import Settings

# -----------------------------------------------------------------------------
# generate and save an interactive HTML map
# -----------------------------------------------------------------------------


Event = Dict[str, Any]
log = logging.getLogger(__name__)

FAMILY_HUES = {
    "flood": 210 / 360,  # blue
    "hurricane": 120 / 360,  # green
    "storm": 10 / 360,  # red/orange
    "thunderstorm": 270 / 360,  # purple
    "earthquake": 45 / 360,  # amber
}

TRUSTED_URL_SUFFIXES = (
    "weather.gov",
    "noaa.gov",
    "usgs.gov",
)
MAX_GEOJSON_BYTES = 2 * 1024 * 1024  # 2 MiB


def _is_url(s: str) -> bool:
    parsed = urlparse(s)
    return parsed.scheme in ("http", "https")


def _host_is_trusted(hostname: str) -> bool:
    host = hostname.lower().strip(".")
    return any(
        host == suffix or host.endswith(f".{suffix}") for suffix in TRUSTED_URL_SUFFIXES
    )


def _host_resolves_public(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for _, _, _, _, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Only https URLs are allowed for AOI downloads")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    if not _host_is_trusted(parsed.hostname):
        raise ValueError(f"Untrusted host for AOI download: {parsed.hostname}")
    if not _host_resolves_public(parsed.hostname):
        raise ValueError(f"Host resolves to a non-public address: {parsed.hostname}")


def _detect_family(event_type: str) -> str:
    s = event_type.lower()
    if "flood" in s:
        return "flood"
    if "hurricane" in s:
        return "hurricane"
    if "earthquake" in s:
        return "earthquake"
    if "thunderstorm" in s:
        return "thunderstorm"
    # keep this order otherwise thunderstorm events
    # will be categorized as storm
    if "storm" in s:
        return "storm"
    return "storm"


def _color_from_event_type(event_type: str) -> str:
    family = _detect_family(event_type)
    base_hue = FAMILY_HUES[family]

    hash_hex = hashlib.md5(event_type.encode()).hexdigest()

    # Use more hash bits for stronger variation
    h_variation = int(hash_hex[:2], 16) / 255.0  # 0–1
    l_variation = int(hash_hex[2:4], 16) / 255.0  # 0–1

    # --- HUE variation (±15 degrees) ---
    hue_offset = (h_variation - 0.5) * (30 / 360)  # ±15°
    hue = (base_hue + hue_offset) % 1.0

    # --- LIGHTNESS variation (wide range) ---
    lightness = 0.35 + 0.35 * l_variation  # 0.35–0.70

    saturation = 0.75

    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255),
        int(g * 255),
        int(b * 255),
    )


def _magnitude_to_radius(mag: float | None) -> float:
    """Map earthquake magnitude to a CircleMarker pixel radius.

    Uses a super-linear curve so higher magnitudes stand out visually,
    clamped so M<3 quakes stay visible and M>7.5 quakes don't dominate.
    """
    if mag is None:
        return 5.0
    base = max(0.0, float(mag) - 1.0)
    return max(4.0, min(22.0, 1.6 * base**1.4))


def _generate_events_html_map(
    settings: "Settings",
    events: dict[str, list["Event"]],
    file_dir: "Path",
):
    """
    Create an interactive map displaying activated events,
    grouped by routing key, and enabling the user to draw a bounding box.
    """

    import folium
    from branca.element import MacroElement
    from folium.features import GeoJson
    from folium.plugins import Draw
    from jinja2 import Template
    from shapely.geometry import MultiPolygon, Point

    class MapDashboardJS(MacroElement):
        def __init__(self):
            super().__init__()
            self._template = Template(
                """
                {% macro html(this, kwargs) %}
                <style>
                    #control-panel {
                        position: absolute; top: 20px; left: 60px; z-index: 1000;
                        background: rgba(255, 255, 255, 0.95); padding: 12px 20px;
                        border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                        display: flex; flex-direction: column; gap: 12px;
                        border: 2px solid #374151; font-family: Arial, sans-serif;
                        max-width: 900px;
                    }
                    .main-row {
                        display: flex; flex-direction: row; gap: 12px;
                        align-items: center; width: 100%; justify-content: flex-start;
                        flex-wrap: nowrap;
                    }
                    .panel-label {
                        font-size: 14px; font-weight: bold; color: #374151;
                        white-space: nowrap;
                    }
                    .settings-panel {
                        display: flex; gap: 12px; align-items: center; padding: 10px;
                        background: #f3f4f6; border-radius: 6px; flex-wrap: wrap;
                        border: 1px dashed #9ca3af; width: 100%; box-sizing: border-box;
                    }
                    #control-panel select, #control-panel input {
                        border: 1px solid #9ca3af; border-radius: 4px; padding: 4px 8px;
                        font-size: 13px; background-color: #ffffff; height: 34px;
                        box-sizing: border-box;
                    }
                    .multi-dropdown {
                        position: relative; display: inline-block;
                    }
                    .multi-dropdown-btn {
                        position: relative;
                        border: 1px solid #9ca3af; border-radius: 4px;
                        padding: 4px 24px 4px 8px; font-size: 13px;
                        background-color: #ffffff; height: 34px; cursor: pointer;
                        width: 160px; text-align: left; overflow: hidden;
                        text-overflow: ellipsis; white-space: nowrap;
                    }
                    .multi-dropdown-btn::after {
                        content: " ▾"; position: absolute; right: 8px; top: 8px;
                    }
                    .multi-dropdown-list {
                        display: none; position: absolute; top: 36px; left: 0;
                        background: #ffffff; border: 1px solid #9ca3af;
                        border-radius: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        z-index: 9999; min-width: 100%; padding: 4px 0;
                        max-height: 250px; overflow-y: auto;
                    }
                    .multi-dropdown-list.open {
                        display: block;
                    }
                    .multi-dropdown-list label {
                        display: flex; align-items: center; gap: 8px; padding: 6px 12px;
                        font-size: 13px; cursor: pointer; white-space: nowrap;
                    }
                    .multi-dropdown-list label:hover {
                        background-color: #f3f4f6;
                    }
                    #action-btn {
                        background-color: #4b5563; color: white; border: none;
                        padding: 0 25px; border-radius: 4px; font-weight: bold;
                        font-size: 14px; cursor: pointer; height: 34px;
                        white-space: nowrap; transition: background-color 0.2s;
                    }
                    #action-btn:hover {
                        background-color: #374151;
                    }
                    .dynamic-opt {
                        display: flex; align-items: center; gap: 6px; font-size: 13px;
                        background: #e5e7eb; padding: 4px 8px; border-radius: 4px;
                        border: 1px solid #d1d5db;
                    }
                    .radio-group {
                        display: flex; gap: 15px; align-items: center; font-size: 13px;
                        background: #ffffff; padding: 4px 10px; border-radius: 4px;
                        border: 1px solid #9ca3af; height: 34px; box-sizing: border-box;
                    }
                    .radio-group label {
                        display: flex; align-items: center; gap: 4px; cursor: pointer;
                    }
                </style>

                <div id="control-panel">
                    <div class="main-row">
                        <span class="panel-label">Workflow:</span>

                        <div class="multi-dropdown" id="func-dropdown">
                            <button type="button" class="multi-dropdown-btn"
                                id="func-btn">All Functionality</button>
                            <div class="multi-dropdown-list" id="func-list">
                                <label><input type="checkbox" value="all" checked>
                                    All</label>
                                <label><input type="checkbox" value="overpasses">
                                    Overpasses</label>
                                <label><input type="checkbox" value="opera_search">
                                    Opera Search</label>
                                <label><input type="checkbox" value="disasters">
                                    Disasters Workflow</label>
                            </div>
                        </div>

                        <div class="multi-dropdown" id="prod-dropdown">
                            <button type="button" class="multi-dropdown-btn"
                                id="prod-btn">All Products</button>
                            <div class="multi-dropdown-list" id="prod-list">
                                <label><input type="checkbox" value="all" checked>
                                    All Products</label>
                                <label><input type="checkbox"
                                    value="OPERA_L3_DSWX-HLS_V1"> DSWX-HLS</label>
                                <label><input type="checkbox"
                                    value="OPERA_L3_DSWX-S1_V1"> DSWX-S1</label>
                                <label><input type="checkbox"
                                    value="OPERA_L3_DIST-ALERT-HLS_V1">
                                    DIST-ALERT-HLS</label>
                                <label><input type="checkbox"
                                    value="OPERA_L3_DIST-ANN-HLS_V1">
                                    DIST-ANN-HLS</label>
                                <label><input type="checkbox"
                                    value="OPERA_L2_RTC-S1_V1"> RTC-S1</label>
                                <label><input type="checkbox"
                                    value="OPERA_L2_CSLC-S1_V1"> CSLC-S1</label>
                                <label><input type="checkbox"
                                    value="OPERA_L3_DISP-S1_V1"> DISP-S1</label>
                            </div>
                        </div>

                        <div style="flex-grow: 1;"></div>
                        <button id="action-btn">SEARCH</button>
                    </div>

                    <div id="next-pass-panel" class="settings-panel">
                        <span class="panel-label">↳ Next Pass:</span>

                        <div class="multi-dropdown" id="sat-dropdown">
                            <button type="button" class="multi-dropdown-btn"
                                id="sat-btn" style="width: 140px;">All Satellites
                            </button>
                            <div class="multi-dropdown-list" id="sat-list">
                                <label><input type="checkbox" value="all" checked>
                                    All Satellites</label>
                                <label><input type="checkbox" value="sentinel-1">
                                    Sentinel-1</label>
                                <label><input type="checkbox" value="sentinel-2">
                                    Sentinel-2</label>
                                <label><input type="checkbox" value="landsat">
                                    Landsat</label>
                                <label><input type="checkbox" value="nisar">
                                    NISAR</label>
                            </div>
                        </div>

                        <input type="number" id="np_lookback"
                            placeholder="Lookback (days)" min="1" max="30"
                            style="width: 130px;">

                        <select id="drcs_enabled" onchange="toggleDrcsDate(this.value)"
                            style="width: 100px;">
                            <option value="no">DRCS: No</option>
                            <option value="yes">DRCS: Yes</option>
                        </select>
                        <input type="text" id="drcs_event_date"
                            placeholder="Event: YYYY-MM-DDTHH:MM" disabled
                            style="width: 170px;">
                    </div>

                    <div id="disasters-panel" class="settings-panel"
                        style="display: none;">
                        <span class="panel-label">↳ Disasters:</span>

                        <select id="dis_action"
                            style="font-weight: bold; background-color: #e5e7eb;">
                            <option value="run" selected>
                                Full Pipeline (Maps & Layouts)
                            </option>
                            <option value="mosaic">Mosaic GeoTIFFs Only</option>
                            <option value="download">Download Granules Only</option>
                        </select>

                        <div class="radio-group">
                            <label><input type="radio" name="date_strat" value="range"
                                checked onchange="toggleDisDateStrat()"> Range</label>
                            <label><input type="radio" name="date_strat" value="single"
                                onchange="toggleDisDateStrat()"> Single</label>
                            <label><input type="radio" name="date_strat" value="recent"
                                onchange="toggleDisDateStrat()"> Recent</label>
                        </div>

                        <div id="dis_range_div"
                            style="display:flex; align-items:center; gap:6px;">
                            <input type="date" id="dis_start_date"> <span>to</span>
                            <input type="date" id="dis_end_date">
                        </div>
                        <input type="date" id="dis_single_date" style="display:none;">
                        <input type="number" id="dis_recent_n" placeholder="# passes"
                            min="1" style="display:none; width: 100px;">

                        <div id="dis_adv_options"
                            style="display:flex; gap:10px; margin-left: auto;">
                            <label class="dynamic-opt">
                                <input type="checkbox" id="opt_nomask"> No Mask
                            </label>
                            <label class="dynamic-opt" id="wrap_rc"
                                style="display:none;">
                                <input type="checkbox" id="opt_rc"> Reclassify Snow/Ice
                            </label>
                            <label class="dynamic-opt" id="wrap_cloud"
                                style="display:none;">
                                <input type="checkbox" id="opt_cloud"> Calc Cloudiness
                            </label>

                            <div class="dynamic-opt" id="wrap_fd"
                                style="display:none; height:34px;
                                box-sizing:border-box;">
                                Filter (-fd): <input type="date" id="opt_fd"
                                    style="height:24px; border:none; padding:0 4px;">
                            </div>
                            <div class="dynamic-opt" id="wrap_st"
                                style="display:none; height:34px;
                                box-sizing:border-box;">
                                Slope (-st): <input type="number" id="opt_st"
                                    placeholder="Deg" min="0" max="100"
                                    style="width:50px; height:24px; border:none;
                                    padding:0 4px;">
                            </div>
                        </div>
                    </div>
                </div>
                {% endmacro %}

                {% macro script(this, kwargs) %}
                var currentBbox = null;
                var currentBboxLayer = null;
                var justDrawn = false;

                function setupMultiDropdown(btnId, listId, allValue, defaultLabel) {
                    var btn = document.getElementById(btnId);
                    var list = document.getElementById(listId);
                    var checkboxes = list.querySelectorAll('input[type=checkbox]');
                    var allBox = list.querySelector('input[value="' + allValue + '"]');

                    btn.addEventListener('click', function(e) {
                        e.stopPropagation();
                        var active = '.multi-dropdown-list.open';
                        var opens = document.querySelectorAll(active);
                        opens.forEach(function(el) {
                            if (el !== list) el.classList.remove('open');
                        });
                        list.classList.toggle('open');
                    });

                    checkboxes.forEach(function(cb) {
                        cb.addEventListener('change', function() {
                            if (cb === allBox && cb.checked) {
                                checkboxes.forEach(function(c) { c.checked = false; });
                                allBox.checked = true;
                            } else if (cb !== allBox && cb.checked) {
                                allBox.checked = false;
                            }
                            updateLabel();
                            if (listId === 'func-list') updatePanels();
                            if (listId === 'prod-list') updateDisasterOptions();
                        });
                    });

                    function updateLabel() {
                        if (allBox.checked) {
                            btn.textContent = defaultLabel;
                            return;
                        }
                        var sel = Array.from(checkboxes)
                            .filter(c => c.checked && c !== allBox)
                            .map(c => c.value);
                        btn.textContent = sel.length ? sel.join(', ') : defaultLabel;
                    }
                }

                function getMultiValues(listId, allValue) {
                    var list = document.getElementById(listId);
                    var allBox = list.querySelector('input[value="' + allValue + '"]');
                    if (allBox.checked) return [allValue];
                    var cbxs = list.querySelectorAll('input[type=checkbox]');
                    return Array.from(cbxs).filter(c => c.checked).map(c => c.value);
                }

                document.addEventListener('click', function() {
                    var active = '.multi-dropdown-list.open';
                    var opens = document.querySelectorAll(active);
                    opens.forEach(el => el.classList.remove('open'));
                });

                setupMultiDropdown(
                    'func-btn', 'func-list', 'all', 'All Functionality'
                );
                setupMultiDropdown(
                    'sat-btn', 'sat-list', 'all', 'All Satellites'
                );
                setupMultiDropdown(
                    'prod-btn', 'prod-list', 'all', 'All Products'
                );

                // --- UI Toggles ---
                function updatePanels() {
                    var funcs = getMultiValues('func-list', 'all');
                    var showNextPass = funcs.includes('all') ||
                                       funcs.includes('overpasses') ||
                                       funcs.includes('opera_search');
                    var showDisasters = funcs.includes('all') ||
                                        funcs.includes('disasters');

                    // Determine if the Products dropdown should be visible
                    var showProducts = funcs.includes('all') ||
                                       funcs.includes('opera_search') ||
                                       funcs.includes('disasters');

                    var nextPanel = document.getElementById('next-pass-panel');
                    nextPanel.style.display = showNextPass ? 'flex' : 'none';

                    var disPanel = document.getElementById('disasters-panel');
                    disPanel.style.display = showDisasters ? 'flex' : 'none';

                    // Toggle the Products dropdown
                    var prodDrop = document.getElementById('prod-dropdown');
                    prodDrop.style.display = showProducts ? 'inline-block' : 'none';

                    // Update Button Text
                    var btn = document.getElementById('action-btn');
                    if (showDisasters && !showNextPass) {
                        btn.textContent = "RUN DISASTERS";
                        btn.style.backgroundColor = "#059669"; // Green
                    } else {
                        btn.textContent = "SEARCH";
                        btn.style.backgroundColor = "#4b5563"; // Default Gray
                    }
                }

                function toggleDrcsDate(val) {
                    var input = document.getElementById('drcs_event_date');
                    input.disabled = (val !== 'yes');
                }

                function toggleDisDateStrat() {
                    var sel = 'input[name="date_strat"]:checked';
                    var val = document.querySelector(sel).value;
                    var dRec = document.getElementById('dis_recent_n');
                    var dSin = document.getElementById('dis_single_date');
                    var dRan = document.getElementById('dis_range_div');

                    dRec.style.display = (val === 'recent') ? 'block' : 'none';
                    dSin.style.display = (val === 'single') ? 'block' : 'none';
                    dRan.style.display = (val === 'range') ? 'flex' : 'none';
                }

                // --- Product-Aware Logic ---
                function updateDisasterOptions() {
                    var prods = getMultiValues('prod-list', 'all');
                    var isAll = prods.includes('all');

                    var hasHLS = isAll || prods.some(p => p.includes('HLS'));
                    var hasDSWxHLS = isAll || prods.some(p => p.includes('DSWX-HLS'));
                    var hasDIST = isAll || prods.some(p => p.includes('DIST'));
                    var hasRTC = isAll || prods.some(
                        p => p.includes('RTC') || p.includes('S1')
                    );

                    var wCloud = document.getElementById('wrap_cloud');
                    var wRc = document.getElementById('wrap_rc');
                    var wFd = document.getElementById('wrap_fd');
                    var wSt = document.getElementById('wrap_st');

                    wCloud.style.display = hasHLS ? 'flex' : 'none';
                    wRc.style.display = hasDSWxHLS ? 'flex' : 'none';
                    wFd.style.display = hasDIST ? 'flex' : 'none';
                    wSt.style.display = hasRTC ? 'flex' : 'none';
                }

                // Initialize UI
                updatePanels();
                toggleDisDateStrat();
                updateDisasterOptions();

                // --- Map Box Drawing ---
                {{this._parent.get_name()}}.on('draw:created', function(e) {
                    if (currentBboxLayer) {
                        {{this._parent.get_name()}}.removeLayer(currentBboxLayer);
                    }
                    currentBboxLayer = e.layer;
                    currentBboxLayer.addTo({{this._parent.get_name()}});
                    var bounds = currentBboxLayer.getBounds();
                    currentBbox = {
                        lat_min: bounds.getSouth(),
                        lat_max: bounds.getNorth(),
                        lon_min: bounds.getWest(),
                        lon_max: bounds.getEast()
                    };
                    justDrawn = true;
                });

                {{this._parent.get_name()}}.on('click', function() {
                    if (justDrawn) {
                        justDrawn = false;
                        return;
                    }
                    if (currentBboxLayer) {
                        {{this._parent.get_name()}}.removeLayer(currentBboxLayer);
                        currentBboxLayer = null;
                        currentBbox = null;
                    }
                });

                // --- Submit Payload ---
                document.getElementById('action-btn').onclick = function() {
                    if (!currentBbox) {
                        alert("Please draw a bounding box!");
                        return;
                    }

                    const payload = {
                        ...currentBbox,
                        search_type: getMultiValues('func-list', 'all'),
                        products: getMultiValues('prod-list', 'all'),

                        // Next Pass Payload
                        satellites: getMultiValues('sat-list', 'all'),
                        np_lookback: document.getElementById('np_lookback').value,
                        drcs: document.getElementById('drcs_enabled').value,
                        event_date: document.getElementById('drcs_event_date').value,

                        // Disasters Payload
                        dis_action: document.getElementById('dis_action').value,
                        dis_date_strat: document.querySelector(
                            'input[name="date_strat"]:checked'
                        ).value,
                        dis_recent_n: document.getElementById('dis_recent_n').value,
                        dis_single_date: document.getElementById(
                            'dis_single_date'
                        ).value,
                        dis_start_date: document.getElementById('dis_start_date').value,
                        dis_end_date: document.getElementById('dis_end_date').value,

                        // Disasters Advanced Opts
                        opt_nomask: document.getElementById('opt_nomask').checked,
                        opt_rc: document.getElementById('opt_rc').checked,
                        opt_cloud: document.getElementById('opt_cloud').checked,
                        opt_fd: document.getElementById('opt_fd').value,
                        opt_st: document.getElementById('opt_st').value
                    };

                    fetch("/process_bbox", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify(payload)
                    }).then(r => r.json()).then(data => {
                        if (data.run_id) {
                            alert("Request submitted. Processing...");
                            checkStatus(data.run_id);
                        }
                    }).catch(e => alert("Error: " + e.message));
                };

                function checkStatus(runId) {
                    var encId = encodeURIComponent(runId);
                    fetch(`/processing_status?run_id=${encId}`)
                        .then(r => r.json())
                        .then(status => {
                            if (status.running) {
                                setTimeout(() => checkStatus(runId), 2000);
                            } else if (status.error) {
                                alert("Failed: " + status.error);
                            } else {
                                window.location.href = `/show_maps?run_id=${encId}`;
                            }
                        });
                }
                {% endmacro %}
            """
            )

    output_file = file_dir / "activated_events_map.html"

    US_CENTER = [39.8283, -98.5795]
    map_object = folium.Map(location=US_CENTER, zoom_start=5, tiles=None)

    # Add base layers
    folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(map_object)
    carto_tiles_url = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
    carto_api_key = os.environ.get("CARTO_API_KEY")
    if carto_api_key:
        carto_tiles_url += f"?key={carto_api_key}"
    folium.TileLayer(
        tiles=carto_tiles_url,
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            ' contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        name="Street Map",
        max_zoom=20,
    ).add_to(map_object)

    # Add grouped event layers
    for event_type, group_events in events.items():
        color = _color_from_event_type(event_type)
        # Use circle for earthquakes, rectangle for other events
        family = _detect_family(event_type)
        border_radius = "50%" if family == "earthquake" else "0"
        color_box = (
            "<span style='display:inline-block; width:12px; height:12px; "
            f"background:{color}; margin-right:6px; border:1px solid #333; "
            f"border-radius:{border_radius};'></span>"
        )
        legend_label = f"{color_box}{event_type} ({len(group_events)})"
        feature_group = folium.FeatureGroup(name=legend_label, show=True)

        for e in group_events:
            geom = e.get("aoi_polygon")
            if geom is None:
                log.debug("Event %s has no AOI geometry", e.get("id"))
                continue
            provider = str(e.get("provider", "")).upper()
            props = e.get("properties") if isinstance(e.get("properties"), dict) else {}
            mag = props.get("mag")
            depth = props.get("depth_km")
            # Extract location for earthquake events
            if family == "earthquake":
                # Use the complete place string from properties
                place = props.get("place")
                location_str = html.escape(str(place)) if place else ""
                popup_rows = [
                    ("Provider", provider),
                    ("Severity", e.get("severity")),
                    ("Location", location_str),
                ]
            else:
                popup_rows = [
                    ("Provider", provider),
                    ("Severity", e.get("severity")),
                    ("Description", e.get("title")),
                ]
            if mag is not None:
                try:
                    popup_rows.append(("Magnitude", f"M {float(mag):.1f}"))
                except (TypeError, ValueError):
                    pass
            if depth is not None:
                try:
                    popup_rows.append(("Depth", f"{float(depth):.1f} km"))
                except (TypeError, ValueError):
                    pass
            popup_html = "<br>".join(
                f"<b>{label}:</b> {value}" for label, value in popup_rows if value
            )
            if family == "earthquake":
                header_parts: list[str] = []
                # Extract region (last part after comma) from place
                place = props.get("place")
                if place:
                    place_str = str(place)
                    region = (
                        place_str.split(",")[-1].strip()
                        if "," in place_str
                        else place_str
                    )
                    header_parts.append(html.escape(region))
                if mag is not None:
                    try:
                        header_parts.append(f"M {float(mag):.1f}")
                    except (TypeError, ValueError):
                        pass
                time_ms = props.get("time")
                if time_ms is not None:
                    try:
                        dt = datetime.fromtimestamp(
                            float(time_ms) / 1000, tz=timezone.utc
                        )
                        header_parts.append(dt.strftime("%Y-%m-%d %H:%M UTC"))
                    except (TypeError, ValueError, OSError, OverflowError):
                        pass
                if header_parts:
                    header_html = f"<b>{' &middot; '.join(header_parts)}</b>"
                    popup_html = (
                        f"{header_html}<br>{popup_html}" if popup_html else header_html
                    )
            if isinstance(geom, Point):
                try:
                    radius = _magnitude_to_radius(
                        float(mag) if mag is not None else None
                    )
                except (TypeError, ValueError):
                    radius = _magnitude_to_radius(None)
                folium.CircleMarker(
                    location=[geom.y, geom.x],
                    radius=radius,
                    color=color,
                    weight=1.5,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.55,
                    popup=folium.Popup(popup_html, max_width=350),
                ).add_to(feature_group)
                continue
            if isinstance(geom, MultiPolygon):
                geometries = geom.geoms
            else:
                geometries = [geom]
            for g in geometries:
                GeoJson(
                    data=g.__geo_interface__,
                    style_function=lambda _, c=color: {
                        "color": c,
                        "weight": 2,
                        "fillColor": c,
                        "fillOpacity": 0.35,
                    },
                    highlight_function=lambda _: {"weight": 3, "fillOpacity": 0.6},
                    popup=folium.Popup(popup_html, max_width=350),
                ).add_to(feature_group)
        feature_group.add_to(map_object)

    # Add Layer controls
    folium.LayerControl(collapsed=False).add_to(map_object)

    draw = Draw(
        draw_options={
            "rectangle": True,
            "polygon": False,
            "circle": False,
            "marker": False,
            "polyline": False,
        },
        edit_options={"edit": True},
    )
    draw.add_to(map_object)

    # After adding draw controls, add the new dashboard
    map_object.add_child(MapDashboardJS())

    # Save HTML
    map_object.save(output_file)
    log.info("Event map written to %s", output_file)


def _bbox_to_geometry(bbox, timestamp_dir):
    from shapely import Point, wkt
    from shapely.geometry import box

    if isinstance(bbox, str):
        bbox_clean = bbox.strip()
        bbox_upper = bbox_clean.upper()
        if bbox_upper.startswith(("POINT", "POLYGON")):
            geometry = wkt.loads(bbox_clean)
        else:
            # if URL, download
            if _is_url(bbox_clean):
                filename = "AOI_from_url.geojson"
                file_path = Path(timestamp_dir) / filename
                bbox_path = _download_url_to_file(bbox_clean, file_path)
            else:
                raise ValueError(
                    "Local file paths are not allowed for event AOI sources"
                )
            geometry = _geometry_from_file(bbox_path)
    else:
        lat_min, lat_max, lon_min, lon_max = bbox
        if lat_min == lat_max and lon_min == lon_max:
            geometry = Point(lon_min, lat_min)
        else:
            geometry = box(lon_min, lat_min, lon_max, lat_max)

    return geometry, geometry.bounds, geometry.centroid


def _add_aoi_to_events(
    events: Iterable[Event],
    file_dir: str,
) -> List[Event]:
    """
    Enrich events with AOI geometry derived from their link.
    Adds:
      - event["aoi_polygon"]
      - event["aoi"]
      - event["centroid"]
    """
    from shapely.geometry import Point

    out: List[Event] = []
    for e in events:
        props = e.get("properties")
        if not isinstance(props, dict):
            props = {}
        event_type = str(props.get("event") or "")
        link = ""
        event_lower = event_type.lower()
        if "earthquake" in event_lower:
            geom = e.get("geometry")
            coords = geom.get("coordinates") if isinstance(geom, dict) else None
            if isinstance(coords, list) and len(coords) >= 2:
                pt = Point(float(coords[0]), float(coords[1]))
                e["aoi_polygon"] = pt
                e["aoi"] = pt.bounds
                e["centroid"] = pt
            else:
                log.debug("Earthquake %s missing point geometry", e.get("id"))
            out.append(e)
            continue
        if "flood" in event_lower:
            raw_link = e.get("link")
            link = str(raw_link).strip() if isinstance(raw_link, str) else ""
        elif "storm" in event_lower:
            affected_zones = props.get("affectedZones", [])
            link = str(affected_zones[0]) if affected_zones else ""
        if not link:
            log.debug("Event %s has no link; skipping AOI", e.get("id"))
            out.append(e)
            continue

        try:
            aoi_polygon, aoi, centroid = _bbox_to_geometry(link, file_dir)

            e["aoi_polygon"] = aoi_polygon
            e["aoi"] = aoi
            e["centroid"] = centroid

        except Exception as exc:
            log.warning(
                "Failed to build AOI for event %s (link=%r): %s",
                e.get("id"),
                link,
                exc,
            )
        out.append(e)
    return out


def _download_url_to_file(
    url: str,
    output_path: str | Path,
    timeout: int = 30,
    ensure_geojson: bool = True,
) -> Path:
    """
    Download a URL and save its content to a file (GeoJSON-safe).
    """
    output_path = Path(output_path)

    if ensure_geojson and output_path.suffix.lower() != ".geojson":
        output_path = output_path.with_suffix(".geojson")

    _validate_remote_url(url)

    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    ctype = (response.headers.get("Content-Type") or "").lower()
    if "json" not in ctype:
        raise ValueError(f"Expected JSON payload from {url}, got Content-Type={ctype}")

    payload = bytearray()
    for chunk in response.iter_content(chunk_size=16384):
        if not chunk:
            continue
        payload.extend(chunk)
        if len(payload) > MAX_GEOJSON_BYTES:
            raise ValueError(
                f"Response from {url} exceeded max size ({MAX_GEOJSON_BYTES} bytes)"
            )

    # Parse JSON to ensure validity
    try:
        data = json.loads(payload.decode("utf-8"))
    except ValueError as e:
        raise ValueError(f"Response from {url} is not valid JSON") from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output_path


def _geometry_from_file(path: str | Path):
    """
    Read a geometry from a spatial file (KML or GeoJSON).
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    path = Path(path)
    suffix = path.suffix.lower()

    # # ---- KML ----
    # if suffix == ".kml":
    #     return create_polygon_from_kml(str(path))

    # ---- GeoJSON ----
    if suffix in (".geojson", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # FeatureCollection
        if data["type"] == "FeatureCollection":
            geometries = [shape(f["geometry"]) for f in data["features"]]
            return geometries[0] if len(geometries) == 1 else unary_union(geometries)

        # Single Feature
        if data.get("type") == "Feature":
            geometry = data.get("geometry")

            if geometry is None:
                affected = data.get("properties", {}).get("affectedZones", [])

                geometries = []
                for zone_url in affected:
                    try:
                        zone_data = requests.get(zone_url).json()
                        if zone_data.get("geometry"):
                            geometries.append(shape(zone_data["geometry"]))
                    except Exception:
                        pass

                if geometries:
                    return (
                        geometries[0]
                        if len(geometries) == 1
                        else unary_union(geometries)
                    )
                else:
                    raise ValueError("No geometry found and affectedZones failed.")

            return shape(geometry)

        # Raw geometry
        return shape(data)

    raise ValueError(f"Unsupported spatial file format: {path}")
