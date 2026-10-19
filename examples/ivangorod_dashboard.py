# -*- coding: utf-8 -*-
"""
Собирает одну HTML-страницу с интерактивной картой и панелью показателей
по результатам анализа Ивангорода (ноутбук ivangorod_blocksnet.ipynb, BlocksNet 1.0.0a9).

Запуск из корня репозитория:
    .venv\\Scripts\\python.exe examples\\ivangorod_dashboard.py

Результат: examples/ivangorod_dashboard.html — откройте в браузере (на этой машине лучше через Chrome;
двойной клик по файлу может не работать).
Для карты (тайлы Esri и библиотека Leaflet с CDN) нужен интернет.
Скрипт ничего не скачивает и не пересчитывает — читает готовые файлы из examples/data/ivangorod/.
"""
import json
import math
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "ivangorod"
OUT = BASE / "ivangorod_dashboard.html"
CRS_METRIC = 32635  # UTM 35N — для площадей в метрах


def fnum(x):
    """Число из таблицы; NaN/None -> None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def round_coords(x, nd=6):
    """Округляет координаты ГеоЖСОН до nd знаков (примерно 10 см)."""
    if isinstance(x, (list, tuple)):
        return [round_coords(v, nd) for v in x]
    if isinstance(x, float):
        return round(x, nd)
    return x


def feature(geom, props):
    m = mapping(geom)
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": m["type"], "coordinates": round_coords(m["coordinates"])},
    }


# ---------- 1. Кварталы ----------
blocks = gpd.read_file(DATA / "blocks_results.geojson")

# Потоки «квартал → школа» из расчёта обеспеченности (файл создаёт ноутбук, Раздел 10)
flows_out, flows_in = {}, {}
links_path = DATA / "provision_school_links.csv"
if links_path.exists():
    for _, lr in pd.read_csv(links_path).iterrows():
        s_id, t_id, v = int(lr["source"]), int(lr["target"]), int(round(float(lr["value"])))
        flows_out.setdefault(s_id, []).append([t_id, v])
        flows_in.setdefault(t_id, []).append([s_id, v])

ZONE_COLS = ["residential", "business", "industrial", "recreation", "agriculture", "special", "transport"]

block_feats = []
for row in blocks.itertuples():
    area_ha = (fnum(row.density_site_area) or 0.0) / 1e4
    pop = fnum(row.population) or 0.0
    bid = int(row.block_id)
    zones = {}
    for z in ZONE_COLS:
        zv = fnum(getattr(row, z, None))
        if zv and zv > 0.005:
            zones[z] = round(zv, 3)
    props = {
        "id": bid,
        "population": pop,
        "area_ha": round(area_ha, 2),
        "pop_density": round(pop / area_ha, 1) if area_ha > 0 else 0.0,
        "prov": fnum(row.provision_strong),
        "pw": fnum(row.provision_weak),
        "access": fnum(row.area_accessibility),
        "land": row.land_use if isinstance(row.land_use, str) else None,
        "capacity": fnum(row.capacity) or 0.0,
        "demand": int(fnum(row.demand) or 0),
        "dwithin": int(fnum(row.demand_within) or 0),
        "dwithout": int(fnum(row.demand_without) or 0),
        "fsi": fnum(row.density_fsi),
        "gsi": fnum(row.density_gsi),
        "mxi": fnum(row.density_mxi),
        "l": fnum(row.density_l),
        "osr": fnum(row.density_osr),
        "fp_area": fnum(row.density_footprint_area),
        "bf_area": fnum(row.density_build_floor_area),
        "lv_area": fnum(row.density_living_area),
        "nl_area": fnum(row.density_non_living_area),
        "sl": fnum(row.density_share_living),
        "snl": fnum(row.density_share_non_living),
        "morph_s": row.morphotype_spacematrix if isinstance(row.morphotype_spacematrix, str) else None,
        "morph_k": row.morphotype_strelka if isinstance(row.morphotype_strelka, str) else None,
        "zones": zones,
        "flows_out": flows_out.get(bid, []),
        "flows_in": flows_in.get(bid, []),
    }
    block_feats.append(feature(row.geometry, props))

# ---------- 2. Школы ----------
schools = gpd.read_file(DATA / "schools_raw.geojson").to_crs(4326)
schools = schools.rename(columns={"name": "school_name"})
school_feats = []
for row in schools.itertuples():
    pt = row.geometry.representative_point()
    name = row.school_name if isinstance(row.school_name, str) else None
    school_feats.append(feature(pt, {"name": name}))

# ---------- 3. Граница ----------
bnd = gpd.read_file(DATA / "boundaries.geojson")
bnd_feats = [feature(bnd.to_crs(4326).union_all(), {})]
municip_km2 = bnd.to_crs(CRS_METRIC).union_all().area / 1e6

# ---------- 4. Числа для сводки ----------
pop_sum = int(round(blocks["population"].fillna(0).sum()))
cap_sum = int(round(blocks["capacity"].fillna(0).sum()))
dem_sum = int(round(blocks["demand"].fillna(0).sum()))
dem_within = float(blocks["demand_within"].fillna(0).sum())
prov_overall = dem_within / dem_sum if dem_sum else 0.0

res = blocks[blocks["demand"].fillna(0) > 0]
n_res = len(res)
s = res["provision_strong"]
full = int((s >= 0.999).sum())
part = int(((s > 0) & (s < 0.999)).sum())
zero = int((s <= 0).sum())
mean_acc = float(blocks["area_accessibility"].mean())

mx = pd.read_parquet(DATA / "acc_mx.parquet")
vals = mx.to_numpy(dtype=float)
n = vals.shape[0]
off = vals[~np.eye(n, dtype=bool)]
med_pair = float(np.nanmedian(off))
med5 = int(round(med_pair / 5.0) * 5)

n_blocks = len(blocks)
n_schools = len(schools)
today = date.today().strftime("%d.%m.%Y")


def sp(x):
    """Целое число с пробелом между тысячами: 7 689."""
    return f"{int(round(x)):,}".replace(",", " ")


# ---------- 5. Тексты панели ----------
summary = (
    '<div class="cards">'
    f'<div class="card"><div class="card-v">{sp(pop_sum)}</div><div class="card-l">человек живёт</div></div>'
    f'<div class="card"><div class="card-v">{n_blocks}</div><div class="card-l">кварталов</div></div>'
    f'<div class="card"><div class="card-v">{round(prov_overall * 100)}%</div>'
    '<div class="card-l">потребности в школах закрыто в пределах нормы</div></div>'
    f'<div class="card"><div class="card-v">{round(mean_acc)} мин</div>'
    '<div class="card-l">средняя доступность территории*</div></div>'
    "</div>"
    '<div class="chips">'
    f'<span class="chip chip-full">полная — {full}</span>'
    f'<span class="chip chip-part">частично — {part}</span>'
    f'<span class="chip chip-none">нулевая — {zero}</span>'
    "</div>"
    f'<div class="legnote">Обеспеченность школами по {n_res} жилым кварталам; '
    f"школ в городе — {n_schools}, вместимость {sp(cap_sum)} мест.</div>"
)

notes = (
    "<p>Обеспеченность — доля потребности квартала в школьных местах "
    "(норматив: 120 мест на 1000 жителей), которую закрывают школы в пределах 15 минут пешком, "
    "с учётом конкуренции между кварталами.</p>"
    f"<p>* Средняя доступность взвешена по площади всех кварталов — включая леса и поля вокруг города "
    f"(общая территория {municip_km2:.0f} км²). Поэтому значение высокое: "
    f"медианный пеший маршрут «квартал → квартал» — около {med5} минут.</p>"
    f"<p>Данные: OpenStreetMap ({today}); расчёт: BlocksNet 1.0.0a9, ходьба 5 км/ч. "
    f"Школы (по OSM): {n_schools} шт., вместимость {sp(cap_sum)} мест, потребность {sp(dem_sum)} мест.</p>"
)

# ---------- 6. Данные и шаблон страницы ----------
data = {
    "blocks": {"type": "FeatureCollection", "features": block_feats},
    "schools": {"type": "FeatureCollection", "features": school_feats},
    "boundary": {"type": "FeatureCollection", "features": bnd_feats},
}
data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

TPL = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ивангород — анализ городской среды</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<style>
  html, body { margin: 0; height: 100%; font-family: "Segoe UI", system-ui, sans-serif; }
  #map { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }
  #panel { position: absolute; top: 0; left: 0; bottom: 0; width: 370px; overflow-y: auto;
           background: #fff; z-index: 1000; box-shadow: 2px 0 10px rgba(0,0,0,.15);
           padding: 18px 18px 30px; box-sizing: border-box; }
  h1 { font-size: 22px; margin: 0 0 2px; }
  .sub { color: #777; font-size: 12px; margin-bottom: 6px; }
  .sec-title { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .4px;
               color: #444; margin: 16px 0 8px; border-top: 1px solid #eee; padding-top: 12px; }
  label.metric { display: flex; align-items: center; gap: 8px; font-size: 14px; padding: 5px 0; cursor: pointer; }
  label.metric input { accent-color: #d94801; }
  label.chk { display: flex; align-items: center; gap: 8px; font-size: 14px; padding: 4px 0; cursor: pointer; }
  #legend { margin-top: 10px; font-size: 12px; color: #555; }
  .legbar { height: 12px; border-radius: 3px; border: 1px solid #ddd; }
  .leglabels { display: flex; justify-content: space-between; margin-top: 3px; }
  .legnote { margin-top: 4px; color: #888; font-size: 12px; }
  .swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px; border: 1px solid #ccc;
            margin-right: 6px; vertical-align: -2px; }
  .legrow { margin-top: 4px; }
  .hint { font-size: 12px; color: #888; margin-top: 8px; }
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .card { background: #f7f7f5; border-radius: 8px; padding: 10px 12px; }
  .card-v { font-size: 20px; font-weight: 700; color: #222; }
  .card-l { font-size: 12px; color: #666; line-height: 1.25; margin-top: 2px; }
  .chips { margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; }
  .chip { font-size: 12px; padding: 3px 9px; border-radius: 20px; }
  .chip-full { background: #d9f0d3; color: #1a6820; }
  .chip-part { background: #fdf3cf; color: #8a6d00; }
  .chip-none { background: #fbdcd9; color: #a33028; }
  .notes { font-size: 12px; color: #777; line-height: 1.45; margin-top: 14px; }
  .notes p { margin: 6px 0; }
  .btn { font-size: 12px; padding: 5px 10px; margin: 4px 4px 0 0; border: 1px solid #ccc;
         border-radius: 6px; background: #fafafa; cursor: pointer; }
  .btn:hover { background: #f0f0ee; }
  .pop { font-size: 13px; line-height: 1.45; }
  .pop-title { font-weight: 700; font-size: 14px; margin-bottom: 3px; }
  .pop-sec { margin-top: 7px; font-weight: 700; color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: .4px; }
  #table-box { display: none; max-height: 320px; overflow: auto; margin-top: 6px; border: 1px solid #eee; border-radius: 6px; }
  table.pt { width: 100%; border-collapse: collapse; font-size: 10.5px; }
  table.pt th, table.pt td { padding: 3px 4px; border-bottom: 1px solid #f0f0f0; text-align: right; white-space: nowrap; }
  table.pt th { position: sticky; top: 0; background: #fafafa; font-weight: 600; }
  @media (max-width: 850px) {
    #panel { width: 100%; bottom: auto; max-height: 52%; }
    #map { top: 52%; }
  }
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h1>Ивангород</h1>
  <div class="sub">Анализ городской среды · BlocksNet 1.0.0a9 · данные OpenStreetMap</div>

  <div class="sec-title">Показатель на карте</div>
  <div id="metric-list"></div>
  <div id="legend"></div>
  <div class="hint">Нажмите на квартал на карте — увидите все его показатели.</div>

  <div class="sec-title">Слои и вид</div>
  <label class="chk"><input type="checkbox" id="chk_schools" checked> Школы (__N_SCHOOLS__)</label>
  <label class="chk"><input type="checkbox" id="chk_boundary" checked> Граница территории</label>
  <div>
    <button id="btn_city" class="btn">Город</button>
    <button id="btn_all" class="btn">Вся территория</button>
  </div>

  <div class="sec-title">Таблица показателей</div>
  <button id="btn_table" class="btn">Показать / скрыть таблицу</button>
  <div id="table-box"></div>

  <div class="sec-title">Город в цифрах</div>
  __SUMMARY__
  <div class="notes">__NOTES__</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<script id="app">
(function(){
"use strict";
if (typeof L === 'undefined') {
  document.getElementById('map').innerHTML = '<div style="padding:30px;font-family:system-ui">' +
    'Не удалось загрузить библиотеку карты — проверьте подключение к интернету и обновите страницу.</div>';
  document.getElementById('metric-list').innerHTML = '<div style="color:#888;font-size:13px">' +
    'Показатели появятся после загрузки карты.</div>';
  return;
}

const DATA = __DATA_JSON__;

const NODATA = '#d9d9d9';
const WARM = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20'];
const PROV = ['#d73027', '#fdae61', '#fee08b', '#a6d96a', '#1a9850'];

const LAND = {
  'LandUse.RESIDENTIAL': {label: 'Жилая', color: '#a6d854'},
  'LandUse.RECREATION': {label: 'Рекреационная', color: '#66c2a5'},
  'LandUse.INDUSTRIAL': {label: 'Промышленная', color: '#b15928'},
  'LandUse.BUSINESS': {label: 'Деловая', color: '#ffd92f'},
  'LandUse.SPECIAL': {label: 'Специальная', color: '#e78ac3'},
  'LandUse.AGRICULTURE': {label: 'Сельскохозяйственная', color: '#dfc27d'},
  'LandUse.TRANSPORT': {label: 'Транспортная', color: '#9e9e9e'}
};

const METRICS = {
  population: {title: 'Население, чел.', stops: WARM, grayText: 'Серый — кварталы без жителей'},
  pop_density: {title: 'Плотность населения, чел./га', stops: WARM, grayText: 'Серый — кварталы без жителей'},
  provision: {title: 'Обеспеченность школами, 0–100%', stops: PROV, grayText: 'Серый — кварталы без жителей'},
  access: {title: 'Доступность территории, мин', stops: WARM, grayText: ''},
  land: {title: 'Землепользование', stops: null, grayText: ''},
  fsi: {title: 'Плотность застройки (FSI), м²/м²', stops: WARM, grayText: 'Серый — кварталы без зданий'},
  gsi: {title: 'Застроенность (GSI)', stops: WARM, grayText: 'Серый — кварталы без зданий'},
  l: {title: 'Этажность застройки (L)', stops: WARM, grayText: 'Серый — кварталы без зданий'},
  morph_s: {title: 'Морфотипы: Spacematrix', stops: null, grayText: 'Серый — кварталы без застройки'},
  morph_k: {title: 'Морфотипы: Strelka', stops: null, grayText: 'Серый — кварталы без застройки'}
};
const ORDER = ['population', 'pop_density', 'provision', 'access', 'land', 'fsi', 'gsi', 'l', 'morph_s', 'morph_k'];

let currentKey = 'population';

const dom = {population: {max: 1}, pop_density: {max: 1}, fsi: {max: 0.001}, gsi: {max: 0.001}, l: {max: 1}, access: {min: Infinity, max: -Infinity}};
DATA.blocks.features.forEach(function(f){
  const p = f.properties;
  if (p.population) dom.population.max = Math.max(dom.population.max, p.population);
  if (p.pop_density) dom.pop_density.max = Math.max(dom.pop_density.max, p.pop_density);
  if (p.fsi) dom.fsi.max = Math.max(dom.fsi.max, p.fsi);
  if (p.gsi) dom.gsi.max = Math.max(dom.gsi.max, p.gsi);
  if (p.l) dom.l.max = Math.max(dom.l.max, p.l);
  if (p.access != null) {
    dom.access.min = Math.min(dom.access.min, p.access);
    dom.access.max = Math.max(dom.access.max, p.access);
  }
});
if (!isFinite(dom.access.min)) { dom.access.min = 0; dom.access.max = 1; }

const CATS = {};
(function(){
  const PAL = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#bc80bd', '#ccebc5'];
  ['morph_s', 'morph_k'].forEach(function(k){
    const vals = [];
    DATA.blocks.features.forEach(function(f){
      const v = f.properties[k];
      if (v && vals.indexOf(v) < 0) vals.push(v);
    });
    vals.sort();
    const m = {};
    vals.forEach(function(v, i){ m[v] = PAL[i % PAL.length]; });
    CATS[k] = m;
  });
})();

function hexToRgb(h){ const v = parseInt(h.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
function rgbToHex(r){ return '#' + r.map(function(x){ return Math.round(x).toString(16).padStart(2, '0'); }).join(''); }
function ramp(stops, t){
  t = Math.max(0, Math.min(1, t));
  const s = (stops.length - 1) * t;
  const i = Math.min(stops.length - 2, Math.floor(s));
  const f = s - i;
  const a = hexToRgb(stops[i]), b = hexToRgb(stops[i + 1]);
  return rgbToHex([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]);
}
function fmtInt(v){ return Math.round(v).toLocaleString('ru-RU'); }

function normFn(k, v){
  if (k === 'population') return Math.sqrt(v / dom.population.max);
  if (k === 'pop_density') return Math.sqrt(v / dom.pop_density.max);
  if (k === 'provision') return v;
  if (k === 'fsi') return v / Math.max(1e-9, dom.fsi.max);
  if (k === 'gsi') return v / Math.max(1e-9, dom.gsi.max);
  if (k === 'l') return v / Math.max(1e-9, dom.l.max);
  return (v - dom.access.min) / Math.max(1e-9, dom.access.max - dom.access.min);
}

function valueOf(p, k){
  if (k === 'population') return p.population;
  if (k === 'pop_density') return p.pop_density;
  if (k === 'provision') return p.prov;
  if (k === 'fsi') return p.fsi;
  if (k === 'gsi') return p.gsi;
  if (k === 'l') return p.l;
  return p.access;
}

function colorFor(p){
  const k = currentKey;
  if (k === 'land') {
    const l = p.land ? LAND[p.land] : null;
    return l ? l.color : NODATA;
  }
  if (k === 'morph_s' || k === 'morph_k') {
    const mv = p[k];
    return (mv && CATS[k][mv]) ? CATS[k][mv] : NODATA;
  }
  const v = valueOf(p, k);
  if (v == null || ((k === 'population' || k === 'pop_density' || k === 'fsi' || k === 'gsi' || k === 'l') && !v)) return NODATA;
  return ramp(METRICS[k].stops, normFn(k, v));
}

const map = L.map('map', {zoomControl: false});
L.control.zoom({position: 'topright'}).addTo(map);
L.control.scale({position: 'bottomright', imperial: false}).addTo(map);
const baseStreet = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 19,
  attribution: 'Tiles &copy; Esri'
}).addTo(map);
const baseSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 19,
  attribution: 'Tiles &copy; Esri'
});
L.control.layers({'Схема': baseStreet, 'Спутник': baseSat}, null, {position: 'topright'}).addTo(map);

const boundaryLayer = L.geoJSON(DATA.boundary, {
  style: {color: '#444', weight: 2, dashArray: '6 4', fill: false}
}).addTo(map);

const schoolsLayer = L.geoJSON(DATA.schools, {
  pointToLayer: function(f, latlng){
    return L.circleMarker(latlng, {radius: 7, color: '#fff', weight: 2, fillColor: '#2c7fb8', fillOpacity: 1});
  },
  onEachFeature: function(f, layer){
    const name = f.properties.name ? f.properties.name : 'Школа';
    layer.bindPopup('<div class="pop"><div class="pop-title">' + name + '</div><div>Школа (OpenStreetMap)</div></div>');
  }
}).addTo(map);

function styleFor(f){
  return {color: '#ffffff', weight: 1, opacity: 1, fillColor: colorFor(f.properties), fillOpacity: 0.8};
}

function row(label, val){ return '<div>' + label + ': <b>' + val + '</b></div>'; }

function popupHtml(p){
  const land = (p.land && LAND[p.land]) ? LAND[p.land].label : 'не определена';
  const prov = p.prov == null ? '—' : Math.round(p.prov * 100) + '%';
  const acc = p.access == null ? '—' : p.access.toFixed(1) + ' мин';
  let html = '<div class="pop"><div class="pop-title">Квартал № ' + p.id + '</div>';
  html += row('Население', p.population ? fmtInt(p.population) + ' чел.' : '0 чел.');
  html += row('Площадь', p.area_ha.toFixed(1) + ' га');
  if (p.population > 0) html += row('Плотность населения', p.pop_density.toFixed(1) + ' чел./га');
  if (p.fsi) {
    html += '<div class="pop-sec">Застройка</div>';
    html += row('Плотность застройки (FSI)', p.fsi.toFixed(2) + ' м²/м²');
    html += row('Застроенность (GSI)', (p.gsi * 100).toFixed(1) + '%');
    html += row('Доля жилья (MXI)', (p.mxi * 100).toFixed(0) + '%');
    html += row('Этажность (L)', p.l.toFixed(1));
    html += row('Открытые пространства (OSR)', p.osr.toFixed(1));
    if (p.fp_area) html += row('Площадь застройки', fmtInt(p.fp_area) + ' м²');
    if (p.bf_area) html += row('Все этажи зданий', fmtInt(p.bf_area) + ' м²');
    if (p.lv_area) html += row('Жилая площадь', fmtInt(p.lv_area) + ' м²');
    if (p.nl_area) html += row('Нежилая площадь', fmtInt(p.nl_area) + ' м²');
    if (p.sl != null && p.snl != null) html += row('Справочно: жилая / нежилая на 1 м² застройки', p.sl.toFixed(2) + ' / ' + p.snl.toFixed(2));
  } else {
    html += row('Застройка', 'зданий по OSM не найдено');
  }
  if (p.morph_s) html += row('Морфотип (Spacematrix)', p.morph_s);
  if (p.morph_k) html += row('Морфотип (Strelka)', p.morph_k);
  html += row('Землепользование', land);
  const zlist = [];
  Object.keys(p.zones || {}).forEach(function(z){ zlist.push([z, p.zones[z]]); });
  zlist.sort(function(a, b){ return b[1] - a[1]; });
  if (zlist.length) {
    const ZLAB = {residential: 'жилая', business: 'деловая', industrial: 'промышленная', recreation: 'зелёная', agriculture: 'сельхоз', special: 'специальная', transport: 'транспортная'};
    html += row('Зоны в квартале', zlist.map(function(z){ return (ZLAB[z[0]] || z[0]) + ' — ' + Math.round(z[1] * 100) + '%'; }).join('; '));
  }
  html += '<div class="pop-sec">Школы и доступность</div>';
  html += row('Обеспеченность школами', prov);
  if (p.pw != null && p.prov != null && Math.abs(p.pw - p.prov) > 0.005) html += row('Обеспеченность (без ограничения 15 мин)', Math.round(p.pw * 100) + '%');
  if (p.demand > 0) {
    html += row('Потребность в школьных местах', fmtInt(p.demand) + ' мест');
    html += row('Закрыто в пределах 15 минут', fmtInt(p.dwithin) + ' мест');
    if (p.dwithout > 0) html += row('Закрыто дальше 15 минут', fmtInt(p.dwithout) + ' мест');
  }
  html += row('Доступность территории', acc);
  if (p.capacity > 0) html += row('Школы в квартале', fmtInt(p.capacity) + ' мест');
  if (p.flows_out && p.flows_out.length) html += row('Ученики (по расчёту)', p.flows_out.map(function(f){ return '→ квартал ' + f[0] + ': ' + fmtInt(f[1]) + ' мест'; }).join('; '));
  if (p.flows_in && p.flows_in.length) html += row('Принимает учеников', p.flows_in.map(function(f){ return 'из квартала ' + f[0] + ': ' + fmtInt(f[1]) + ' мест'; }).join('; '));
  html += '</div>';
  return html;
}

const blocksLayer = L.geoJSON(DATA.blocks, {
  style: styleFor,
  onEachFeature: function(f, layer){
    layer.bindPopup(popupHtml(f.properties), {maxWidth: 300});
    layer.on('mouseover', function(){
      layer.setStyle({color: '#222', weight: 2.5});
      layer.bringToFront();
    });
    layer.on('mouseout', function(){
      layer.setStyle(styleFor(f));
    });
  }
}).addTo(map);

function fitCity(){
  const b = L.latLngBounds([]);
  blocksLayer.eachLayer(function(l){
    if (l.feature && l.feature.properties.population > 0 && l.getBounds) b.extend(l.getBounds());
  });
  if (b.isValid()) map.fitBounds(b, {padding: [40, 40]});
  else map.fitBounds(boundaryLayer.getBounds(), {padding: [30, 30]});
}
fitCity();

function gradientCss(k){
  const m = METRICS[k];
  let lo, hi;
  if (k === 'provision') { lo = 0; hi = 1; }
  else if (k === 'access') { lo = dom.access.min; hi = dom.access.max; }
  else if (k === 'population') { lo = 0; hi = dom.population.max; }
  else if (k === 'pop_density') { lo = 0; hi = dom.pop_density.max; }
  else if (k === 'fsi') { lo = 0; hi = dom.fsi.max; }
  else if (k === 'gsi') { lo = 0; hi = dom.gsi.max; }
  else if (k === 'l') { lo = 0; hi = dom.l.max; }
  else { lo = 0; hi = 1; }
  const cols = [];
  for (let i = 0; i <= 24; i++) cols.push(ramp(m.stops, normFn(k, lo + (hi - lo) * i / 24)));
  return 'linear-gradient(to right,' + cols.join(',') + ')';
}

function renderLegend(){
  const box = document.getElementById('legend');
  let html = '';
  if (currentKey === 'morph_s' || currentKey === 'morph_k') {
    Object.keys(CATS[currentKey]).forEach(function(v){
      html += '<div class="legrow"><span class="swatch" style="background:' + CATS[currentKey][v] + '"></span>' +
              v + '</div>';
    });
    html += '<div class="legrow"><span class="swatch" style="background:' + NODATA + '"></span>Без застройки</div>';
  } else if (currentKey === 'land') {
    Object.keys(LAND).forEach(function(k){
      html += '<div class="legrow"><span class="swatch" style="background:' + LAND[k].color + '"></span>' +
              LAND[k].label + '</div>';
    });
    html += '<div class="legrow"><span class="swatch" style="background:' + NODATA + '"></span>' +
            'Не определена</div>';
  } else {
    const m = METRICS[currentKey];
    let lo, hi;
    if (currentKey === 'population') { lo = '0'; hi = fmtInt(dom.population.max); }
    else if (currentKey === 'pop_density') { lo = '0'; hi = fmtInt(dom.pop_density.max); }
    else if (currentKey === 'provision') { lo = '0%'; hi = '100%'; }
    else if (currentKey === 'fsi') { lo = '0'; hi = dom.fsi.max.toFixed(2); }
    else if (currentKey === 'gsi') { lo = '0'; hi = dom.gsi.max.toFixed(2); }
    else if (currentKey === 'l') { lo = '1'; hi = dom.l.max.toFixed(1); }
    else { lo = dom.access.min.toFixed(0); hi = dom.access.max.toFixed(0); }
    html += '<div class="legbar" style="background:' + gradientCss(currentKey) + '"></div>';
    html += '<div class="leglabels"><span>' + lo + '</span><span>' + hi + '</span></div>';
    if (m.grayText) html += '<div class="legnote">' + m.grayText + '</div>';
  }
  box.innerHTML = html;
}

function setMetric(k){
  currentKey = k;
  blocksLayer.eachLayer(function(l){ l.setStyle(styleFor(l.feature)); });
  renderLegend();
}

const list = document.getElementById('metric-list');
ORDER.forEach(function(k, idx){
  const lab = document.createElement('label');
  lab.className = 'metric';
  const inp = document.createElement('input');
  inp.type = 'radio';
  inp.name = 'metric';
  inp.value = k;
  if (idx === 0) inp.checked = true;
  inp.addEventListener('change', function(){ if (inp.checked) setMetric(k); });
  const span = document.createElement('span');
  span.textContent = METRICS[k].title;
  lab.appendChild(inp);
  lab.appendChild(span);
  list.appendChild(lab);
});

document.getElementById('chk_schools').addEventListener('change', function(e){
  if (e.target.checked) schoolsLayer.addTo(map); else map.removeLayer(schoolsLayer);
});
document.getElementById('chk_boundary').addEventListener('change', function(e){
  if (e.target.checked) boundaryLayer.addTo(map); else map.removeLayer(boundaryLayer);
});

function buildTable(){
  const box = document.getElementById('table-box');
  let h = '<table class="pt"><thead><tr>' +
          '<th>№</th><th>Нас.</th><th>FSI</th><th>GSI</th><th>MXI</th><th>L</th><th>OSR</th><th>Обесп.</th><th>Дост.</th>' +
          '</tr></thead><tbody>';
  DATA.blocks.features.forEach(function(f){
    const p = f.properties;
    h += '<tr>' +
      '<td>' + p.id + '</td>' +
      '<td>' + (p.population ? fmtInt(p.population) : '—') + '</td>' +
      '<td>' + (p.fsi ? p.fsi.toFixed(2) : '—') + '</td>' +
      '<td>' + (p.gsi ? (p.gsi * 100).toFixed(1) + '%' : '—') + '</td>' +
      '<td>' + (p.mxi ? (p.mxi * 100).toFixed(0) + '%' : '—') + '</td>' +
      '<td>' + (p.l ? p.l.toFixed(1) : '—') + '</td>' +
      '<td>' + (p.osr ? p.osr.toFixed(1) : '—') + '</td>' +
      '<td>' + (p.prov == null ? '—' : Math.round(p.prov * 100) + '%') + '</td>' +
      '<td>' + (p.access == null ? '—' : p.access.toFixed(0)) + '</td>' +
      '</tr>';
  });
  h += '</tbody></table>';
  box.innerHTML = h;
}
let tableBuilt = false;
document.getElementById('btn_table').addEventListener('click', function(){
  if (!tableBuilt) { buildTable(); tableBuilt = true; }
  const box = document.getElementById('table-box');
  box.style.display = (box.style.display === 'block') ? 'none' : 'block';
});
document.getElementById('btn_all').addEventListener('click', function(){
  map.fitBounds(boundaryLayer.getBounds(), {padding: [30, 30]});
});
document.getElementById('btn_city').addEventListener('click', fitCity);

renderLegend();
})();
</script>
</body>
</html>
"""

html = (
    TPL.replace("__DATA_JSON__", data_json)
    .replace("__SUMMARY__", summary)
    .replace("__NOTES__", notes)
    .replace("__N_SCHOOLS__", str(n_schools))
)
OUT.write_text(html, encoding="utf-8")

print("Готово!")
print("Файл:", OUT)
print("Откройте его в браузере (для карты нужен интернет).")
print("Если двойной клик не срабатывает — перетащите файл в открытое окно Chrome.")
