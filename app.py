"""
FIELDWISE AI — AI production optimisation for mature oil & gas fields
=====================================================================
IEW 2027 Hackathon · Problem Statement 3
"AI-Based Production Optimization and Reservoir Modelling for Mature Fields"

Run:   pip install -r requirements.txt
       streamlit run app.py

The app stays idle until a CSV / Excel file is uploaded. Then it:
  1. maps & cleans the data (auditable report)
  2. trains an AI virtual flow meter (oil-rate model) and validates it on unseen dates
  3. fits decline curves and forecasts 180 days
  4. learns gas-lift performance curves and optimises lift-gas allocation
  5. screens artificial-lift equipment and raises early warnings
  6. ranks wells for engineering attention, with reasons

Layout: 1 config · 2 ingestion & cleaning · 3 analytics · 4 optimiser · 5 UI kit · 6 pages · 7 main
Internal units: liquids m³/d · gas Sm³/d · pressure bar · choke size 1/64 in
"""
from __future__ import annotations

import difflib
import hashlib
import html
import io
import os
import re
from dataclasses import astuple, dataclass

# One thread per model fit. At dashboard data sizes this is fast (< 1 s), and it avoids a severe slowdown (up to
# 100×) when several Streamlit sessions or processes share the same CPU cores and threads fight each other.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score

try:                                                    # also applies if numpy/sklearn were imported before this file
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except Exception:  # noqa: BLE001
    pass

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
APP = "FIELDWISE AI"
AUTHOR = "Kavinkarthick"

CORE_COLUMNS = ["Date", "Well", "Oil_Rate", "Water_Rate", "Gas_Rate",
                "Wellhead_Pressure", "Bottomhole_Pressure", "Injection_Rate"]
SENSOR_COLUMNS = ["Pump_Vibration", "Motor_Temperature", "Motor_Current", "Pump_Frequency", "Energy_kWh"]
OPTIONAL_COLUMNS = ["Choke_Size", "Hours_On"] + SENSOR_COLUMNS
CORE_NUMERIC = CORE_COLUMNS[2:]
NUMERIC_COLUMNS = CORE_NUMERIC + OPTIONAL_COLUMNS
REQUIRED = ["Date", "Well", "Oil_Rate"]          # everything else is optional or derived
# Helper fields: not used directly, but let FIELDWISE derive missing rates (e.g. water from water cut)
AUX_COLUMNS = ["Water_Cut", "Liquid_Rate", "GOR"]
ALL_FIELDS = CORE_COLUMNS + OPTIONAL_COLUMNS + AUX_COLUMNS
RATE_COLUMNS = ["Oil_Rate", "Water_Rate", "Gas_Rate", "Injection_Rate"]
PRESSURE_COLUMNS = ["Wellhead_Pressure", "Bottomhole_Pressure"]
SETPOINTS = ["Choke_Size"]   # changed in steps: carried forward, never interpolated or capped
# Production rates come from periodic well tests (discrete samples) → kept exactly as measured, NEVER imputed.
# Pressures, lift gas and pump signals are recorded continuously → only short gaps are bridged.
TEST_MEASURED = ["Oil_Rate", "Water_Rate", "Gas_Rate"]
GAP_LIMIT = 3                 # continuous signals: bridge at most this many consecutive missing readings
TEST_MODE_THRESHOLD = 0.8     # oil measured on fewer than 80 % of rows → the file holds well tests, not daily rates

DESCRIPTIONS = {
    "Date": "Observation date", "Well": "Unique well ID", "Oil_Rate": "Oil production rate",
    "Water_Rate": "Water production rate", "Gas_Rate": "Produced gas rate",
    "Wellhead_Pressure": "Flowing wellhead / tubing-head pressure", "Bottomhole_Pressure": "Flowing bottom-hole pressure",
    "Injection_Rate": "Lift-gas (or injection) rate", "Choke_Size": "Production choke (bean) size",
    "Pump_Vibration": "Pump vibration", "Motor_Temperature": "Motor temperature", "Motor_Current": "Motor current",
    "Pump_Frequency": "VFD / pump frequency", "Energy_kWh": "Energy used in the interval",
    "Hours_On": "Hours the well produced that day (0 = shut in)",
    "Water_Cut": "Water cut / BS&W (% or fraction) — used to derive missing rates",
    "Liquid_Rate": "Gross liquid rate — used to derive missing oil or water",
    "GOR": "Gas-oil ratio — used to derive a missing gas rate",
}

ALIASES = {
    "Date": ["date", "datetime", "timestamp", "time", "sample date", "observation date", "production date"],
    "Well": ["well", "well id", "well name", "wellbore", "well bore", "well bore code", "wellbore name", "wellno", "well no", "uwi", "api", "well identifier"],
    "Oil_Rate": ["oil rate", "oil", "bore oil vol", "oil vol", "oil volume", "net oil", "oil production", "oil production rate", "qo", "oil m3 d", "oil rate (m3/d)",
                 "oil rate (m³/d)", "bopd", "oil bopd"],
    "Water_Rate": ["water rate", "water", "bore wat vol", "wat vol", "water vol", "water volume", "water production", "water production rate", "qw", "water m3 d",
                   "water rate (m3/d)", "water rate (m³/d)", "bwpd"],
    "Gas_Rate": ["gas rate", "gas", "bore gas vol", "gas vol", "gas volume", "gas production", "gas production rate", "qg", "gas mscf d", "gas rate (mcf/d)"],
    "Wellhead_Pressure": ["wellhead pressure", "whp", "tubing head pressure", "thp", "well head pressure", "fthp"],
    "Bottomhole_Pressure": ["bottomhole pressure", "bottom hole pressure", "bhp", "flowing bottomhole pressure",
                            "fbhp", "pwf", "downhole pressure"],
    "Injection_Rate": ["injection rate", "injection", "gas lift rate", "lift gas rate", "lift gas", "gas injection",
                       "water injection rate", "qinj", "glir"],
    "Choke_Size": ["choke size", "choke", "bean", "bean size", "choke 64ths", "choke (1/64 in)", "choke (64ths)",
                   "ck", "choke diameter"],
    "Pump_Vibration": ["pump vibration", "vibration", "vibration rms"],
    "Motor_Temperature": ["motor temperature", "motor temp", "motor temperature c"],
    "Motor_Current": ["motor current", "current", "motor amps", "motor current a"],
    "Pump_Frequency": ["pump frequency", "frequency", "vfd frequency", "frequency hz", "esp freq", "esp frequency",
                       "freq", "pump freq", "vsd frequency", "drive frequency", "motor frequency", "vfd freq"],
    "Energy_kWh": ["energy kwh", "energy", "power consumption kwh", "energy consumed", "energy consumption"],
    "Hours_On": ["hours on", "hours online", "hours on line", "uptime", "up time", "run hours", "hrs on", "on hours",
                 "producing hours", "production hours", "on stream hours", "onstream hours", "hours on stream"],
    "Water_Cut": ["water cut", "watercut", "wc", "bsw", "bs&w", "bs and w", "water cut %", "water cut (%)"],
    "Liquid_Rate": ["liquid rate", "liquid", "gross rate", "gross liquid", "total liquid", "gross fluid", "ql",
                    "fluid rate", "total fluid", "blpd"],
    "GOR": ["gor", "gas oil ratio", "gas/oil ratio", "gas-oil ratio"],
}

_LIQ = {"m³/d": 1.0, "bbl/d": 0.1589872949, "STB/d": 0.1589872949, "m³/h": 24.0, "L/s": 86.4, "L/min": 1.44}
_GAS = {"Sm³/d": 1.0, "Mscf/d": 28.3168466, "scf/d": 0.0283168466, "MMSCFD": 28316.8466, "Nm³/h": 24.0}
_PRES = {"bar": 1.0, "kPa": 0.01, "MPa": 10.0, "psi": 0.0689475729, "kgf/cm²": 0.980665}
UNITS = {  # standard = source × factor (°F also gets an offset)
    "Oil_Rate": _LIQ, "Water_Rate": _LIQ, "Gas_Rate": _GAS, "Injection_Rate": {**_GAS, **_LIQ},
    "Wellhead_Pressure": _PRES, "Bottomhole_Pressure": _PRES,
    "Choke_Size": {"1/64 in": 1.0, "in": 64.0, "mm": 64.0 / 25.4},
    "Motor_Temperature": {"°C": 1.0, "°F": 5.0 / 9.0}, "Motor_Current": {"A": 1.0, "mA": 0.001},
    "Pump_Frequency": {"Hz": 1.0, "rpm": 1.0 / 60.0}, "Pump_Vibration": {"mm/s": 1.0, "in/s": 25.4},
    "Energy_kWh": {"kWh": 1.0, "Wh": 0.001, "MWh": 1000.0},
    "Hours_On": {"h": 1.0, "%": 0.24},
    "Liquid_Rate": _LIQ,
    "GOR": {"Sm³/m³": 1.0, "scf/bbl": 0.178108, "Mscf/bbl": 178.108},
}
STD_UNIT = {"Oil_Rate": "m³/d", "Water_Rate": "m³/d", "Gas_Rate": "Sm³/d", "Injection_Rate": "Sm³/d",
            "Wellhead_Pressure": "bar", "Bottomhole_Pressure": "bar", "Choke_Size": "1/64 in",
            "Motor_Temperature": "°C", "Motor_Current": "A", "Pump_Frequency": "Hz", "Pump_Vibration": "mm/s",
            "Energy_kWh": "kWh", "Liquid_Rate": "m³/d", "GOR": "Sm³/m³", "Hours_On": "h"}
MISSING = {"", "nan", "none", "null", "n/a", "na", "unknown", "—", "-", "<na>", "#n/a", "nat"}
M3_TO_BBL = 6.289811
GRID_CO2 = 0.71   # kg CO2 per kWh — Indian grid average; adjustable in Settings

PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "WATCH": 2, "STABLE": 3}
COLORS = {"teal": "#2DD4BF", "blue": "#60A5FA", "amber": "#FBBF24", "red": "#F87171", "violet": "#A78BFA",
          "muted": "#64748B", "grid": "rgba(148,163,184,0.12)", "text": "#CBD5E1"}
PRIORITY_COLORS = {"CRITICAL": COLORS["red"], "HIGH": "#FB923C", "WATCH": COLORS["amber"], "STABLE": COLORS["teal"]}


# =============================================================================
# 2. INGESTION & CLEANING
# =============================================================================
def _norm(name) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())


def _strip_units(name) -> str:
    """'Oil Rate (bbl/d)' → 'oil rate';  'THP [psi]' → 'thp';  'Choke_in' stays recognisable."""
    return _norm(re.sub(r"[\(\[\{].*?[\)\]\}]", " ", str(name)))


_LOOKUP = {_norm(a): canon for canon, names in ALIASES.items() for a in [canon, *names]}
# Keyword fallback for headers no alias list anticipated (first match wins, most specific first)
_KEYWORDS = [
    ("Hours_On", ("uptime", "hours on", "run hours", "hrs on", "on stream")),
    ("Water_Cut", ("water cut", "watercut", "bsw", "bs&w")), ("GOR", ("gas oil ratio", "gas/oil", "gor")),
    ("Liquid_Rate", ("liquid", "gross", "total fluid", "blpd")),
    ("Choke_Size", ("choke", "bean")), ("Injection_Rate", ("lift gas", "gas lift", "injection", "glir", "gl inj", "inj rate", "gl rate", "inj")),
    ("Bottomhole_Pressure", ("bottom hole", "bottomhole", "bhp", "downhole press", "pwf")),
    ("Wellhead_Pressure", ("wellhead", "well head", "tubing head", "thp", "whp")),
    ("Pump_Vibration", ("vibration",)), ("Motor_Temperature", ("motor temp", "winding temp")),
    ("Motor_Current", ("amps", "current")), ("Pump_Frequency", ("frequency", "freq", "hz", "vfd", "vsd")),
    ("Energy_kWh", ("kwh", "energy", "power")), ("Water_Rate", ("water", "bwpd", "qw", "wat vol", "wat rate", "wtr")),
    ("Gas_Rate", ("gas", "qg", "mscf")), ("Oil_Rate", ("oil", "bopd", "qo", "crude")),
    ("Well", ("well",)), ("Date", ("date", "time", "day")),
]
# Unit hints found in header text → unit option
_UNIT_HINTS = [
    (r"scf\s*/\s*(bbl|stb)", "scf/bbl"), (r"sm3\s*/\s*sm3|sm³\s*/\s*sm³|m3\s*/\s*m3|m³\s*/\s*m³", "Sm³/m³"),
    (r"\bmmscf", "MMSCFD"), (r"\bmscf|\bmcf", "Mscf/d"), (r"\bscf", "scf/d"), (r"nm3|nm³", "Nm³/h"),
    (r"\bbopd|\bbwpd|\bbbl|\bstb", "bbl/d"), (r"m3\s*/\s*h|m³\s*/\s*h", "m³/h"), (r"\bl\s*/\s*s\b", "L/s"),
    (r"\bpsi", "psi"), (r"\bkpa", "kPa"), (r"\bmpa", "MPa"), (r"kgf|kg\s*/\s*cm", "kgf/cm²"), (r"\bbar", "bar"),
    (r"°\s*f|\bdeg\s*f\b|\bdegf\b|fahrenheit", "°F"), (r"°\s*c|\bdeg\s*c\b|celsius", "°C"), (r"\brpm", "rpm"),
    (r"\bmwh", "MWh"), (r"\bkwh", "kWh"), (r"\bwh\b", "Wh"), (r"in\s*/\s*s|ips", "in/s"),
    (r"64", "1/64 in"), (r"\binch|\bin\b|\"", "in"), (r"\bmm\b", "mm"), (r"\bma\b", "mA"),
]


def detect_unit(header: str, field: str):
    """Guess the source unit from the header text, e.g. 'THP (psi)' → psi. None if no confident hint."""
    text = str(header).lower()
    opts = UNITS.get(field, {})
    for pattern, unit in _UNIT_HINTS:
        if unit in opts and re.search(pattern, text):
            return unit
    return None


def infer_columns(raw: pd.DataFrame):
    """Work out what each column means from its NAME and its VALUES.
    Returns ({field: source column}, {field: how it was recognised})."""
    cols = [c for c in raw.columns if str(c).strip() and not str(c).lower().startswith("unnamed")]
    mapped, how, used = {}, {}, set()
    sample = raw.head(300)

    def numeric_ok(col):   # a measurement column must hold numbers
        v = sample[col].dropna()
        return len(v) == 0 or pd.api.types.is_numeric_dtype(v) or _to_number(v).notna().mean() > 0.8

    def not_a_level(col):  # 'DP choke', 'delta P', 'choke UOM' are a difference or a unit label, not the quantity
        t = _norm(col).split()
        return any(w in t for w in ("dp", "delta", "diff", "differential", "uom", "unit", "units", "flag", "code"))

    def take(field, col, reason):
        mapped[field], how[field] = str(col), reason
        used.add(col)

    for col in cols:                                    # 1) known names, with or without unit text
        canon = _LOOKUP.get(_norm(col)) or _LOOKUP.get(_strip_units(col))
        if canon and canon not in mapped and col not in used:
            take(canon, col, "known name")
    for canon, words in _KEYWORDS:                      # 2) keywords inside longer headers (if unambiguous)
        if canon in mapped:
            continue
        hits = [c for c in cols if c not in used and any(w in _strip_units(c) for w in words)
                and (canon in AUX_COLUMNS or not ("ratio" in _norm(c) or "cut" in _norm(c) or "cum" in _norm(c).split()))
                and (canon in ("Date", "Well") or (numeric_ok(c) and not not_a_level(c)))]
        if len(hits) == 1:
            take(canon, hits[0], "keyword in header")
    names = {_strip_units(a): canon for canon, al in ALIASES.items() for a in [canon, *al] if len(_strip_units(a)) > 3}
    for col in cols:                                    # 3) similar spelling (typos, abbreviations)
        if col in used or not_a_level(col):
            continue
        best = difflib.get_close_matches(_strip_units(col), list(names), n=1, cutoff=0.82)
        if best and names[best[0]] not in mapped:
            take(names[best[0]], col, f"similar to '{best[0]}'")
    if "Date" not in mapped:                            # 4) content: a column whose values are dates
        for col in cols:
            if col in used:
                continue
            v = sample[col].dropna()
            if len(v) and not pd.api.types.is_numeric_dtype(v):
                ok = pd.to_datetime(v.astype(str), errors="coerce", format="mixed").notna().mean()
                if ok > 0.9:
                    take("Date", col, "values look like dates")
                    break
            elif pd.api.types.is_datetime64_any_dtype(v):
                take("Date", col, "values are dates")
                break
    if "Well" not in mapped:                            # 5) content: repeating text labels = well IDs
        best, score = None, 0
        for col in cols:
            if col in used:
                continue
            v = raw[col].dropna()
            if len(v) == 0 or pd.api.types.is_numeric_dtype(v) or pd.api.types.is_datetime64_any_dtype(v):
                continue
            n = v.nunique()
            if 1 <= n <= max(2, len(v) // 3):
                sc = 1 - n / len(v) + (0.5 if v.astype(str).str.contains(r"\d").mean() > 0.8 else 0)
                if sc > score:
                    best, score = col, sc
        if best is not None:
            take("Well", best, "repeating labels look like well IDs")
    return mapped, how


def map_columns(raw: pd.DataFrame, mapping: dict | None = None):
    """Rename to FIELDWISE fields. `mapping` ({field: source column}) overrides automatic recognition.
    Returns (frame, {field: source header}, ignored headers)."""
    if mapping is None:
        mapping, _ = infer_columns(raw)
    lookup = {str(c): c for c in raw.columns}
    rename, mapped = {}, {}
    for field, src in mapping.items():
        if src is not None and str(src) in lookup and lookup[str(src)] not in rename:
            rename[lookup[str(src)]] = field
            mapped[field] = str(src)
    frame = raw[list(rename)].rename(columns=rename).copy()
    return frame, mapped, [str(c) for c in raw.columns if c not in rename]


def _to_number(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.mask(s.str.lower().isin(MISSING))
    return pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def derive_fields(frame: pd.DataFrame, rep: dict) -> pd.DataFrame:
    """Use petroleum relationships to fill fields the file does not contain directly."""
    has = lambda c: c in frame.columns and frame[c].notna().any()  # noqa: E731
    if has("Water_Cut"):
        wc = frame["Water_Cut"]
        frame["Water_Cut"] = np.where(wc.max() <= 1.0, wc, wc / 100.0)   # % or fraction
        frame["Water_Cut"] = frame["Water_Cut"].clip(0, 0.999)
    if not has("Oil_Rate") and has("Liquid_Rate") and has("Water_Cut"):
        frame["Oil_Rate"] = frame["Liquid_Rate"] * (1 - frame["Water_Cut"])
        rep["derived"].append("Oil rate = liquid rate × (1 − water cut)")
    if not has("Oil_Rate") and has("Liquid_Rate") and has("Water_Rate"):
        frame["Oil_Rate"] = (frame["Liquid_Rate"] - frame["Water_Rate"]).clip(lower=0)
        rep["derived"].append("Oil rate = liquid rate − water rate")
    if not has("Water_Rate") and has("Oil_Rate"):
        if has("Water_Cut"):
            frame["Water_Rate"] = frame["Oil_Rate"] * frame["Water_Cut"] / (1 - frame["Water_Cut"])
            rep["derived"].append("Water rate = oil rate × WC / (1 − WC)")
        elif has("Liquid_Rate"):
            frame["Water_Rate"] = (frame["Liquid_Rate"] - frame["Oil_Rate"]).clip(lower=0)
            rep["derived"].append("Water rate = liquid rate − oil rate")
    if not has("Gas_Rate") and has("Oil_Rate") and has("GOR"):
        frame["Gas_Rate"] = frame["Oil_Rate"] * frame["GOR"]
        rep["derived"].append("Gas rate = oil rate × GOR")
    return frame.drop(columns=[c for c in AUX_COLUMNS if c in frame.columns])


_INJ_WORDS = {"injection", "injector", "inj", "wi", "gi", "wag", "water injection", "gas injection", "disposal", "wdw"}
_PROD_WORDS = {"production", "producer", "prod", "op", "oil producer", "oil", "gp", "gas producer", "producing"}


SIG = "Sig: "          # prefix for extra continuous signals carried from the file (e.g. choke ΔP, wellhead temperature)
_LEAKY = ("oil", "gas", "water", "wat", "liquid", "fluid", "gor", "wor", "glr", "cut", "bsw", "cum", "vol", "rate",
          "prod", "inj", "wi", "gi", "bopd", "bwpd", "mscf", "bbl", "stb", "sm3", "m3", "total", "allocated", "code", "id")


def extra_signals(raw: pd.DataFrame, used: set, max_n: int = 12) -> list:
    """Other numeric columns that look like continuous measurements (temperatures, ΔP, annulus pressure …).
    Anything that could be a production volume / ratio is excluded, so the model can never peek at the answer."""
    out = []
    wells = None
    for col in raw.columns:
        if col in used or str(col).strip() == "" or len(out) >= max_n:
            continue
        toks = set(_strip_units(col).split()) | set(_norm(col).split())
        if toks & set(_LEAKY) or any(t.startswith(("oil", "gas", "water", "liquid")) for t in toks):
            continue
        v = _to_number(raw[col])
        if v.notna().mean() < 0.3 or v.nunique() < 10:
            continue
        if wells is None:
            wcol = next((c for c in raw.columns if str(c) == "Well"), None)
            wells = raw[wcol].astype(str) if wcol is not None else pd.Series("all", index=raw.index)
        if v.groupby(wells).nunique().max() <= 3:          # constant per well → an ID or a property, not a signal
            continue
        out.append(col)
    return out


def split_injection_rows(raw: pd.DataFrame):
    """Real exports often mix injector rows into the production table (e.g. FLOW_KIND = injection / production,
    WELL_TYPE = WI / OP). Returns (rows to analyse, column used, rows removed) — production rows only."""
    best = None
    for col in raw.columns:
        v = raw[col]
        if pd.api.types.is_numeric_dtype(v) or pd.api.types.is_datetime64_any_dtype(v):
            continue
        low = v.dropna().astype(str).str.strip().str.lower()
        if low.empty or low.nunique() > 6:
            continue
        kinds = set(low.unique())
        inj, prod = kinds & _INJ_WORDS, kinds & _PROD_WORDS
        if inj and prod and kinds <= (_INJ_WORDS | _PROD_WORDS):
            score = 2 if ("flow" in _norm(col) or "injection" in kinds) else 1   # FLOW_KIND beats WELL_TYPE
            if best is None or score > best[0]:
                best = (score, col, inj)
    if best is None:
        return raw, None, 0
    _, col, inj = best
    drop = raw[col].astype(str).str.strip().str.lower().isin(inj)
    return raw.loc[~drop], str(col), int(drop.sum())


def clean_data(raw: pd.DataFrame, unit_choices: dict, mapping: dict | None = None):
    """Auditable cleaning. Only Date, Well and Oil rate are strictly required (oil may be derived).
    Returns (clean frame | None, report)."""
    raw, inj_col, inj_rows = split_injection_rows(raw)
    frame, mapped, ignored = map_columns(raw, mapping)
    extras = extra_signals(raw, set(map(str, mapped.values())) | set(mapped.values()))
    for col in extras:
        frame[SIG + str(col)] = _to_number(raw[col]).to_numpy()
    rep = {"injection_rows_removed": (inj_col, inj_rows) if inj_rows else None, "input_rows": len(raw), "mapped": mapped, "ignored": ignored, "bad_keys": 0, "duplicates": 0,
           "invalid": 0, "imputed": {}, "capped": 0, "dropped_optional": [], "conversions": [],
           "injection_missing": False, "derived": [], "absent": []}
    missing = [c for c in ["Date", "Well"] if c not in frame.columns]
    if missing:
        rep["missing"] = missing
        return None, rep

    try:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", format="mixed")
    except (TypeError, ValueError):
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    well = frame["Well"].astype(str).str.strip().str.upper()
    frame["Well"] = well.mask(well.isin({"", "NAN", "NONE", "<NA>", "NULL"}))
    n0 = len(frame)
    frame = frame.dropna(subset=["Date", "Well"])
    rep["bad_keys"] = n0 - len(frame)

    for col in [c for c in NUMERIC_COLUMNS + AUX_COLUMNS if c in frame.columns]:
        frame[col] = _to_number(frame[col])

    for col, unit in unit_choices.items():
        if col not in frame.columns or unit not in UNITS.get(col, {}):
            continue
        target = "m³/d" if (col == "Injection_Rate" and unit in _LIQ) else STD_UNIT[col]
        if unit == target:
            continue
        offset = -32.0 * 5.0 / 9.0 if unit == "°F" else 0.0
        frame[col] = frame[col] * UNITS[col][unit] + offset
        rep["conversions"].append({"Field": col, "From": unit, "To": target})

    frame = derive_fields(frame, rep)
    if "Oil_Rate" not in frame.columns or frame["Oil_Rate"].isna().all():
        rep["missing"] = ["Oil_Rate"]
        return None, rep
    # Optional measurements that are absent: water/gas → 0 (stated), pressures → left empty (models skip them)
    for col in ["Water_Rate", "Gas_Rate"]:
        if col not in frame.columns:
            frame[col] = 0.0
            rep["absent"].append(col)
    if "Injection_Rate" not in frame.columns:
        frame["Injection_Rate"] = 0.0
        rep["injection_missing"] = True
    numeric = [c for c in NUMERIC_COLUMNS if c in frame.columns] + [c for c in frame.columns if str(c).startswith(SIG)]
    rep["extra_signals"] = [c[len(SIG):] for c in frame.columns if str(c).startswith(SIG)]
    miss_frac = {c: float(frame[c].isna().mean()) for c in numeric}

    for col in [c for c in RATE_COLUMNS + PRESSURE_COLUMNS + SETPOINTS if c in frame.columns]:
        bad = frame[col] < 0
        rep["invalid"] += int(bad.sum())
        frame.loc[bad, col] = np.nan
    # A pressure of exactly 0 while the well is flowing is a dead gauge, not a real reading
    flowing = (frame["Hours_On"] >= 12) if "Hours_On" in frame.columns else (frame["Oil_Rate"] > 0)
    rep["dead_gauge"] = {}
    for col in [c for c in PRESSURE_COLUMNS if c in frame.columns]:
        dead = flowing & (frame[col] == 0)
        if dead.sum():
            rep["dead_gauge"][col] = int(dead.sum())
            frame.loc[dead, col] = np.nan

    for col in [c for c in OPTIONAL_COLUMNS if c in frame.columns]:
        best_well = frame[col].notna().groupby(frame["Well"]).mean().max() if len(frame) else 0
        if miss_frac.get(col, 0) > 0.6 and not best_well >= 0.4:    # e.g. ESP data exists only on the ESP wells
            frame = frame.drop(columns=col)
            numeric.remove(col)
            rep["dropped_optional"].append(col)

    n0 = len(frame)
    frame = frame.drop_duplicates()
    if frame.duplicated(["Well", "Date"]).any():
        frame = frame.groupby(["Well", "Date"], as_index=False)[numeric].mean()
    rep["duplicates"] = n0 - len(frame)
    frame = frame.sort_values(["Well", "Date"]).reset_index(drop=True)

    # Missing values, by the nature of each measurement (fully vectorised):
    #   • well-test rates (oil / water / gas): left exactly as measured — no interpolation, no fill
    #   • set-points (choke): carried forward — the setting persists until someone changes it
    #   • continuous signals (pressures, lift gas, pump data): only gaps of ≤ GAP_LIMIT readings are bridged
    wells = frame["Well"]
    pos = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)
    rep["kept_as_measured"] = {}
    for col in numeric:
        n_miss = int(frame[col].isna().sum())
        if col in TEST_MEASURED:
            rep["kept_as_measured"][col] = {"measured": int(frame[col].notna().sum()), "not_measured": n_miss}
            continue
        if not n_miss:
            continue
        if col in SETPOINTS:
            frame[col] = frame.groupby("Well")[col].ffill()
            frame[col] = frame.groupby("Well")[col].bfill()
        else:
            ok = frame[col].notna()
            prev_v = frame[col].groupby(wells).ffill()
            next_v = frame[col].groupby(wells).bfill()
            prev_p = pos.where(ok).groupby(wells).ffill()
            next_p = pos.where(ok).groupby(wells).bfill()
            span = (next_p - prev_p).replace(0, np.nan)
            interp = prev_v + (next_v - prev_v) * (pos - prev_p) / span
            short_gap = (span - 1) <= GAP_LIMIT
            near_edge = ((pos - prev_p) <= GAP_LIMIT) | ((next_p - pos) <= GAP_LIMIT)
            fill = interp.where(short_gap)
            fill = fill.fillna(prev_v.where(next_p.isna() & near_edge)).fillna(next_v.where(prev_p.isna() & near_edge))
            frame[col] = frame[col].fillna(fill)
        rep["imputed"][col] = n_miss - int(frame[col].isna().sum())

    # Per-well 3×IQR capping (vectorised) — measurements only, never setpoints
    cap_cols = [c for c in numeric if c not in SETPOINTS]
    if cap_cols:
        grp = frame.groupby("Well")[cap_cols]
        q = grp.quantile([0.25, 0.75])
        q1 = q.xs(0.25, level=1).reindex(wells).to_numpy()
        q3 = q.xs(0.75, level=1).reindex(wells).to_numpy()
        iqr = q3 - q1
        enough = (grp.transform("count").to_numpy() >= 10) & (iqr > 0)
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        nonneg = np.array([c in RATE_COLUMNS + PRESSURE_COLUMNS for c in cap_cols])
        lo = np.where(nonneg, np.maximum(lo, 0.0), lo)
        vals = frame[cap_cols].to_numpy(dtype=float)
        flowing = (frame["Hours_On"] >= 23.5).to_numpy()[:, None] if "Hours_On" in frame.columns \
            else np.ones((len(frame), 1), bool)
        # zeros are real "off" states and shut-in build-up is real physics → only cap readings taken while flowing
        out = enough & flowing & ((vals > hi) | ((vals < lo) & (vals != 0)))
        rep["capped"] = int(out.sum())
        frame[cap_cols] = np.where(out, np.clip(vals, lo, hi), vals)

    for col in PRESSURE_COLUMNS:                     # absent pressures: explicit empty columns
        if col not in frame.columns:
            frame[col] = np.nan
            rep["absent"].append(col)
    # Keep every row that holds at least one real measurement (sensor-only days are valuable between well tests)
    frame = frame.dropna(subset=[c for c in numeric if c in frame.columns], how="all").reset_index(drop=True)
    # Wells that never produced oil (injectors, observation wells) are set aside from production analysis
    producing = (frame["Oil_Rate"] > 0).groupby(frame["Well"]).any()
    rep["non_producers"] = sorted(producing.index[~producing])
    if rep["non_producers"] and producing.any():
        frame = frame[frame["Well"].isin(producing.index[producing])].reset_index(drop=True)
    n_oil = int(frame["Oil_Rate"].notna().sum())
    if n_oil < 10:
        rep["error"] = f"Only {n_oil} oil-rate measurements found — at least 10 are needed."
        return None, rep
    measured = frame["Oil_Rate"].notna()
    rep["test_mode"] = bool(measured.mean() < TEST_MODE_THRESHOLD)
    gaps = frame.loc[measured].groupby("Well")["Date"].diff().dt.total_seconds() / 86400
    rep["test_gap_days"] = float(gaps[gaps > 0].median()) if (gaps > 0).any() else np.nan
    rep["oil_measured"] = n_oil
    if len(frame) < 30:
        rep["error"] = f"Only {len(frame)} usable rows after cleaning — at least 30 are needed."
        return None, rep
    if frame["Date"].nunique() < 10:
        rep["error"] = "At least 10 distinct dates are needed to train and validate the models."
        return None, rep
    rep["output_rows"] = len(frame)
    return frame, rep


# ---- 2b. AUTO-STRUCTURING — turn any sheet layout into one tidy table (one row = one well on one date) ----
_UNIT_TOKEN = re.compile(
    r"^[\(\[]?\s*(bbl|stb|bopd|bwpd|blpd|m3|m³|sm3|sm³|nm3|scf|mscf|mmscf|mmscfd|mcf|psi|psig|psia|bar|barg|bara|"
    r"kpa|mpa|kgf|kg/cm2|°c|°f|degc|degf|c|f|hz|rpm|a|amp|amps|ma|kwh|mwh|wh|mm/s|in/s|%|frac|fraction|in|inch|"
    r"mm|/64|64ths|1/64|d|day|days|hr|h|unit|units)\b", re.I)
_LABEL_ROW = re.compile(r"^\s*(well(?:\s*(?:name|id|no\.?|number))?|asset|wellbore|string)\s*[:=#\-–]\s*(.+?)\s*$", re.I)
_TOTAL_ROW = re.compile(r"^\s*(grand\s+)?(total|sum|average|avg|mean|subtotal|cumulative)\b", re.I)
_WELL_TOKEN = r"(?:[A-Za-z]{1,8}[-_ #]?\d{1,4}[A-Za-z]?|well\s*[-_#]?\s*\d{1,4})"
_NOT_WELL = re.compile(r"^\s*(co2|h2s|n2|o2|h2o|so2|no2|nox|ch4|c[1-9]\+?|pm\d+|q[1-4]|h[12]|fy\s*\d+|"
                       r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*[-_ ]?\s*\d{2,4}|"
                       r"[a-z]+\s*[-_ ]?\s*(19|20)\d\d|col_\d+|unnamed.*)\s*$", re.I)
_WELL_ONLY_RX = re.compile(rf"^\s*{_WELL_TOKEN}\s*$", re.I)


class _WellOnly:
    @staticmethod
    def match(s):
        return bool(_WELL_ONLY_RX.match(str(s))) and not _NOT_WELL.match(str(s))


_WELL_ONLY = _WellOnly()
_WELL_FIRST = re.compile(rf"^\s*({_WELL_TOKEN})\s*[-_:|/.]+\s*(.+?)\s*$", re.I)
_WELL_LAST = re.compile(rf"^\s*(.+?)\s*[-_:|/.]+\s*({_WELL_TOKEN})\s*$", re.I)
_WELL_SPACE_FIRST = re.compile(rf"^\s*({_WELL_TOKEN})\s+(.+?)\s*$", re.I)


def _is_number(x) -> bool:
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return np.isfinite(x)
    try:
        float(str(x).replace(",", "").strip())
        return True
    except ValueError:
        return False


def _is_dateish(x) -> bool:
    if isinstance(x, (pd.Timestamp, np.datetime64)) or hasattr(x, "year"):
        return True
    s = str(x).strip()
    return bool(re.match(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", s) or
                re.match(r"^\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{2,4}", s) or re.match(r"^[A-Za-z]{3,9}[-\s]\d{2,4}$", s))


def _row_profile(row) -> tuple:
    """(fill fraction, header-likeness, data-likeness) of one row."""
    vals = [v for v in row if pd.notna(v) and str(v).strip() != ""]
    if not vals:
        return 0.0, 0.0, 0.0
    dates = sum(1 for v in vals if _is_dateish(v))
    text = sum(1 for v in vals if not _is_number(v) and not _is_dateish(v))
    num = sum(1 for v in vals if _is_number(v)) + dates
    headerish = text / len(vals)
    if len(vals) >= 5 and dates / len(vals) > 0.7:     # a row of dates = header of a transposed sheet
        headerish = max(headerish, dates / len(vals))
    return len(vals) / len(row), headerish, num / len(vals)


def _str_matrix(df: pd.DataFrame) -> np.ndarray:
    """All cells as stripped strings ('' for empty) — column-wise, so it stays fast on large sheets."""
    return np.column_stack([df[c].astype(str).str.strip().where(df[c].notna(), "").to_numpy()
                            for c in df.columns]) if df.shape[1] else np.empty((len(df), 0), dtype=object)


def _first_cells(sv: np.ndarray, k: int = 2) -> list:
    """First k non-empty cells of every row (vectorised)."""
    out, mask = [], sv != ""
    rows = np.arange(len(sv))
    for _ in range(k):
        has = mask.any(axis=1)
        idx = mask.argmax(axis=1)
        out.append(pd.Series(np.where(has, sv[rows, idx], ""), dtype=str))
        mask = mask.copy()
        mask[rows, idx] = False
    return out


def _find_header(grid: pd.DataFrame, log: list, where: str):
    """Locate the header row (skipping titles/blank lines), a units row below it, or a two-level header."""
    g = grid.dropna(how="all").dropna(axis=1, how="all")
    if g.empty:
        return None
    g = g.reset_index(drop=True)
    g.columns = range(g.shape[1])
    # Section labels such as "Well: W-001" above each block → carried down as the well ID
    sv = _str_matrix(g)
    first_cell = _first_cells(sv, 1)[0].set_axis(g.index)
    n_cells = pd.Series((sv != "").sum(axis=1), index=g.index)
    lab = first_cell.str.extract(_LABEL_ROW)[1].where(n_cells <= 2)
    bare = first_cell.where((n_cells == 1) & first_cell.map(_WELL_ONLY.match))   # a lone "W-001" line
    lab = lab.fillna(bare) if bare.notna().sum() >= 2 else lab
    block_well = None
    if lab.notna().sum() >= 2:
        block_well = lab.ffill()
        g = g[lab.isna()]
        log.append(f"{where}: found {lab.notna().sum()} section labels (e.g. '{first_cell[lab.notna()].iloc[0]}') "
                   "— used them as the well ID for the rows below each label")
    head = g.head(40)
    profiles = [_row_profile(r) for r in head.itertuples(index=False)]
    best, best_score = 0, -1.0
    for i, (fill, text, _) in enumerate(profiles):
        below = [p[2] if p[1] < 0.7 else 0.3 for p in profiles[i + 1:i + 6] if p[0] > 0]
        data_below = np.mean(below) if below else 0
        score = fill * text * (0.4 + data_below) - 0.002 * i
        if fill >= 0.4 and text >= 0.5 and score > best_score:
            best, best_score = i, score
    names = [str(v).strip() if pd.notna(v) and str(v).strip() else "" for v in g.iloc[best]]
    start = best + 1
    # Two-level header where the upper level is sparse (merged cells, e.g. well names above Oil/Water/Gas)
    if best > 0:
        up = [str(v).strip() if pd.notna(v) and str(v).strip() else "" for v in g.iloc[best - 1]]
        filled = [j for j, v in enumerate(up) if v]
        if len(filled) >= 2 and min(filled) > 0 and all(not _is_number(up[j]) for j in filled) \
                and len({up[j] for j in filled}) >= 2:
            top = pd.Series(up).replace("", np.nan).ffill().fillna("").tolist()
            names = [f"{t} | {n}" if t and n and t != n else (n or t) for t, n in zip(top, names)]
            log.append(f"{where}: combined a two-level header (e.g. well name over measurement)")
            best -= 1
    if best > 0:
        log.append(f"{where}: skipped {best} title/blank row(s) above the header")
    # Units row directly under the header (e.g. 'bbl/d', 'psi') → merge into names
    if start < len(g):
        nxt = [v for v in g.iloc[start] if pd.notna(v) and str(v).strip()]
        if nxt and sum(bool(_UNIT_TOKEN.match(str(v).strip())) for v in nxt) / len(nxt) >= 0.5:
            units = [str(v).strip() if pd.notna(v) else "" for v in g.iloc[start]]
            names = [f"{n} ({u.strip('()[]')})" if n and u else (n or u) for n, u in zip(names, units)]
            start += 1
            log.append(f"{where}: merged a units row into the column names")
        # Two-level header: a sparse top row (merged cells, e.g. well names) above a full row of labels
        elif start < len(g) and best_score > 0:
            fill2, text2, _ = _row_profile(g.iloc[start])
            fill1 = profiles[best][0]
            if text2 >= 0.6 and fill2 > fill1 + 0.15:
                top = pd.Series(names).replace("", np.nan).ffill().fillna("").tolist()
                sub = [str(v).strip() if pd.notna(v) else "" for v in g.iloc[start]]
                names = [f"{t} | {s}" if t and s and t != s else (s or t) for t, s in zip(top, sub)]
                start += 1
                log.append(f"{where}: combined a two-level header (e.g. well name over measurement)")
    names = [n or f"col_{j}" for j, n in enumerate(names)]
    seen = {}
    for j, n in enumerate(names):                     # de-duplicate names
        if n in seen:
            seen[n] += 1
            names[j] = f"{n}.{seen[n]}"
        else:
            seen[n] = 0
    data = g.iloc[start:].copy()
    g_index = data.index  # noqa: F841 — original positions kept so section labels can be re-attached
    data.columns = names
    # Remove repeated header lines, totals/averages rows and empty rows
    first = data.columns[0]
    dsv = _str_matrix(data)
    is_rep = pd.Series((dsv == np.array(names, dtype=object)).mean(axis=1) > 0.6, index=data.index)
    c1, c2 = _first_cells(dsv, 2)
    is_tot = pd.Series((c1.str.match(_TOTAL_ROW) | c2.str.match(_TOTAL_ROW)).to_numpy(), index=data.index)
    if is_rep.any():
        log.append(f"{where}: removed {int(is_rep.sum())} repeated header row(s)")
    if is_tot.any():
        log.append(f"{where}: removed {int(is_tot.sum())} total/average row(s)")
    data = data[~is_rep & ~is_tot].dropna(how="all")
    if block_well is not None and "Well" not in data.columns:
        data.insert(0, "Well", block_well.reindex(data.index).astype(str).str.strip())
    data = data.loc[:, [c for c in data.columns if data[c].notna().any() and not str(c).startswith("col_")]
                    or list(data.columns)]
    _ = first
    return data.reset_index(drop=True)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Numbers stored as text → numbers (only when most values parse)."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        num = pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
        if s.notna().sum() and num.notna().sum() / s.notna().sum() >= 0.8:
            out[c] = num
    return out


def smart_dates(s: pd.Series):
    """Parse dates robustly: Excel serial numbers, day-first vs month-first detection, mixed formats.
    Returns (parsed series, description)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, "native dates"
    nn = s.dropna()
    if len(nn) and nn.map(lambda x: isinstance(x, pd.Timestamp) or hasattr(x, "isoformat")).mean() > 0.9:
        return pd.to_datetime(s, errors="coerce"), "native dates"
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9 and num.between(20000, 80000).mean() > 0.9:
        return pd.to_datetime(num, unit="D", origin="1899-12-30"), "Excel serial numbers"
    if num.notna().mean() > 0.9 and num.between(1900, 2100).mean() > 0.9:
        return pd.to_datetime(num.astype("Int64").astype(str), format="%Y", errors="coerce"), "years"
    txt = s.astype(str).str.strip()
    parts = txt.str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")
    dayfirst = None
    if parts.notna().all(axis=1).mean() > 0.5:
        a, b = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
        if (a > 12).any() and not (b > 12).any():
            dayfirst = True
        elif (b > 12).any() and not (a > 12).any():
            dayfirst = False
        else:
            dayfirst = True   # ambiguous everywhere → international convention (dd/mm)
    try:
        out = pd.to_datetime(txt, errors="coerce", format="mixed", dayfirst=bool(dayfirst))
    except (TypeError, ValueError):
        out = pd.to_datetime(txt, errors="coerce", dayfirst=bool(dayfirst))
    how = {True: "day-first (dd/mm)", False: "month-first (mm/dd)", None: "text dates"}[dayfirst]
    return out, how


def _date_column(df: pd.DataFrame):
    m, _ = infer_columns(df)
    if "Date" in m:
        return m["Date"]
    for c in df.columns:
        if df[c].notna().any() and df[c].dropna().head(50).map(_is_dateish).mean() > 0.8:
            return c
    return None


def _build_date(df: pd.DataFrame, log: list, where: str) -> pd.DataFrame:
    """Year / Month / Day columns → one Date column."""
    low = {_norm(c): c for c in df.columns}
    y = next((low[k] for k in ("year", "yr", "yyyy") if k in low), None)
    mth = next((low[k] for k in ("month", "mon", "mm", "month name") if k in low), None)
    if y is None or mth is None:
        return df
    dcol = next((low[k] for k in ("day", "dd") if k in low), None)
    months = pd.to_numeric(df[mth], errors="coerce")
    if months.isna().mean() > 0.5:
        months = pd.to_datetime(df[mth].astype(str).str[:3], format="%b", errors="coerce").dt.month
    day = pd.to_numeric(df[dcol], errors="coerce").fillna(1) if dcol else 1
    df = df.copy()
    df["Date"] = pd.to_datetime(dict(year=pd.to_numeric(df[y], errors="coerce"), month=months, day=day), errors="coerce")
    log.append(f"{where}: built Date from {', '.join(str(c) for c in (y, mth, dcol) if c is not None)} columns")
    return df.drop(columns=[c for c in (y, mth, dcol) if c is not None])


def _split_well_header(name: str):
    """'W-001 Oil (bbl/d)' → ('W-001', 'Oil (bbl/d)');  'Oil | W-001' → ('W-001', 'Oil')."""
    s = str(name)
    if " | " in s:
        a, b = [p.strip() for p in s.split(" | ", 1)]
        if _WELL_ONLY.match(a):
            return a, b
        if _WELL_ONLY.match(b):
            return b, a
    for rx, wi, mi in ((_WELL_FIRST, 1, 2), (_WELL_LAST, 2, 1), (_WELL_SPACE_FIRST, 1, 2)):
        m = rx.match(s)
        if m and not _WELL_ONLY.match(m.group(mi)) and not _NOT_WELL.match(m.group(wi)):
            measure = m.group(mi)
            if _LOOKUP.get(_strip_units(measure)) or any(w in _strip_units(measure) for _, ws in _KEYWORDS for w in ws):
                return m.group(wi), measure
    if _WELL_ONLY.match(s):
        return s.strip(), None
    return None, None


def _unpivot_wide(df: pd.DataFrame, log: list, where: str, measure_name: str) -> pd.DataFrame:
    """Columns like 'W-001 Oil', 'W-002 Oil' (or just 'W-001', 'W-002') → rows with a Well column."""
    m, _ = infer_columns(df)
    if "Well" in m and m["Well"] in df.columns and not _WELL_ONLY.match(str(m["Well"])):
        return df
    parsed = {c: _split_well_header(c) for c in df.columns}
    wide = {c: p for c, p in parsed.items() if p[0]}
    wells = {p[0].upper() for p in wide.values()}
    if len(wells) < 2 or len(wide) < 2:
        return df
    ids = [c for c in df.columns if c not in wide]
    parts = []
    for well in sorted(wells, key=str):
        cols = {c: (p[1] or measure_name) for c, p in wide.items() if p[0].upper() == well}
        part = df[ids + list(cols)].rename(columns=cols).copy()
        part.insert(0, "Well", well.replace("_", "-").replace(" ", "-").upper())
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    meas = sorted({str(p[1] or measure_name) for p in wide.values()})
    log.append(f"{where}: wide layout — unpivoted {len(wells)} wells × {len(meas)} measure(s) "
               f"({', '.join(meas[:5])}{'…' if len(meas) > 5 else ''}) into rows")
    return out


def _maybe_transpose(df: pd.DataFrame, log: list, where: str) -> pd.DataFrame:
    """Dates running across the top (one row per well/measure) → dates down the side."""
    heads = list(df.columns[1:])
    if len(heads) >= 5 and np.mean([_is_dateish(h) for h in heads]) > 0.7:
        key = df.columns[0]
        t = df.set_index(key).T.reset_index().rename(columns={"index": "Date"})
        t.columns = [str(c) for c in t.columns]
        log.append(f"{where}: dates ran across columns — transposed so each row is one date")
        return t
    return df


def _signature(df: pd.DataFrame) -> frozenset:
    return frozenset(_strip_units(c) for c in df.columns)


def auto_structure(grids: dict, file_stem: str):
    """Any workbook/CSV layout → one tidy table. Returns (table | None, log)."""
    log, frames = [], {}
    for name, grid in grids.items():
        where = f"sheet '{name}'" if len(grids) > 1 else "file"
        t = _find_header(grid, log, where)
        if t is None or len(t) < 3 or t.shape[1] < 2:
            if len(grids) > 1:
                log.append(f"{where}: skipped (empty or not a data table)")
            continue
        t = _maybe_transpose(t, log, where)
        t = _build_date(t, log, where)
        t = _coerce_types(t)
        t = _unpivot_wide(t, log, where, measure_name=name if len(grids) > 1 else "Value")
        dcol = _date_column(t)
        if dcol is None:
            log.append(f"{where}: skipped (no date column — looks like notes, a guide or a lookup table)")
            continue
        parsed, how = smart_dates(t[dcol])
        t = t.assign(**{str(dcol): parsed}).rename(columns={dcol: "Date"})
        t = t[t["Date"].notna()]
        log.append(f"{where}: dates read as {how}")
        frames[name] = t
    if not frames:
        return None, log + ["No sheet contained a recognisable time-series table."]

    # One sheet per well? (no well column anywhere, several similar sheets) → sheet name becomes the Well
    no_well = {n: f for n, f in frames.items() if "Well" not in infer_columns(f)[0]}
    if len(no_well) >= 2 and len(no_well) == len(frames):
        sigs = {_signature(f) for f in frames.values()}
        if len(sigs) <= max(1, len(frames) // 2):
            for n in frames:
                frames[n] = frames[n].assign(Well=str(n).strip().upper())
            log.append(f"{len(frames)} sheets share the same layout and have no well column — "
                       "treated each sheet as one well (sheet name = well ID)")
    elif len(frames) == 1 and no_well:
        n = next(iter(frames))
        wid = (str(n) if len(grids) > 1 and not str(n).lower().startswith("sheet") else file_stem).strip().upper()
        frames[n] = frames[n].assign(Well=wid)
        log.append(f"no well column — treated the data as a single well '{wid}'")

    # Normalise the Well column name, then stack same-layout sheets and join different ones on Date + Well
    for n, f in frames.items():
        wcol = infer_columns(f)[0].get("Well")
        if wcol and wcol != "Well" and wcol in f.columns:
            frames[n] = f.rename(columns={wcol: "Well"})
        frames[n]["Well"] = frames[n]["Well"].astype(str).str.strip().str.upper()
    # Same well written differently ("ALPHA 101", "Alpha-101", "ALPHA_101") → one ID (the most common spelling)
    allw = pd.concat([f["Well"] for f in frames.values()], ignore_index=True)
    key = allw.str.replace(r"[^A-Z0-9]", "", regex=True)
    canon = allw.groupby(key).agg(lambda s: s.value_counts().index[0])
    variants = allw.groupby(key).nunique()
    if (variants > 1).any():
        ex = allw[key == variants[variants > 1].index[0]].unique()[:2]
        log.append(f"matched well names written differently (e.g. '{ex[0]}' = '{ex[-1]}')")
    for n in frames:
        k = frames[n]["Well"].str.replace(r"[^A-Z0-9]", "", regex=True)
        frames[n]["Well"] = k.map(canon).fillna(frames[n]["Well"])
    # Daily (or coarser) data + timestamps with a time of day (e.g. well tests at 10:30) → align to the day
    gap_days = {n: (f.sort_values("Date").groupby("Well")["Date"].diff().dt.total_seconds() / 86400).median()
                for n, f in frames.items()}
    base_n = max(frames, key=lambda n: len(frames[n]))
    if pd.notna(gap_days.get(base_n)) and gap_days[base_n] >= 0.9:
        timed = [n for n, f in frames.items() if (f["Date"].dt.normalize() != f["Date"]).any()]
        for n in timed:
            frames[n]["Date"] = frames[n]["Date"].dt.normalize()
        if timed:
            log.append(f"aligned timestamps with a time of day to their date in: {', '.join(map(str, timed))}")
    groups = []
    for n, f in frames.items():
        for g in groups:
            a, b = _signature(f), _signature(g[0][1])
            if len(a & b) / max(len(a | b), 1) >= 0.7:
                g.append((n, f))
                break
        else:
            groups.append([(n, f)])
    tables = []
    for g in groups:
        if len(g) > 1:
            log.append(f"stacked {len(g)} sheets with the same layout: {', '.join(str(n) for n, _ in g)}")
        tables.append(pd.concat([f for _, f in g], ignore_index=True))
    tables.sort(key=len, reverse=True)                  # the biggest table (e.g. daily operations) is the base
    out = tables[0]
    for t in tables[1:]:
        keys = ["Date", "Well"]
        overlap = [c for c in t.columns if c in out.columns and c not in keys]
        t = t.drop(columns=overlap)
        t = t.groupby(keys, as_index=False).mean(numeric_only=True)
        out = out.merge(t, on=keys, how="outer")
        log.append(f"joined a second table on Date + Well (added {', '.join(map(str, t.columns.drop(keys)[:6]))})")
    out = out.sort_values(["Well", "Date"]).reset_index(drop=True)
    dup = out.duplicated(["Well", "Date"]).mean()
    if dup > 0.05:
        log.append(f"⚠ CHECK: {dup:.0%} of rows share the same well and date — the layout may hide a grouping "
                   "(e.g. several wells in one block). Review the Before/After preview.")
    log.append(f"result: {len(out):,} rows × {out.shape[1]} columns, {out['Well'].nunique()} well(s)")
    return out, log


def read_grid_csv(data: bytes) -> pd.DataFrame:
    """CSV as a raw grid: sniff the delimiter, tolerate title lines with fewer fields, keep everything as text."""
    import csv
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sample = text[:50000]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delim = ","
    lines = sample.splitlines()[:2000]
    width = max((len(r) for r in csv.reader(lines, delimiter=delim)), default=1)
    return pd.read_csv(io.StringIO(text), sep=delim, header=None, names=range(width), dtype=str,
                       skip_blank_lines=False, engine="c")


def _read_excel(data: bytes) -> dict:
    """Every sheet as a raw grid (no header assumed). Fastest available engine: calamine → openpyxl."""
    last = None
    for engine in ("calamine", "openpyxl"):
        try:
            return pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, engine=engine)
        except (ImportError, ValueError) as exc:
            last = exc
    if isinstance(last, ImportError):
        raise ValueError("Excel support is not installed on this server (add 'openpyxl' to requirements.txt), "
                         "or upload the data as CSV.") from last
    raise ValueError(f"Could not read the workbook ({last}). Try saving it as .xlsx or CSV.") from last


# Heavy results are cached with cache_resource and keyed by a cheap file token (not by hashing the data):
# no hashing of large DataFrames and no copying of results on every Streamlit rerun.
@st.cache_resource(show_spinner=False, max_entries=3)
def read_file(key: str, _data: bytes, name: str) -> dict:
    """Return {sheet name: raw grid}. A CSV gives a single 'data' sheet. Nothing about the layout is assumed."""
    if not _data:
        raise ValueError("The file is empty.")
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        sheets = {str(k): v for k, v in _read_excel(_data).items() if not v.dropna(how="all").empty}
        if not sheets:
            raise ValueError("The workbook has no non-empty sheets.")
        return sheets
    grid = read_grid_csv(_data)
    if grid.dropna(how="all").empty:
        raise ValueError("No readable data found in the CSV.")
    return {"data": grid}


@st.cache_resource(show_spinner=False, max_entries=3)
def structure_cached(key: str, _grids: dict, stem: str):
    """Turn any layout into one tidy table using the built-in rules."""
    tidy, log = auto_structure(_grids, stem)
    return tidy, log, None


@st.cache_resource(show_spinner=False, max_entries=3)
def clean_cached(key: str, _tidy: pd.DataFrame, units: tuple, mapping: tuple):
    return clean_data(_tidy, dict(units), dict(mapping))


def template_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02"], "Well": ["W-001", "W-001"], "Oil_Rate": [85.0, 84.6],
        "Water_Rate": [174.0, 175.2], "Gas_Rate": [7200.0, 7150.0], "Wellhead_Pressure": [42.0, 41.8],
        "Bottomhole_Pressure": [118.0, 117.6], "Injection_Rate": [14000, 14200], "Choke_Size": [28, 28],
        "Pump_Vibration": [2.1, 2.2], "Motor_Temperature": [72.0, 72.4], "Motor_Current": [54.0, 53.5],
        "Pump_Frequency": [48.0, 48.0], "Energy_kWh": [390.0, 388.0]})


def schema_frame() -> pd.DataFrame:
    return pd.DataFrame([{"Column": c,
                          "Needed": "Required" if c in REQUIRED else "Recommended" if c in CORE_COLUMNS else "Optional",
                          "Unit": STD_UNIT.get(c, "% or fraction" if c == "Water_Cut" else "—"), "Meaning": DESCRIPTIONS[c]}
                         for c in CORE_COLUMNS + OPTIONAL_COLUMNS + AUX_COLUMNS])


@st.cache_data(show_spinner=False)
def template_xlsx() -> bytes | None:
    """Excel template; None if no Excel writer is installed (the CSV template still works)."""
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            template_frame().to_excel(xw, index=False, sheet_name="Production")
            schema_frame().to_excel(xw, index=False, sheet_name="Guide")
        return buf.getvalue()
    except ImportError:
        return None


@dataclass
class Field:
    df: pd.DataFrame
    name: str
    report: dict
    lift_is_gas: bool
    key: str = ""
    raw: pd.DataFrame | None = None      # the sheet exactly as uploaded (all columns) — used by Ask FIELDWISE

    @property
    def sensors(self):
        return [c for c in SENSOR_COLUMNS if c in self.df.columns]

    @property
    def has_choke(self):
        return "Choke_Size" in self.df.columns

    @property
    def has_lift(self):
        return not self.report.get("injection_missing") and self.df["Injection_Rate"].sum() > 0

    @property
    def inj_unit(self):
        return "Sm³/d" if self.lift_is_gas else "m³/d"


# =============================================================================
# 3. ANALYTICS
# =============================================================================
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["Well", "Date"]).copy()
    d["Liquid_Rate"] = d["Oil_Rate"] + d["Water_Rate"]              # NaN on days without a well test (not 0)
    d["Water_Cut"] = 100 * d["Water_Rate"] / d["Liquid_Rate"].replace(0, np.nan)
    d["Delta_P"] = d["Bottomhole_Pressure"] - d["Wellhead_Pressure"]
    d["Days"] = (d["Date"] - d.groupby("Well")["Date"].transform("min")).dt.total_seconds() / 86400
    return d.reset_index(drop=True)


def is_test_mode(df: pd.DataFrame) -> bool:
    """True when oil is only measured on some rows (periodic well tests) rather than every day."""
    return bool(df["Oil_Rate"].notna().mean() < TEST_MODE_THRESHOLD)


# ---- 3a. AI virtual flow meter ------------------------------------------------
def _last_before(values: pd.Series, wells: pd.Series) -> pd.Series:
    """For every row: the most recent non-missing value STRICTLY before it, within the same well (vectorised)."""
    return values.groupby(wells).shift(1).groupby(wells).ffill()


def _next_after(values: pd.Series, wells: pd.Series) -> pd.Series:
    return values.groupby(wells).shift(-1).groupby(wells).bfill()


def _add_reference(d: pd.DataFrame, interval: int, test_mode: bool) -> pd.DataFrame:
    """Attach the previous well test to every row — only REAL measured rates are used, never interpolated ones.

    • Test mode (oil measured on some days only): every measured row is a well test.
    • Daily mode (oil measured every day): a well test is simulated every `interval` measurements, so the model
      can be scored on exactly the job it does in the field — estimating the days between tests.
    The reference is always the last test strictly BEFORE the row, so a test never 'sees' its own value."""
    d = d.copy()
    measured = d["Oil_Rate"].notna()
    if test_mode:
        is_test = measured
    else:   # simulated tests are taken the way real ones are: on days the well is flowing
        eligible = measured & (d["Oil_Rate"] > 0)
        k = d.loc[eligible].groupby("Well").cumcount()
        is_test = pd.Series(False, index=d.index)
        is_test.loc[k.index] = (k % interval == 0).to_numpy()
    d["Is_Test"] = is_test
    w = d["Well"]
    for src, ref in [("Oil_Rate", "Ref_Oil"), ("Water_Cut", "Ref_WC"), ("Liquid_Rate", "Ref_Liquid"),
                     ("Bottomhole_Pressure", "Ref_BHP"), ("Wellhead_Pressure", "Ref_WHP"), ("Days", "Ref_Day")]:
        d[ref] = _last_before(d[src].where(is_test), w)
    d["Next_Oil"] = _next_after(d["Oil_Rate"].where(is_test), w)
    d["Next_Day"] = _next_after(d["Days"].where(is_test), w)
    # trend between the last two tests
    tw = w[is_test]
    prev_oil = d.loc[is_test, "Oil_Rate"].groupby(tw).shift(1).reindex(d.index)      # previous test, on test rows
    prev_day = d.loc[is_test, "Days"].groupby(tw).shift(1).reindex(d.index)
    ref2_oil, ref2_day = _last_before(prev_oil.where(is_test), w), _last_before(prev_day.where(is_test), w)
    d["Test_Trend"] = (d["Ref_Oil"] - ref2_oil) / (d["Ref_Day"] - ref2_day).clip(lower=1) / d["Ref_Oil"].clip(lower=1)
    d["Days_Since_Test"] = d["Days"] - d["Ref_Day"]
    d["dBHP_Since_Test"] = d["Bottomhole_Pressure"] - d["Ref_BHP"]
    d["dWHP_Since_Test"] = d["Wellhead_Pressure"] - d["Ref_WHP"]
    # how far every other continuous signal has moved since the last test (choke, ΔP, temperatures …):
    # oil changes when the conditions change, so these 'since-test' moves are the strongest real-field clues
    for c in ["Choke_Size", "Injection_Rate"] + [c for c in d.columns if str(c).startswith(SIG)]:
        if c in d.columns and d[c].notna().any():
            ref = _last_before(d[c].where(is_test), w)
            d["Δ " + c] = d[c] - ref
            d["÷ " + c] = d[c] / ref.where(ref.abs() > 1e-6)
    d["Well_Code"] = d["Well"].astype("category").cat.codes
    return d


def vfm_features(d: pd.DataFrame) -> list:
    f = ["Wellhead_Pressure", "Bottomhole_Pressure", "Delta_P", "Injection_Rate", "Days", "Well_Code",
         "Ref_Oil", "Ref_WC", "Ref_Liquid", "Days_Since_Test", "dBHP_Since_Test", "dWHP_Since_Test", "Test_Trend"]
    if "Choke_Size" in d.columns:
        f.insert(3, "Choke_Size")
    return f + [c for c in ("Motor_Current", "Pump_Frequency", "Motor_Temperature") if c in d.columns] \
        + [c for c in d.columns if str(c).startswith((SIG, "Δ ", "÷ "))]


_MISSING = -999.0     # trees treat this as "signal not recorded"; nothing is invented for the gap


class VFMEnsemble:
    """Gradient boosting + extra trees; both learn oil as a multiple of the previous well-test rate.
    Sized for speed: same blind-test accuracy as a 4× larger ensemble at ~1/6 of the training time."""

    def __init__(self, features, n_wells):
        self.features = features
        cat = [features.index("Well_Code")] if n_wells <= 250 else None
        self.gb = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.16, max_leaf_nodes=31, loss="absolute_error",
                                                l2_regularization=1.0, categorical_features=cat, random_state=42)
        self.et = ExtraTreesRegressor(n_estimators=30, min_samples_leaf=3, max_features=0.5, n_jobs=1, random_state=42)

    def _X(self, d):
        return d[self.features].astype(float).fillna(_MISSING).to_numpy()

    def fit(self, d: pd.DataFrame):
        y = (d["Oil_Rate"] / d["Ref_Oil"].clip(lower=1.0)).to_numpy()
        self.seen = d["Well"].value_counts()
        self.lo, self.hi = np.nanquantile(y, [0.005, 0.995]) if len(y) else (0.0, 10.0)
        self.gb.fit(self._X(d), y)
        sub = d.sample(min(len(d), 10000), random_state=42)   # tuned: same blind accuracy, ~2× faster
        self.et.fit(self._X(sub), (sub["Oil_Rate"] / sub["Ref_Oil"].clip(lower=1.0)).to_numpy())
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        X = self._X(d)
        ratio = np.clip(0.5 * self.gb.predict(X) + 0.5 * self.et.predict(X), max(self.lo, 0.0), self.hi)
        # a well the model has (almost) never seen: no learned behaviour yet → hold its last test
        new = d["Well"].map(self.seen).fillna(0).to_numpy() < 5
        ratio = np.where(new, 1.0, ratio)
        return ratio * d["Ref_Oil"].clip(lower=1.0).to_numpy()


RIDGE_SIGNALS = ["Bottomhole_Pressure", "Wellhead_Pressure", "Injection_Rate", "Choke_Size",
                 "Motor_Current", "Pump_Frequency"]


class WellRegression:
    """Per-well ridge regression of oil on that well's own continuous signals (pressures, lift gas, choke, time).
    Fitted on real well tests only — a physics-like 'productivity' model that can estimate any day the sensors ran."""

    def __init__(self, lam: float = 1.0, min_tests: int = 8):
        self.lam, self.min_tests, self.models = lam, min_tests, {}

    def fit(self, rows: pd.DataFrame):
        for w, g in rows.groupby("Well"):
            if len(g) < self.min_tests:
                continue
            cols = [c for c in RIDGE_SIGNALS if c in g.columns and g[c].notna().mean() >= 0.5 and g[c].std() > 1e-9]
            if not cols:
                continue
            A = g[cols].astype(float)
            mu, sd = A.mean(), A.std().replace(0, 1.0)
            Z = np.column_stack([np.ones(len(A)), ((A - mu) / sd).fillna(0.0).to_numpy()])
            pen = np.eye(Z.shape[1]) * self.lam
            pen[0, 0] = 0.0
            beta = np.linalg.solve(Z.T @ Z + pen, Z.T @ g["Oil_Rate"].to_numpy(float))
            self.models[w] = (cols, mu, sd, beta, 1.5 * float(g["Oil_Rate"].max()))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        out = np.full(len(rows), np.nan)
        for w, idx in rows.groupby("Well").indices.items():
            if w not in self.models:
                continue
            cols, mu, sd, beta, cap = self.models[w]
            A = rows.iloc[idx][cols].astype(float)
            Z = np.column_stack([np.ones(len(A)), ((A - mu) / sd).fillna(0.0).to_numpy()])   # missing signal → neutral
            out[idx] = np.clip(Z @ beta, 0.0, cap)                 # never beyond what the well has ever done
        return out


def _mix(P: np.ndarray, w) -> np.ndarray:
    """Weighted mix of the estimate columns in P; a missing estimate gives its weight to the others."""
    w = np.asarray(w, float)
    W = np.where(np.isnan(P), 0.0, w)
    tot = W.sum(1)
    out = (np.nan_to_num(P) * W).sum(1) / np.where(tot > 0, tot, 1.0)
    return np.where(tot > 0, out, P[:, 0])


_WEIGHT_GRID = [(a / 10, b / 10, (10 - a - b) / 10) for a in range(11) for b in range(11 - a)]


def _best_weights(P: np.ndarray, y: np.ndarray):
    """Weights (AI, regression, last test) with the lowest absolute error on an OLDER hold-out window."""
    errs = [np.abs(_mix(P, w) - y).sum() for w in _WEIGHT_GRID]
    return _WEIGHT_GRID[int(np.argmin(errs))]


def _scores(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(y - p)
    return {"R²": r2_score(y, p) if len(y) > 1 else np.nan, "MAE (m³/d)": mean_absolute_error(y, p),
            "Accuracy (%)": 100 * (1 - err.sum() / max(np.abs(y).sum(), 1e-9)),
            "Within ±10 % (%)": 100 * np.mean(err <= 0.10 * np.maximum(np.abs(y), 1e-9))}


@st.cache_resource(show_spinner=False, max_entries=3)
def train_vfm(key: str, _df: pd.DataFrame, interval: int):
    """Learns oil rate ONLY from rows where oil was really measured; scores on later, unseen measurements;
    then estimates oil for every day — including days with no test — from the continuous signals."""
    test_mode = is_test_mode(_df)
    src = _df
    # Daily VOLUMES on partly-shut-in days (e.g. 6 h on stream) are not the well's rate. With an hours-on column the
    # model learns the FLOWING rate (volume × 24 / hours) and converts back to the day's volume at the end.
    hours_basis = (not test_mode and "Hours_On" in src.columns and src["Hours_On"].notna().mean() > 0.5
                   and (src["Hours_On"] < 23.5).mean() > 0.02)
    if hours_basis:
        h = src["Hours_On"].clip(0, 24)
        k = (24.0 / h).where(h >= 6)
        src = src.assign(Oil_Daily=src["Oil_Rate"], Oil_Rate=src["Oil_Rate"] * k, Water_Rate=src["Water_Rate"] * k, Gas_Rate=src["Gas_Rate"] * k)
    d = _add_reference(enrich(src), interval, test_mode)
    feats = [c for c in vfm_features(d) if d[c].notna().mean() > 0.3]
    known = d[d["Oil_Rate"].notna() & d["Ref_Oil"].notna()]          # real targets only
    split = known["Date"].quantile(0.8)
    train, test = known[known["Date"] <= split], known[known["Date"] > split]
    if len(train) < 20 or len(test) < 5:
        raise ValueError("Not enough measured oil rates for a chronological train/test split.")
    n_wells = d["Well"].nunique()

    def members(fit_rows, rows):
        mdl, reg = VFMEnsemble(feats, n_wells).fit(fit_rows), WellRegression().fit(fit_rows)
        return mdl, reg, np.column_stack([mdl.predict(rows), reg.predict(rows), rows["Ref_Oil"].to_numpy(float)])

    # Blend weights are chosen on an OLDER window (64–80 % of the history), so the score on the last 20 % is a
    # genuine blind test. Holding the last test is a candidate too: on erratic real wells it can earn weight.
    inner = known["Date"].quantile(0.64)
    tr_i, va_i = train[train["Date"] <= inner], train[train["Date"] > inner]
    if len(tr_i) >= 20 and len(va_i) >= 5:
        weights = _best_weights(members(tr_i, va_i)[2], va_i["Oil_Rate"].to_numpy(float))
    else:
        weights = (0.5, 0.5, 0.0)
    w_ml = weights[0]
    val_model, val_reg, P = members(train, test)
    p_ml, p_reg = P[:, 0], P[:, 1]
    both = ~np.isnan(p_reg)
    test = test.assign(Pred=_mix(P, weights), Pred_ML=p_ml, Pred_Reg=p_reg)
    span = (test["Next_Day"] - test["Ref_Day"]).replace(0, np.nan)
    interp = test["Ref_Oil"] + (test["Next_Oil"] - test["Ref_Oil"]) * (test["Days"] - test["Ref_Day"]) / span
    has_i = interp.notna()
    rows = [
        {"Dataset": "Training (seen)", **_scores(train["Oil_Rate"], _mix(np.column_stack(
            [val_model.predict(train), val_reg.predict(train), train["Ref_Oil"].to_numpy(float)]), weights)),
         "Rows": len(train)},
        {"Dataset": "Validation (unseen later measurements)", **_scores(test["Oil_Rate"], test["Pred"]), "Rows": len(test)},
        {"Dataset": "Baseline · carry last well test" + (f" (blend weight {weights[2]:.0%})" if weights[2] else ""), **_scores(test["Oil_Rate"], test["Ref_Oil"]), "Rows": len(test)},
        {"Dataset": f"  ↳ boosted trees alone (weight {w_ml:.0%})", **_scores(test["Oil_Rate"], test["Pred_ML"]), "Rows": len(test)},
    ]
    if both.sum() >= 5:
        rows.append({"Dataset": f"  ↳ per-well regression alone (weight {weights[1]:.0%})",
                     **_scores(test.loc[both, "Oil_Rate"], test.loc[both, "Pred_Reg"]), "Rows": int(both.sum())})
    if has_i.sum() >= 5:
        rows.append({"Dataset": "Baseline · interpolate between tests (needs the NEXT test)",
                     **_scores(test.loc[has_i, "Oil_Rate"], interp[has_i]), "Rows": int(has_i.sum())})
    metrics = pd.DataFrame(rows)
    importance = pd.DataFrame({"Feature": feats, "Importance": val_model.et.feature_importances_})

    est = d[d["Ref_Oil"].notna()].copy()                              # every day after the first test
    final, final_reg, P_all = members(known, est)
    raw_pred = _mix(P_all, weights)
    up = np.ones(len(est))
    if test_mode and "Hours_On" in est.columns:
        # tests measure the FLOWING rate; on untested days the daily average = flowing rate × hours on / 24
        up = (est["Hours_On"].clip(0, 24) / 24).where(est["Oil_Rate"].isna()).fillna(1.0).to_numpy()
    elif hours_basis:      # back from flowing rate to the day's volume, on every day
        up = (est["Hours_On"].clip(0, 24) / 24).fillna(1.0).to_numpy()
        est["Oil_Rate"] = est["Oil_Daily"]
    est = est.assign(Pred=raw_pred * up, Split=np.where(est["Date"] <= split, "Train", "Validation"), Pred_Unseen=np.nan)
    tix = est.index.intersection(test.index)
    est.loc[tix, "Pred_Unseen"] = test.loc[tix, "Pred"] * pd.Series(up, index=est.index).loc[tix]
    m = est[est["Oil_Rate"].notna()]
    m = m.assign(_ae=(m["Oil_Rate"] - m["Pred"]).abs(), _aeu=(m["Oil_Rate"] - m["Pred_Unseen"]).abs(),
                 _yu=m["Oil_Rate"].where(m["Pred_Unseen"].notna()))
    agg = m.groupby("Well", sort=True)[["_ae", "Oil_Rate", "_aeu", "_yu"]].sum(min_count=1)
    per_well = pd.DataFrame({"Well": agg.index,
                             "History accuracy (%)": 100 * (1 - agg["_ae"] / agg["Oil_Rate"].clip(lower=1e-9)).to_numpy(),
                             "Unseen accuracy (%)": 100 * (1 - agg["_aeu"] / agg["_yu"].clip(lower=1e-9)).to_numpy()})
    return {"metrics": metrics, "history": _scores(m["Oil_Rate"], m["Pred"]),
            "pred": est[["Date", "Well", "Oil_Rate", "Pred", "Pred_Unseen", "Split", "Is_Test"]].reset_index(drop=True),
            "importance": importance.sort_values("Importance"), "per_well": per_well, "split": split,
            "test_mode": test_mode, "n_tests": int(d["Is_Test"].sum()), "w_ml": w_ml, "weights": weights,
            "untested_days": int(est["Oil_Rate"].isna().sum())}


# ---- 3b. Decline curve analysis (Arps) --------------------------------------------
def _arps(t, qi, di, b):
    return qi / np.power(1 + b * di * t, 1 / b)


@st.cache_resource(show_spinner=False, max_entries=3)
def fit_declines(key: str, _df: pd.DataFrame, horizon: int = 180):
    df = _df
    params, curves = [], []
    notes = {}
    tests = is_test_mode(df)
    win, min_pts = (3, 6) if tests else (7, 15)
    for well, g in df.sort_values("Date").groupby("Well", sort=True):
        meas = g.dropna(subset=["Oil_Rate"])                 # fit ONLY real measured rates, at their real dates
        meas = meas[meas["Oil_Rate"] > 0]                    # shut-in days are downtime, not decline
        if "Hours_On" in meas.columns and (meas["Hours_On"] >= 20).sum() >= min_pts:
            meas = meas[~(meas["Hours_On"] < 20)]            # partial-hour days understate the well's capacity
        if len(meas) < min_pts:
            continue
        if "Choke_Size" in g.columns:   # a bean change starts a new decline segment
            change = np.flatnonzero(g["Choke_Size"].diff().fillna(0).to_numpy() != 0)
            if len(change):
                since = g["Date"].iloc[change[-1]]
                if (meas["Date"] >= since).sum() >= max(min_pts, 2 * win):
                    meas = meas[meas["Date"] >= since]
                    notes[well] = f"fitted since choke change on {since:%d %b %Y}"
        s = meas.set_index("Date")["Oil_Rate"].rolling(win, min_periods=1).mean()
        t = (s.index - s.index[0]).total_seconds().to_numpy() / 86400
        y = s.to_numpy()
        if len(y) < min_pts or y.max() <= 0:
            continue
        try:
            (qi, di, b), _ = curve_fit(_arps, t, y, p0=[y[:7].mean(), 1e-3, 0.5],
                                       bounds=([1e-6, 1e-7, 0.01], [y.max() * 3, 0.2, 1.0]), maxfev=20000)
        except (RuntimeError, ValueError):
            slope = np.polyfit(t, np.log(np.maximum(y, 1e-3)), 1)[0]
            qi, di, b = y[0], max(-slope, 1e-7), 0.01
        fit = _arps(t, qi, di, b)
        r2 = 1 - np.sum((y - fit) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        d_now = di / (1 + b * di * t[-1])
        tf = t[-1] + np.arange(1, horizon + 1)
        qf = _arps(tf, qi, di, b)
        params.append({"Well": well, "Current rate (m³/d)": float(_arps(t[-1], qi, di, b)), "b-factor": b,
                       "Decline (%/yr)": 100 * (1 - np.power(1 + b * d_now * 365, -1 / b)),
                       "Next 180 d (m³)": float(qf.sum()), "Fit R²": r2, "Basis": notes.get(well, "full history")})
        curves.append(pd.DataFrame({"Well": well, "Date": s.index, "Fitted": fit}))
        curves.append(pd.DataFrame({"Well": well, "Date": s.index[-1] + pd.to_timedelta(np.arange(1, horizon + 1), "D"),
                                    "Forecast": qf}))
    return pd.DataFrame(params), (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


# ---- 3c. Gas-lift performance curves ------------------------------------------------
@dataclass
class LiftCurve:
    """oil = b0 + b1·x + b2·x² + b3·days + b4·choke   (x = lift gas / scale), evaluated today."""
    well: str
    beta: np.ndarray
    scale: float
    days_now: float
    choke_now: float
    x_now: float
    x_min: float
    x_max: float
    r2: float
    shape: str

    def oil(self, inj):
        x = np.asarray(inj, float) / self.scale
        return np.maximum(self.beta[0] + self.beta[1] * x + self.beta[2] * x ** 2 + self.beta[3] * self.days_now
                          + self.beta[4] * self.choke_now, 0.0)

    def marginal(self, inj):
        """Extra oil (m³/d) per 1,000 units of lift gas at this setting."""
        x = np.asarray(inj, float) / self.scale
        return (self.beta[1] + 2 * self.beta[2] * x) / self.scale * 1000


@st.cache_resource(show_spinner=False, max_entries=3)
def fit_lift_curves(key: str, _df: pd.DataFrame, window: int = 150):
    df = _df
    d = enrich(df)
    pos = d.loc[d["Injection_Rate"] > 0, "Injection_Rate"]
    scale = 1000.0 if (len(pos) and pos.median() > 2000) else 1.0
    curves, skipped = {}, {}
    min_pts = 12 if is_test_mode(df) else 20
    for well, full in d.groupby("Well", sort=True):
        g = full.dropna(subset=["Oil_Rate", "Injection_Rate"]).tail(window)   # real measured oil only
        x = g["Injection_Rate"].to_numpy() / scale
        if len(g) < min_pts or np.std(x) < 1e-6 or x.max() <= 0:
            skipped[well] = "not enough measured oil with varying lift gas"
            continue
        ch = g["Choke_Size"].to_numpy() if "Choke_Size" in g.columns else np.zeros(len(g))
        y = g["Oil_Rate"].to_numpy()
        X = np.column_stack([np.ones(len(g)), x, x ** 2, g["Days"].to_numpy(), ch])
        keep = [0, 1, 2, 3] + ([4] if np.std(ch) > 1e-6 else [])
        beta = np.zeros(5)
        beta[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
        shape = "Concave · diminishing returns"
        if beta[2] >= 0:   # no curvature in the data → linear response, bounded to observed range
            keep = [k for k in keep if k != 2]
            beta = np.zeros(5)
            beta[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
            shape = "Linear · within observed range"
        r2 = 1 - np.sum((y - X @ beta) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        cur = full.dropna(subset=["Injection_Rate"]).iloc[-1]                  # today's settings
        ch_now = float(cur["Choke_Size"]) if "Choke_Size" in full.columns and pd.notna(cur.get("Choke_Size")) else float(ch[-1])
        curves[well] = LiftCurve(well, beta, scale, float(full["Days"].iloc[-1]), ch_now,
                                 float(cur["Injection_Rate"]), float(g["Injection_Rate"].min()),
                                 float(g["Injection_Rate"].max()), float(r2), shape)
    return curves, skipped


# ---- 3d. Equipment health ------------------------------------------------------------
Z_ALERT, Z_WARN = 4.0, 2.5   # robust σ from the well's own healthy baseline


@st.cache_resource(show_spinner=False, max_entries=3)
def screen_equipment(key: str, _df: pd.DataFrame, sensors: tuple):
    df = _df
    """Per-well robust z-scores vs own baseline → Isolation Forest → alert + early warning (trend to threshold)."""
    sensors = list(sensors)
    if len(sensors) < 2:
        return None, None
    running = pd.Series(True, index=df.index)            # a stopped pump (0 Hz, 0 A, shut-in) is not an anomaly
    for c in ("Pump_Frequency", "Motor_Current"):
        if c in df.columns:
            running &= ~(df[c] <= 0.5)
    if "Hours_On" in df.columns:
        running &= ~(df["Hours_On"] < 12)
    d = df.loc[running, ["Date", "Well"] + sensors].dropna().sort_values(["Well", "Date"]).reset_index(drop=True)
    if len(d) < 40:
        return None, None
    # Rolling baseline: each reading is compared with the same pump's previous ~60 running days, skipping the last 14
    # (so a developing fault is not absorbed into "normal"). Slow, natural drift — e.g. lower motor load as the
    # reservoir declines — is followed; sudden or accelerating changes stand out. A stop of > 5 days (e.g. a pump
    # change) starts a new baseline, so a new pump is not judged against the old one.
    W, SKIP = 60, 14
    seg = d.groupby("Well")["Date"].diff().dt.days.gt(5).groupby(d["Well"]).cumsum()
    keys = [d["Well"], seg]
    z = np.zeros((len(d), len(sensors)))
    gid = d["Well"].astype(str) + "#" + seg.astype(str)
    first20 = d.groupby(gid).cumcount() < 20
    for j, c in enumerate(sensors):
        roll = d[c].groupby(gid).rolling(W, min_periods=15)
        def _lag(r):
            return r.reset_index(level=0, drop=True).sort_index().groupby(gid).shift(SKIP)
        med, q1, q3 = _lag(roll.median()), _lag(roll.quantile(0.25)), _lag(roll.quantile(0.75))
        head = d[c].where(first20)                                       # start of a segment: its first readings
        first_med = head.groupby(gid).transform("median")
        first_iqr = head.groupby(gid).transform(lambda x: x.quantile(0.75) - x.quantile(0.25))
        med, iqr = med.fillna(first_med), (q3 - q1).fillna(first_iqr)
        spread = np.maximum(iqr / 1.349, 0.02 * med.abs() + 1e-6)
        z[:, j] = ((d[c] - med) / spread).fillna(0.0).to_numpy()
    pos = d.groupby("Well").cumcount() / d.groupby("Well")["Date"].transform("size")
    base = (pos < 0.6).to_numpy()
    iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=1).fit(z[base])
    d["Score"] = -iso.score_samples(z)
    thr = float(np.quantile(d.loc[base, "Score"], 0.995))
    calm = float(np.median(d.loc[base, "Score"]))
    d["Alert"] = d["Score"] > thr
    zcols = [f"z_{s}" for s in sensors]
    d[zcols] = z
    rows = []
    for well, g in d.groupby("Well", sort=True):
        now = float(g["Score"].tail(3).mean())
        tail = g.tail(10)
        slope = np.polyfit(np.arange(len(tail)), tail["Score"].to_numpy(), 1)[0] if len(tail) > 4 else 0.0
        days_to = (thr - now) / slope if (slope > 1e-6 and now < thr) else np.nan
        zl = g[zcols].tail(3).mean()
        drv = zl.abs().idxmax()
        z_now = float(abs(zl[drv]))
        z_slope = np.polyfit(np.arange(len(tail)), tail[drv].abs().to_numpy(), 1)[0] if len(tail) > 4 else 0.0
        rising = (np.isfinite(days_to) and days_to <= 14 and now >= calm + 0.7 * (thr - calm)) or \
                 (z_now >= Z_WARN and z_slope > 0)
        status = "ALERT" if (now > thr or z_now >= Z_ALERT) else ("EARLY WARNING" if rising else "NORMAL")
        if status == "EARLY WARNING" and not np.isfinite(days_to) and z_slope > 0:
            days_to = (Z_ALERT - z_now) / z_slope
        rows.append({"Well": well, "Status": status,
                     "Health": float(np.clip(min(100 * (1 - (now - calm) / max(thr - calm, 1e-9)), 100 * (1 - z_now / Z_ALERT)), 0, 100)),
                     "Days to alert": days_to, "Main signal": f"{drv[2:].replace('_', ' ')} {zl[drv]:+.1f}σ",
                     "Score": now})
    return d, pd.DataFrame(rows).assign(Threshold=thr)


# ---- 3e. Well ranking --------------------------------------------------------------------
@dataclass
class Settings:
    wc_watch: float = 70.0
    wc_high: float = 90.0
    decline_watch: float = 15.0
    decline_high: float = 30.0
    test_interval: int = 14
    co2: float = GRID_CO2
    oil_price: float = 70.0


def rank_wells(df, dca, eq, s: Settings) -> pd.DataFrame:
    d = enrich(df)
    latest = d.groupby("Well").tail(1).drop(columns=["Oil_Rate", "Water_Rate", "Liquid_Rate", "Water_Cut"])
    cols = ["Oil_Rate", "Water_Rate", "Liquid_Rate"] + (["Energy_kWh"] if "Energy_kWh" in d.columns else [])
    n_recent = 3 if is_test_mode(df) else 7          # average of the latest REAL measurements, never filled values
    # Rates and water cut describe the well WHILE IT PRODUCES: shut-in zeros would hide a 97 % water cut.
    flowing = d[(d["Oil_Rate"].fillna(0) + d["Water_Rate"].fillna(0)) > 0]
    recent = pd.DataFrame({c: d.dropna(subset=[c]).groupby("Well").tail(n_recent).groupby("Well")[c].mean()
                           for c in cols}).rename_axis("Well").reset_index()          # current rates (0 if shut in)
    fl = pd.DataFrame({c: flowing.dropna(subset=[c]).groupby("Well").tail(n_recent).groupby("Well")[c].mean()
                       for c in ["Oil_Rate", "Water_Rate", "Liquid_Rate"]})
    recent["Oil while flowing"] = recent["Well"].map(fl["Oil_Rate"])
    recent["Liquid while flowing"] = recent["Well"].map(fl["Liquid_Rate"])
    recent["WC while flowing"] = recent["Well"].map(100 * fl["Water_Rate"] / fl["Liquid_Rate"].replace(0, np.nan))
    # A well whose latest measurements are all zero is shut in — note since when
    meas = d.dropna(subset=["Oil_Rate"])
    last_n = meas.groupby("Well").tail(n_recent)
    shut = last_n.groupby("Well")["Oil_Rate"].max().fillna(0) <= 0
    last_flow = flowing.groupby("Well")["Date"].max()
    since = {}
    for w in shut.index[shut.to_numpy()]:
        after = meas.loc[meas["Well"] == w, "Date"]
        if w in last_flow.index:
            after = after[after > last_flow[w]]
        if len(after):
            since[w] = after.min()
    recent["Shut in since"] = pd.to_datetime(recent["Well"].map(since))
    last_test = d.dropna(subset=["Oil_Rate"]).groupby("Well")["Date"].max().rename("Last test")
    recent = recent.merge(last_test.reset_index(), on="Well", how="left")
    latest = latest.drop(columns=[c for c in cols if c in latest.columns]).merge(recent, on="Well")
    latest["Water_Cut"] = latest["WC while flowing"]      # a shut-in well keeps the water cut it last produced at
    if "Energy_kWh" in latest.columns:
        latest["kWh_per_bbl"] = latest["Energy_kWh"] / (latest["Oil_Rate"] * M3_TO_BBL).replace(0, np.nan)
    latest = latest.merge(dca[["Well", "Decline (%/yr)"]], on="Well", how="left") if not dca.empty \
        else latest.assign(**{"Decline (%/yr)": np.nan})
    if eq is not None:
        latest = latest.merge(eq[["Well", "Status", "Main signal", "Days to alert", "Health"]], on="Well", how="left")
    else:
        latest = latest.assign(Status="NO DATA", **{"Main signal": "", "Days to alert": np.nan, "Health": np.nan})

    def score(r):
        pts, why = 0, []
        if r["Water_Cut"] >= s.wc_high:
            pts += 2; why.append(f"water cut {r['Water_Cut']:.0f}%")
        elif r["Water_Cut"] >= s.wc_watch:
            pts += 1; why.append(f"water cut {r['Water_Cut']:.0f}%")
        dec = r["Decline (%/yr)"]
        if pd.notna(dec) and dec >= s.decline_high:
            pts += 2; why.append(f"steep decline {dec:.0f}%/yr")
        elif pd.notna(dec) and dec >= s.decline_watch:
            pts += 1; why.append(f"decline {dec:.0f}%/yr")
        if pd.notna(r.get("Shut in since")):
            pts += 1; why.append(f"shut in since {r['Shut in since']:%d %b %Y}")
        if r["Status"] == "ALERT":
            pts += 3; why.append(f"equipment alert ({r['Main signal']})")
        elif r["Status"] == "EARLY WARNING":
            pts += 2; why.append(f"early warning, ~{r['Days to alert']:.0f} d to alert ({r['Main signal']})")
        level = "CRITICAL" if pts >= 4 else "HIGH" if pts >= 3 else "WATCH" if pts >= 1 else "STABLE"
        return pd.Series({"Score": pts, "Priority": level, "Why": ", ".join(why) or "within limits"})

    latest = pd.concat([latest.reset_index(drop=True), latest.apply(score, axis=1).reset_index(drop=True)], axis=1)
    return latest.sort_values(["Score", "Oil_Rate"], ascending=[False, False]).reset_index(drop=True)


@st.cache_resource(show_spinner=False, max_entries=6)
def rank_cached(key: str, _df, _dca, _eq, _s: Settings, settings_key: tuple):
    return rank_wells(_df, _dca, _eq, _s)


# ---- 3g. FORECASTING — predict the next N periods of any measurement, per well -------------------
FREQ_RULES = [(0.9, "h", "hour"), (1.5, "D", "day"), (10, "W", "week"), (45, "M", "month"), (120, "Q", "quarter"),
              (1e9, "Y", "year")]
DEFAULT_STEPS = {"h": 48, "D": 30, "W": 12, "M": 6, "Q": 4, "Y": 3}
HELD_CONSTANT = {"Choke_Size", "Injection_Rate"}       # operator set-points: assumed unchanged unless the user plans otherwise
NON_NEGATIVE = set(RATE_COLUMNS + PRESSURE_COLUMNS + ["Liquid_Rate", "Choke_Size"] + SENSOR_COLUMNS)
FC_LABELS = {"Liquid_Rate": "Gross liquid rate", "Water_Cut": "Water cut"}
FC_UNITS = {**STD_UNIT, "Water_Cut": "%", "Liquid_Rate": "m³/d"}
SUMMABLE = set(RATE_COLUMNS + ["Liquid_Rate", "Energy_kWh"])   # field total = sum over wells (others: average)


def detect_frequency(df: pd.DataFrame) -> tuple:
    """Spacing of the data (hour / day / week / month / quarter / year) from the median gap within wells."""
    gaps = df.sort_values(["Well", "Date"]).groupby("Well")["Date"].diff().dt.total_seconds() / 86400
    gaps = gaps[gaps > 0]
    med = float(gaps.median()) if len(gaps) else 1.0
    for lim, code, label in FREQ_RULES:
        if med <= lim:
            return code, label
    return "D", "day"


def _to_period(dates: pd.Series, code: str) -> pd.Series:
    try:
        return dates.dt.to_period(code)
    except (ValueError, TypeError):           # older pandas spells hourly 'H' and yearly 'A'
        return dates.dt.to_period({"h": "H", "Y": "A"}.get(code, code))


def _period_dates(periods: pd.PeriodIndex, last_date: pd.Timestamp, code: str) -> pd.DatetimeIndex:
    """Period labels → dates in the same convention as the user's data (e.g. month-end, or Sunday weeks)."""
    if code in ("M", "Q", "Y"):
        return periods.to_timestamp(how="end").normalize() if last_date.is_month_end else periods.to_timestamp()
    start = periods.to_timestamp()
    return start + (last_date - start[-1])


def _future_dates(last_date: pd.Timestamp, code: str, H: int) -> list:
    if code in ("M", "Q", "Y"):
        months = {"M": 1, "Q": 3, "Y": 12}[code]
        if last_date.is_month_end:
            return [last_date + pd.offsets.MonthEnd(months * k) for k in range(1, H + 1)]
        return [last_date + pd.DateOffset(months=months * k) for k in range(1, H + 1)]
    step = {"h": pd.Timedelta(hours=1), "D": pd.Timedelta(days=1), "W": pd.Timedelta(weeks=1)}[code]
    return [last_date + step * k for k in range(1, H + 1)]


FREQ_DAYS = {"h": 1 / 24, "D": 1.0, "W": 7.0, "M": 30.44, "Q": 91.3, "Y": 365.25}
CODE_LABEL = {"h": "hour", "D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}


def variable_cadence(d: pd.DataFrame, col: str, main_code: str) -> str:
    """How often THIS measurement is really recorded (a well test every ~12 days → monthly), never finer than the data."""
    x = d.dropna(subset=[col])
    if x.empty:
        return main_code
    code, _ = detect_frequency(x)
    return code if FREQ_DAYS[code] > FREQ_DAYS[main_code] else main_code


def regular_series(d: pd.DataFrame, col: str, code: str) -> dict:
    """Measurements placed on a common calendar grid at `code` spacing. Several readings in one period are averaged;
    periods with no reading stay EMPTY (NaN) — nothing is interpolated or filled."""
    x = d[["Well", "Date", col]].dropna()
    if x.empty:
        return {}
    x = x.assign(P=_to_period(x["Date"], code))
    last_date = d["Date"].max()                                     # the field's "today"
    end = _to_period(pd.Series([last_date]), code).iloc[0]
    grid = pd.period_range(x["P"].min(), end, freq=end.freq)
    wells = {}
    for w, g in x.groupby("Well", sort=True):
        v = g.groupby("P")[col].mean().reindex(grid).to_numpy(dtype=float)
        if np.isfinite(v).sum() >= 3:
            wells[str(w)] = v
    return {"grid": grid, "wells": wells, "last_date": last_date, "code": code,
            "density": float(np.mean([np.isfinite(v[np.argmax(np.isfinite(v)):]).mean() for v in wells.values()]))
            if wells else 0.0}


# -- candidate methods: each takes a history that may contain gaps (NaN) and returns the next H grid values --
def _obs(y):
    i = np.flatnonzero(np.isfinite(y))
    return i, y[i]


def _fc_level(y, H):
    _, v = _obs(y)
    return np.full(H, float(np.mean(v[-3:])))


def _fc_trend(y, H):
    i, v = _obs(y)
    W = max(10, 3 * H)
    keep = i >= len(y) - W
    if keep.sum() < 3:
        keep = np.arange(len(i)) >= len(i) - 6
    if keep.sum() < 2:
        return None
    slope, icpt = np.polyfit(i[keep], v[keep], 1)
    return icpt + slope * (len(y) + np.arange(H))


def _fc_expdecline(y, H):
    i, v = _obs(y)
    keep = i >= len(y) - max(30, 4 * H)
    if keep.sum() < 4 or (v[keep] <= 0).any():
        return None
    slope, icpt = np.polyfit(i[keep], np.log(v[keep]), 1)
    if slope > 0:                                   # not declining → leave it to the trend methods
        return None
    return np.exp(icpt + slope * (len(y) + np.arange(H)))


_HA, _HB, _HP = (g.ravel() for g in np.meshgrid([0.1, 0.25, 0.45, 0.7, 0.9], [0.02, 0.08, 0.2], [0.8, 0.9, 0.98]))


def _fc_holt(y, H):
    """Damped-trend exponential smoothing; 45 parameter sets fitted at once, the best one-step fit wins.
    A missing period is NOT filled: the model simply carries its level/trend forward and learns nothing from it."""
    y = y[-400:]
    i, v = _obs(y)
    if len(v) < 6:
        return None
    y = y[i[0]:]
    lvl = np.full(_HA.size, v[0])
    tr = np.full(_HA.size, float(np.mean(np.diff(v[:6]) / np.maximum(np.diff(i[:6]), 1))))
    sse = np.zeros(_HA.size)
    for val in y[1:]:
        f = lvl + _HP * tr
        if np.isnan(val):
            lvl, tr = f, _HP * tr
            continue
        sse += (val - f) ** 2
        new = _HA * val + (1 - _HA) * f
        tr = _HB * (new - lvl) + (1 - _HB) * _HP * tr
        lvl = new
    k = int(np.argmin(sse))
    return lvl[k] + tr[k] * np.cumsum(_HP[k] ** np.arange(1, H + 1))


def _features(y):
    """Shape of the recent history (observed points only), scaled by the recent level."""
    s = pd.Series(y)
    m3, m7, m14, m30 = (s.rolling(k, min_periods=1).mean() for k in (3, 7, 14, 30))
    sd7 = s.rolling(7, min_periods=2).std().fillna(0.0)
    sl7 = (m3 - m3.shift(6)) / 6
    sl30 = (m7 - m7.shift(23)) / 23
    scale = (m30.abs() + 0.05 * s.abs().expanding().mean() + 1e-9).to_numpy()
    F = np.column_stack([s, m3, m7, m14, sl7.fillna(0.0), sl30.fillna(0.0), sd7]) / scale[:, None]
    return F, scale


def _gb_train(histories: list, H: int):
    """Global gradient-boosted model across all wells: predicts y[t+h] / level[t] from the recent shape and h.
    Trained only on pairs where BOTH y[t] and y[t+h] were really measured."""
    feats = [(_features(y), y) for y in histories if np.isfinite(y).sum() >= 15]
    if not feats:
        return None
    Hg = H + 3                                       # a few extra steps: the last reading may be a little old
    cap = max(400, 30000 // Hg)
    rng = np.random.default_rng(0)
    Xs, Ys = [], []
    for h in range(1, Hg + 1):
        xs, ys = [], []
        for (F, sc), y in feats:
            idx = np.arange(5, len(y) - h)
            idx = idx[np.isfinite(y[idx]) & np.isfinite(y[idx + h])]
            if len(idx):
                xs.append(np.column_stack([F[idx], np.full(len(idx), h)]))
                ys.append(y[idx + h] / sc[idx])
        if xs:
            X, Y = np.vstack(xs), np.concatenate(ys)
            if len(Y) > cap:
                keep = rng.choice(len(Y), cap, replace=False)
                X, Y = X[keep], Y[keep]
            Xs.append(X)
            Ys.append(Y)
    if not Xs or sum(len(y) for y in Ys) < 300:
        return None
    model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.12, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0)   # tuned: same accuracy, ~2× faster
    model.hmax = Hg
    return model.fit(np.vstack(Xs), np.concatenate(Ys))


def _fc_gb_batch(model, hist: dict, H: int) -> dict:
    """ML forecasts for every well in ONE predict call, starting from each well's latest real reading."""
    if model is None or not hist:
        return {}
    wells, rows, scales = [], [], []
    for w, y in hist.items():
        i, _ = _obs(y)
        if len(i) < 10:
            continue
        F, sc = _features(y)
        L = i[-1]
        steps = np.clip(len(y) - L + np.arange(H), 1, model.hmax)
        wells.append(w)
        rows.append(np.column_stack([np.tile(F[L], (H, 1)), steps]))
        scales.append(sc[L])
    if not wells:
        return {}
    pred = model.predict(np.vstack(rows)).reshape(len(wells), H) * np.asarray(scales)[:, None]
    return dict(zip(wells, pred))


def _candidates(y, H, held, gb_pred) -> dict:
    if held:
        _, v = _obs(y)
        return {"Held at current setting": np.full(H, float(v[-1]))}
    out = {"Recent level": _fc_level(y, H), "Linear trend": _fc_trend(y, H),
           "Holt (damped trend)": _fc_holt(y, H), "Exponential decline": _fc_expdecline(y, H)}
    if gb_pred is not None:
        out["ML · boosted trees (all wells)"] = gb_pred
    return {k: v for k, v in out.items() if v is not None and np.isfinite(v).all()}


def _blend(errs: dict) -> dict:
    """Inverse-error weights over the two best methods (more stable than picking a single winner)."""
    if not errs:
        return {}
    best = sorted(errs, key=errs.get)[:2]
    inv = {m: 1.0 / (errs[m] + 1e-9) for m in best}
    tot = sum(inv.values())
    return {m: v / tot for m, v in inv.items()}


def _combine(cands: dict, wts: dict) -> np.ndarray:
    wts = {m: w for m, w in wts.items() if m in cands} or {next(iter(cands)): 1.0}
    tot = sum(wts.values())
    return sum(cands[m] * (w / tot) for m, w in wts.items())


@st.cache_resource(show_spinner=False, max_entries=32)
def forecast_series(key: str, var: str, H: int, code: str, _series: dict) -> dict:
    """Rolling-origin blind test + weighted ensemble for one variable across all wells, on REAL readings only.

    For k = 1, 2, 3 the last k·H calendar periods are hidden; every method forecasts them from the readings before
    (the ML model is retrained without the hidden data) and is scored ONLY against periods that were really measured.
    Methods are weighted by those errors; the most recent window is scored with weights chosen on older windows
    only, so the reported accuracy is a true blind test. All methods are then refitted to predict the next H periods."""
    held = var in HELD_CONSTANT
    grid, ys, last_date = _series["grid"], _series["wells"], _series["last_date"]
    G = len(grid)
    nonneg = var in NON_NEGATIVE or all((y[np.isfinite(y)] >= 0).all() for y in ys.values())
    need = max(6, min(20, H + 5))                   # real readings needed before a cut
    # blind-test windows end at the last period with readings for a fair share of wells (the current period may be
    # only a few days old and hold no test yet — scoring against it would be meaningless)
    counts = np.sum([np.isfinite(y) for y in ys.values()], axis=0)
    have = np.flatnonzero(counts >= max(1, 0.3 * len(ys)))
    E = int(have[-1]) + 1 if len(have) else G
    origins = [k for k in (1, 2, 3) if E - k * H > 0 and
               any(np.isfinite(y[:E - k * H]).sum() >= need for y in ys.values())]
    cands, actual = {}, {}
    for k in origins + [0]:
        cut = E - k * H if k else G
        hist = {w: y[:cut] for w, y in ys.items() if np.isfinite(y[:cut]).sum() >= (need if k else 3)}
        gb = None
        if not held and len(hist) and sum(np.isfinite(h).sum() for h in hist.values()) >= 60:
            try:
                gb = _gb_train(list(hist.values()), H)
            except Exception:  # noqa: BLE001 — the other methods still run
                gb = None
        gb_preds = _fc_gb_batch(gb, hist, H)
        for w, hy in hist.items():
            c = _candidates(hy, H, held, gb_preds.get(w))
            if not c:
                continue
            cands[(w, k)] = c
            if k:
                act = ys[w][cut:cut + H]
                if np.isfinite(act).any():
                    actual[(w, k)] = act

    def mae(fc, act):
        m = np.isfinite(act)
        return float(np.mean(np.abs(fc[m] - act[m])))

    rows, blind, per_well, rel_err = [], {}, [], {h: [] for h in range(H)}
    abs_e = abs_y = abs_base = 0.0
    optimistic = False
    future = _future_dates(last_date, code, H) if code not in ("M", "Q", "Y") else \
        list(_period_dates(pd.period_range(grid[-1] + 1, periods=H, freq=grid.freq), last_date, code))
    hist_dates = _period_dates(grid, last_date, code)
    for w, y in ys.items():
        if (w, 0) not in cands:
            continue
        ks = [k for k in origins if (w, k) in actual and (w, k) in cands]
        err = {k: {m: mae(fc, actual[(w, k)]) for m, fc in cands[(w, k)].items()} for k in ks}

        def avg_err(use):
            common = set.intersection(*(set(err[k]) for k in use)) if use else set()
            return {m: float(np.mean([err[k][m] for k in use])) for m in common}

        final_w = _blend(avg_err(ks)) if ks else {"Recent level": 1.0, "Held at current setting": 1.0}
        acc_w = None
        if 1 in ks:
            older = [k for k in ks if k != 1]
            bw = _blend(avg_err(older)) if older else _blend(avg_err([1]))
            optimistic |= not older
            pred = _combine(cands[(w, 1)], bw)
            if nonneg:
                pred = np.maximum(pred, 0)
            act = actual[(w, 1)]
            msk = np.isfinite(act)
            base = cands[(w, 1)].get("Recent level", cands[(w, 1)].get("Held at current setting"))
            abs_e += float(np.abs(pred[msk] - act[msk]).sum())
            abs_y += float(np.abs(act[msk]).sum())
            abs_base += float(np.abs(base[msk] - act[msk]).sum())
            acc_w = 100 * (1 - np.abs(pred[msk] - act[msk]).sum() / max(np.abs(act[msk]).sum(), 1e-9))
            blind[w] = (hist_dates[E - H:E], act, pred)
        for k in ks:                                     # error spread by horizon → P10/P90 (real readings only)
            p = _combine(cands[(w, k)], final_w)
            rel = (actual[(w, k)] - p) / np.maximum(np.abs(p), 1e-6)
            for h in range(H):
                if np.isfinite(rel[h]):
                    rel_err[h].append(rel[h])
        fc = _combine(cands[(w, 0)], final_w)
        if nonneg:
            fc = np.maximum(fc, 0)
        if var == "Water_Cut":
            fc = np.clip(fc, 0, 100)
        method = " + ".join(f"{m} {wt:.0%}" for m, wt in sorted(final_w.items(), key=lambda x: -x[1]) if m in cands[(w, 0)])
        for h, (dt, v) in enumerate(zip(future, fc)):
            rows.append((w, dt, h + 1, float(v), method))
        n_obs = int(np.isfinite(y).sum())
        per_well.append({"Well": w, "Blind-test accuracy (%)": acc_w, "Real readings used": n_obs, "Method": method})

    lo, hi = np.zeros(H), np.zeros(H)
    for h in range(H):
        e = np.asarray(rel_err[h], dtype=float)
        if len(e) >= 8:
            lo[h], hi[h] = np.quantile(e, 0.1), np.quantile(e, 0.9)
        else:
            lo[h], hi[h] = -0.05 * np.sqrt(h + 1), 0.05 * np.sqrt(h + 1)
    lo, hi = np.minimum.accumulate(np.minimum(lo, 0)), np.maximum.accumulate(np.maximum(hi, 0))
    fc = pd.DataFrame(rows, columns=["Well", "Date", "Step", "P50", "Method"])
    if len(fc):
        fc["P10"] = fc["P50"] * (1 + lo[fc["Step"].to_numpy() - 1])
        fc["P90"] = fc["P50"] * (1 + hi[fc["Step"].to_numpy() - 1])
    else:
        fc["P10"], fc["P90"] = [], []
    if held:
        fc["P10"] = fc["P90"] = fc["P50"]
    if nonneg:
        fc[["P10", "P50", "P90"]] = fc[["P10", "P50", "P90"]].clip(lower=0)
    if var == "Water_Cut":
        fc[["P10", "P50", "P90"]] = fc[["P10", "P50", "P90"]].clip(0, 100)
    hist = pd.DataFrame([(w, dt, v) for w, y in ys.items() for dt, v in zip(hist_dates, y) if np.isfinite(v)],
                        columns=["Well", "Date", "Value"])
    tested = abs_y > 0
    return {
        "fc": fc[["Well", "Date", "Step", "P10", "P50", "P90", "Method"]],
        "hist": hist, "blind": blind, "per_well": pd.DataFrame(per_well),
        "accuracy": 100 * (1 - abs_e / abs_y) if tested else np.nan,
        "baseline": 100 * (1 - abs_base / abs_y) if tested else np.nan,
        "tested_wells": len(blind), "optimistic": optimistic, "held": held,
        "code": code, "steps": H, "density": _series["density"],
    }


def forecastable(f: "Field") -> tuple:
    """(enriched frame incl. extra numeric columns from the file, {column: label}) of everything that can be forecast."""
    d = enrich(f.df)
    opts = {}
    for c in ["Oil_Rate", "Water_Rate", "Gas_Rate", "Liquid_Rate", "Water_Cut", "Wellhead_Pressure",
              "Bottomhole_Pressure", "Injection_Rate", "Choke_Size"] + SENSOR_COLUMNS:
        enough = d[c].notna().sum() >= 10 if c in d.columns else False      # sparse well tests count too
        if c in d.columns and enough and d[c].nunique() > 1:
            opts[c] = FC_LABELS.get(c) or FIELD_LABELS.get(c, c)
    mapped = f.report.get("mapped", {})
    if f.raw is not None and "Date" in mapped and "Well" in mapped:
        used = set(mapped.values())
        def _numeric(col):
            v = f.raw[col].dropna()
            return len(v) >= 10 and pd.to_numeric(v, errors="coerce").notna().mean() > 0.8
        extra_cols = [c for c in f.raw.columns if str(c) not in used and _numeric(c)]
        if extra_cols:
            ex = pd.DataFrame({"Date": smart_dates(f.raw[mapped["Date"]])[0],
                               "Well": f.raw[mapped["Well"]].astype(str).str.strip().str.upper()})
            for c in extra_cols:
                ex[f"x::{c}"] = pd.to_numeric(f.raw[c], errors="coerce")
            ex = ex.dropna(subset=["Date"]).groupby(["Well", "Date"], as_index=False).mean(numeric_only=True)
            d = d.merge(ex, on=["Well", "Date"], how="left")
            for c in extra_cols:
                if d[f"x::{c}"].notna().sum() >= 10 and d[f"x::{c}"].nunique() > 1:
                    opts[f"x::{c}"] = f"{c}  (from your file)"
    return d, opts


def run_forecast(f: "Field", var: str, H: int, d: pd.DataFrame | None = None) -> dict | None:
    """H is in the data's main spacing (e.g. days). A measurement recorded less often (e.g. monthly well tests) is
    forecast on its own spacing, with the horizon converted, so no in-between values are ever invented."""
    if d is None:
        d, _ = forecastable(f)
    if var not in d.columns or d[var].notna().sum() < 3:
        return None
    main, _ = detect_frequency(f.df)
    code = variable_cadence(d, var, main)
    steps = int(max(1, np.ceil(H * FREQ_DAYS[main] / FREQ_DAYS[code] - 1e-9)))
    series = regular_series(d, var, code)
    if not series or not series["wells"]:
        return None
    return forecast_series(f"{f.key}|{var}|{code}", var, steps, code, series)


def fc_unit(var: str) -> str:
    return "" if var.startswith("x::") else FC_UNITS.get(var, "")


def fc_label(var: str) -> str:
    return var[3:] if var.startswith("x::") else FC_LABELS.get(var, FIELD_LABELS.get(var, var))


def to_source_units(var: str, values, report: dict):
    """Convert standard-unit values back into the unit the user's file used (for export)."""
    conv = {c["Field"]: c["From"] for c in report.get("conversions", [])}
    unit = conv.get(var)
    if not unit or var not in UNITS:
        return values, fc_unit(var)
    offset = -32.0 * 5.0 / 9.0 if unit == "°F" else 0.0
    return (np.asarray(values, dtype=float) - offset) / UNITS[var][unit], unit


def steps_from_text(text: str, code: str, default: int) -> int:
    """'next 3 months' with daily data → 90 steps, etc."""
    m = re.search(r"(?:next|coming|following)\s+(\d+)\s*(hour|day|week|month|quarter|year)s?", text.lower())
    if not m:
        m2 = re.search(r"(?:next|coming|following)\s+(hour|day|week|month|quarter|year)\b", text.lower())
        if not m2:
            return default
        n, unit = 1, m2.group(1)
    else:
        n, unit = int(m.group(1)), m.group(2)
    days = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30.44, "quarter": 91.3, "year": 365.25}
    per = {"h": 1 / 24, "D": 1, "W": 7, "M": 30.44, "Q": 91.3, "Y": 365.25}[code]
    return int(max(1, min(365, round(n * days[unit] / per))))


# ---- 3f. Ask FIELDWISE — answer free-form questions about the uploaded sheet ---------
# Offline engine: understands common production questions without any API, and can use ANY numeric column
_QUANTITIES = [  # (keywords, label, how to compute from a frame, unit, is_rate)
    (("energy per barrel", "per barrel", "kwh per bbl", "kwh/bbl", "kwh bbl", "specific energy", "energy intensity"),
     "Energy per barrel", lambda g: g["Energy_kWh"].sum() / max(g["Oil_Rate"].sum() * M3_TO_BBL, 1e-9), "kWh/bbl", False),
    (("water cut", "watercut", "bsw", "wc"), "Water cut", lambda g: 100 * g["Water_Rate"].sum() / max(g["Water_Rate"].sum() + g["Oil_Rate"].sum(), 1e-9), "%", False),
    (("gas oil ratio", "gas-oil ratio", "gor"), "GOR", lambda g: g["Gas_Rate"].sum() / max(g["Oil_Rate"].sum(), 1e-9), "Sm³/m³", False),
    (("water oil ratio", "wor"), "WOR", lambda g: g["Water_Rate"].sum() / max(g["Oil_Rate"].sum(), 1e-9), "m³/m³", False),
    (("liquid", "gross"), "Liquid rate", "Liquid_Rate", "m³/d", True),
    (("lift gas", "gas lift", "injection"), "Lift gas", "Injection_Rate", "", True),
    (("oil", "production", "crude"), "Oil rate", "Oil_Rate", "m³/d", True),
    (("water",), "Water rate", "Water_Rate", "m³/d", True),
    (("gas",), "Gas rate", "Gas_Rate", "Sm³/d", True),
    (("wellhead", "whp", "thp", "tubing head"), "Wellhead pressure", "Wellhead_Pressure", "bar", False),
    (("bottomhole", "bottom hole", "bhp", "downhole"), "Bottom-hole pressure", "Bottomhole_Pressure", "bar", False),
    (("choke", "bean"), "Choke size", "Choke_Size", "/64 in", False),
    (("vibration",), "Pump vibration", "Pump_Vibration", "mm/s", False),
    (("temperature", "temp"), "Motor temperature", "Motor_Temperature", "°C", False),
    (("current", "amps"), "Motor current", "Motor_Current", "A", False),
    (("frequency", "hz"), "Pump frequency", "Pump_Frequency", "Hz", False),
    (("energy", "kwh", "power"), "Energy", "Energy_kWh", "kWh", True),
]


def _offline_forecast(question: str, q: str, f: "Field", named: list) -> dict:
    code, per = detect_frequency(f.df)
    steps = steps_from_text(question, code, DEFAULT_STEPS.get(code, 12))
    pick = [("water cut", "Water_Cut"), ("liquid", "Liquid_Rate"), ("gross", "Liquid_Rate"), ("water", "Water_Rate"),
            ("gas lift", "Injection_Rate"), ("lift gas", "Injection_Rate"), ("gas", "Gas_Rate"),
            ("bottom", "Bottomhole_Pressure"), ("bhp", "Bottomhole_Pressure"), ("wellhead", "Wellhead_Pressure"),
            ("whp", "Wellhead_Pressure"), ("thp", "Wellhead_Pressure"), ("vibration", "Pump_Vibration"),
            ("temperature", "Motor_Temperature"), ("current", "Motor_Current"), ("energy", "Energy_kWh")]
    d, opts = forecastable(f)
    var = None
    for x in opts:                                  # the file's own extra columns, matched by their words
        if x.startswith("x::"):
            words = [w for w in re.findall(r"[a-z]+", _strip_units(x[3:])) if len(w) > 3]
            if words and any(f" {w}" in q for w in words):
                var = x
                break
    var = var or next((v for k, v in pick if f" {k}" in q), "Oil_Rate")
    if var not in opts:
        return {"engine": "Offline engine · forecasting model", "result": None, "fig": None,
                "answer": f"This file has no usable {fc_label(var).lower()} data to forecast. I can forecast: "
                          f"{', '.join(opts.values())}."}
    r = run_forecast(f, var, steps, d)
    if r is None:
        return {"engine": "Offline engine", "result": None, "fig": None,
                "answer": f"Not enough history to forecast {fc_label(var).lower()}."}
    fc = r["fc"] if not named else r["fc"][r["fc"]["Well"].isin(named)]
    agg = "sum" if (var in SUMMABLE and r["density"] >= TEST_MODE_THRESHOLD) else "mean"
    tot = fc.groupby("Date")[["P10", "P50", "P90"]].agg(agg).reset_index()
    unit = fc_unit(var)
    scope = (", ".join(named) if named else "the field") + \
        ("" if len(named) == 1 else (" (sum of wells)" if agg == "sum" else " (average per well)"))
    if r["code"] != code:
        scope += f" — measured only every few days, so forecast per {CODE_LABEL[r['code']]} from real tests only"
    vol = f" — about **{tot['P50'].sum():,.0f} {unit.replace('/d', '')}** in total" if unit.endswith("/d") and per == "day" else ""
    ans = (f"Predicted {fc_label(var).lower()} for {scope}, next **{steps} {per}s**: average **{tot['P50'].mean():,.1f} {unit}**"
           f" (likely range {tot['P10'].mean():,.1f}–{tot['P90'].mean():,.1f}){vol}. "
           f"Blind-test accuracy {fmt(r['accuracy'], '.1f', '%')} on the last {steps} {per}s. See the Forecast page for details.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(tot.Date) + list(tot.Date[::-1]), y=list(tot.P90) + list(tot.P10[::-1]), fill="toself",
                             fillcolor="rgba(251,191,36,.15)", line=dict(width=0), name="Likely range"))
    fig.add_trace(go.Scatter(x=tot.Date, y=tot.P50, name="Forecast", line=dict(color=COLORS["amber"], dash="dash")))
    per_well = fc.pivot_table(index="Well", values="P50", aggfunc="mean").rename(columns={"P50": f"Avg next {steps} {per}s"})
    return {"engine": "Offline engine · forecasting model", "answer": ans, "result": per_well.reset_index(), "fig": fig}


def ask_offline(question: str, f: "Field") -> dict:
    canon = lambda t: " " + " ".join(re.sub(r"[^a-z0-9&]+", " ", str(t).lower()).split()) + " "  # noqa: E731
    q = canon(question)
    d = enrich(f.df)
    wells = sorted(d["Well"].unique())
    named = [w for w in wells if canon(w) in q]
    if named:
        d = d[d["Well"].isin(named)]
    m = re.search(r"last (\d+)\s*(day|week|month|year)s?", q)
    if m:
        days = int(m.group(1)) * {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)]
        d = d[d["Date"] > d["Date"].max() - pd.Timedelta(days=days)]
    for y in re.findall(r"\b(20\d\d)\b", q):
        d = d[d["Date"].dt.year == int(y)]
    if d.empty:
        return {"engine": "Offline engine", "answer": "No rows match that well / period.", "result": None, "fig": None}
    if re.search(r"\b(predict|prediction|forecast|projection|project|future|will be|expected|next \d*\s*"
                 r"(hour|day|week|month|quarter|year))", q):
        return _offline_forecast(question, q, f, named)

    # Which quantity? built-in engineering quantities first, then any other column in the sheet by name similarity
    declining = any(k in q for k in ("declin", "dropping", "falling", "decreas"))
    qty = next((x for x in _QUANTITIES if any(canon(k) in q or canon(k + "s") in q for k in x[0])), None)
    if qty and isinstance(qty[2], str) and (qty[2] not in d.columns or d[qty[2]].isna().all()):
        qty = None
    if qty and qty[1] == "Energy per barrel" and "Energy_kWh" not in d.columns:
        return {"engine": "Offline engine", "result": None, "fig": None,
                "answer": "Energy per barrel needs an energy (kWh) column, which this sheet does not have."}
    if qty is None and declining:
        qty = next(x for x in _QUANTITIES if x[2] == "Oil_Rate")
    col_label, unit, is_rate, func, extra = None, "", False, None, None
    if qty:
        _, col_label, how, unit, is_rate = qty
        func = how if callable(how) else None
        series_col = None if callable(how) else how
    else:
        words = re.findall(r"[a-z][a-z0-9_]+", q)
        best, score = None, 0.0
        for col in f.raw.columns:
            num = pd.to_numeric(f.raw[col], errors="coerce")
            if num.notna().mean() < 0.3 or num.nunique() < 2:
                continue
            name = _strip_units(col)
            sc = max([difflib.SequenceMatcher(None, name, " ".join(words[i:i + n])).ratio()
                      for n in (1, 2, 3) for i in range(max(1, len(words) - n + 1))] or [0])
            if sc > score:
                best, score = col, sc
        if best is None or score < 0.72:
            cols = ", ".join(map(str, f.raw.columns[:25]))
            return {"engine": "Offline engine", "result": None, "fig": None,
                    "answer": f"I couldn't tell which data that question needs. Columns in your sheet: {cols}. "
                              "Try naming one of them, e.g. “average <column> per well”."}
        extra = pd.DataFrame({"Date": pd.to_datetime(f.raw[f.report["mapped"]["Date"]], errors="coerce"),
                              "Well": f.raw[f.report["mapped"]["Well"]].astype(str).str.strip().str.upper(),
                              "Value": pd.to_numeric(f.raw[best], errors="coerce")}).dropna()
        extra = extra[extra["Well"].isin(d["Well"].unique()) & extra["Date"].between(d["Date"].min(), d["Date"].max())]
        d, series_col, col_label = extra, "Value", f"'{best}'"

    per_well = any(k in q for k in ("per well", "each well", "by well", "which well", "which wells", "every well",
                                    "top", "rank", "compare", "highest", "lowest", "most", "least", "worst", "best"))
    trend = any(k in q for k in ("trend", "over time", "history", "plot", "chart", "graph", "show"))
    agg = ("sum" if any(k in q for k in ("total", "cumulative", "sum", "produced", "how much")) else
           "max" if any(k in q for k in ("max", "highest", "peak", "most", "top", "best")) else
           "min" if any(k in q for k in ("min", "lowest", "least", "worst", "smallest")) else "mean")

    def value(g):
        if func:
            return func(g)
        s = g[series_col].dropna()                       # real measurements only
        if agg == "sum" and is_rate:   # volume = average measured rate × days in the period, per well (no filling)
            days = (g["Date"].max() - g["Date"].min()).total_seconds() / 86400 + 1
            return float(sum(gw[series_col].mean() * days for _, gw in g.dropna(subset=[series_col]).groupby("Well")))
        return float(getattr(s, agg)()) if len(s) else np.nan

    vol_unit = unit.replace("/d", "") if (agg == "sum" and is_rate and unit.endswith("/d")) else unit
    u = f" {vol_unit}" if vol_unit else ""
    if agg == "sum" and is_rate and col_label.endswith(" rate"):
        col_label = col_label[:-5] + " produced"
    if declining:
        rows = []
        col_ = series_col or "Oil_Rate"
        n_w = 3 if is_test_mode(f.df) else 30
        for w, g in d.groupby("Well"):
            g = g.dropna(subset=[col_]).sort_values("Date")          # compare real measurements only
            if col_ in RATE_COLUMNS + ["Liquid_Rate"]:
                g = g[g[col_] > 0]                                    # a shut-in day is downtime, not decline
            a, b = g.tail(n_w)[col_].mean(), g.iloc[-2 * n_w:-n_w][col_].mean()
            rows.append({"Well": w, f"Last {n_w} measurements": a, f"Previous {n_w}": b,
                         "Change (%)": (a / b - 1) * 100 if b else np.nan})
        res = pd.DataFrame(rows).sort_values("Change (%)")
        dec = res[res["Change (%)"] < -2]
        ans = (f"**{len(dec)} of {len(res)} wells** are declining (>2% drop, last {n_w} vs previous {n_w} measurements). "
               f"Steepest: " + ", ".join(f"{r.Well} ({r['Change (%)']:.1f}%)" for _, r in dec.head(3).iterrows())) if len(dec) else \
              f"No well shows a decline of more than 2% between its last two windows of {n_w} measurements."
        return {"engine": "Offline engine", "answer": ans, "result": res, "fig": None}
    if trend:
        if func:
            t = d.groupby("Date").apply(func, include_groups=False).rename(col_label).reset_index()
            fig = px.line(t, x="Date", y=col_label)
        elif d[series_col].notna().mean() < TEST_MODE_THRESHOLD:     # sparse (well tests): plot the real points per well
            t = d.dropna(subset=[series_col])[["Date", "Well", series_col]]
            fig = px.line(t, x="Date", y=series_col, color="Well", markers=True)
        else:
            t = d.groupby(["Date", "Well"])[series_col].mean().reset_index() if (per_well or named) else \
                d.groupby("Date")[series_col].agg("sum" if is_rate else "mean").reset_index()
            fig = px.line(t, x="Date", y=series_col, color="Well" if "Well" in t.columns else None)
        return {"engine": "Offline engine", "answer": f"{col_label} over time" + (f" for {', '.join(named)}" if named else "") + ".",
                "result": None, "fig": fig}
    if per_well or (not named and agg in ("max", "min")):
        res = d.groupby("Well").apply(value, include_groups=False).rename(col_label).reset_index()
        res = res.sort_values(col_label, ascending=(agg == "min"))
        top = res.iloc[0]
        word = {"sum": "total", "max": "highest", "min": "lowest", "mean": "average"}[agg]
        ans = (f"**{top['Well']}** has the {word if agg in ('max', 'min') else 'highest ' + word} {col_label}: "
               f"**{top[col_label]:,.2f}{u}**.")
        return {"engine": "Offline engine", "answer": ans, "result": res, "fig": None}
    val = value(d) if (func or agg == "sum" or not is_rate) else \
        float(d.groupby("Well")[series_col].mean().sum())     # field rate = Σ wells' average measured rate
    scope = ", ".join(named) if named else "the field"
    word = {"sum": "Total", "max": "Maximum", "min": "Minimum", "mean": "Average"}[agg]
    return {"engine": "Offline engine", "answer": f"{word} {col_label} for {scope}: **{val:,.2f}{u}** "
            f"({d['Date'].min():%d %b %Y} → {d['Date'].max():%d %b %Y}).", "result": None, "fig": None}


# =============================================================================
# 4. LIFT-GAS OPTIMISER — exact dynamic programme over discretised gas levels
# =============================================================================
def optimise_lift(curves: dict, wells: list, budget: float, min_frac: float, max_frac: float, units: int = 120):
    unit = budget / units
    options = []
    for w in wells:
        c = curves[w]
        lo = max(min_frac * c.x_now, 0.8 * c.x_min)
        hi = max(min(max_frac * c.x_now, 1.15 * c.x_max), lo)
        lv = np.arange(int(np.floor(lo / unit)), int(np.ceil(hi / unit)) + 1)
        options.append((w, lv, c.oil(lv * unit)))
    dp = np.full(units + 1, -np.inf)
    dp[0] = 0.0
    picks = []
    for _, lv, val in options:
        new, arg = np.full(units + 1, -np.inf), np.full(units + 1, -1)
        for i, (lvl, v) in enumerate(zip(lv, val)):
            if lvl > units:
                continue
            cand = np.full(units + 1, -np.inf)
            cand[lvl:] = dp[:units + 1 - lvl] + v
            better = cand > new
            new[better], arg[better] = cand[better], i
        picks.append(arg)
        dp = new
    if not np.isfinite(dp).any():
        return None
    b = int(np.argmax(dp))
    chosen = [0] * len(options)
    for i in range(len(options) - 1, -1, -1):
        chosen[i] = picks[i][b]
        b -= options[i][1][chosen[i]]
    rows = []
    for (w, lv, val), i in zip(options, chosen):
        c = curves[w]
        new_inj = float(lv[i] * unit)
        rows.append({"Well": w, "Current lift gas": c.x_now, "Optimised lift gas": new_inj,
                     "Oil now (model)": float(c.oil(c.x_now)), "Oil optimised": float(val[i]),
                     "Δ Oil": float(val[i] - c.oil(c.x_now)), "Marginal gain / 1k gas": float(c.marginal(new_inj)),
                     "Curve": c.shape})
    return pd.DataFrame(rows)


# =============================================================================
# 5. UI KIT
# =============================================================================
def _w():
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:  # noqa: BLE001
        return {"use_container_width": True}


W = _w()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--bg:#070D14;--panel:#0E1621;--line:#1E2A3A;--text:#E2E8F0;--muted:#94A3B8;--dim:#64748B;
      --teal:#2DD4BF;--amber:#FBBF24;--red:#F87171;--blue:#60A5FA;}
html,body,.stApp{font-family:'Inter',system-ui,sans-serif;}
.stApp{background:radial-gradient(1200px 600px at 90% -10%,rgba(45,212,191,.08),transparent 60%),
       radial-gradient(900px 500px at -10% 110%,rgba(96,165,250,.07),transparent 60%),var(--bg);color:var(--text);}
.block-container{padding:2.2rem 2.2rem 3rem;max-width:1480px;}
header[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:#0A121C;border-right:1px solid var(--line);}
/* All custom blocks are plain divs (no Streamlit heading anchors), with explicit line-heights → no text overlap */
.fw-brand{display:flex;align-items:center;gap:10px;margin:0 0 14px;}
.fw-logo{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#2DD4BF,#0EA5E9);display:grid;
         place-items:center;font-weight:800;color:#04201C;font-size:16px;flex:0 0 36px;}
.fw-brand-name{font-weight:800;letter-spacing:.16em;font-size:13px;color:var(--text);line-height:1.3;}
.fw-brand-sub{font-size:11px;color:var(--dim);line-height:1.3;}
.fw-hero{border:1px solid var(--line);border-radius:22px;padding:36px 40px;margin-bottom:22px;
         background:linear-gradient(135deg,rgba(45,212,191,.12),rgba(14,22,33,.92) 50%),var(--panel);}
.fw-eyebrow{color:var(--teal);font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;line-height:1.4;margin-bottom:12px;}
.fw-hero-title{font-size:clamp(26px,3vw,40px);font-weight:800;line-height:1.18;margin:0 0 14px;color:#F8FAFC;}
.fw-hero-text{font-size:16px;line-height:1.65;color:var(--muted);max-width:880px;margin:0;}
.fw-page{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;
         margin:0 0 20px;padding-bottom:16px;border-bottom:1px solid var(--line);}
.fw-page-title{font-size:28px;font-weight:800;line-height:1.25;color:#F8FAFC;}
.fw-page-sub{color:var(--muted);font-size:14px;line-height:1.55;margin-top:6px;max-width:860px;}
.fw-pill{display:inline-block;padding:6px 12px;border-radius:999px;font-size:12px;font-weight:600;line-height:1.3;
         background:rgba(45,212,191,.12);color:var(--teal);border:1px solid rgba(45,212,191,.28);white-space:nowrap;}
.fw-kpi{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px 18px;height:116px;
        display:flex;flex-direction:column;justify-content:space-between;overflow:hidden;margin-bottom:14px;}
.fw-kpi-label{color:var(--muted);font-size:11.5px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;line-height:1.3;
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fw-kpi-value{font-family:'JetBrains Mono',monospace;font-size:clamp(17px,1.6vw,25px);font-weight:700;color:#F8FAFC;
              line-height:1.25;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fw-kpi-sub{color:var(--dim);font-size:12px;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fw-kpi.good .fw-kpi-value{color:var(--teal);} .fw-kpi.warn .fw-kpi-value{color:var(--amber);} .fw-kpi.bad .fw-kpi-value{color:var(--red);}
.fw-card-h{margin:20px 0 8px;}
.fw-card-title{font-size:15px;font-weight:700;color:#F1F5F9;line-height:1.35;}
.fw-card-sub{font-size:12.5px;color:var(--dim);line-height:1.4;margin-top:2px;}
.fw-feature{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;min-height:168px;margin-bottom:8px;}
.fw-feature-icon{font-size:22px;line-height:1;margin-bottom:12px;}
.fw-feature-title{font-weight:700;font-size:15px;color:#F1F5F9;line-height:1.35;margin-bottom:6px;}
.fw-feature-text{font-size:13.5px;line-height:1.55;color:var(--muted);}
.fw-step{display:flex;gap:14px;align-items:flex-start;padding:12px 0;border-bottom:1px dashed var(--line);}
.fw-step-n{flex:0 0 30px;height:30px;border-radius:9px;background:rgba(45,212,191,.12);color:var(--teal);font-weight:800;
           display:grid;place-items:center;font-size:13px;}
.fw-step-t{font-weight:600;color:#F1F5F9;font-size:14px;line-height:1.4;}
.fw-step-d{color:var(--muted);font-size:13px;line-height:1.5;margin-top:2px;}
.fw-note{border:1px solid var(--line);border-left:3px solid var(--teal);background:var(--panel);border-radius:10px;
         padding:13px 16px;color:var(--muted);font-size:13.5px;line-height:1.6;margin:12px 0;}
.fw-note.warn{border-left-color:var(--amber);} .fw-note b{color:#F1F5F9;}
.fw-insight{background:linear-gradient(135deg,rgba(45,212,191,.10),rgba(96,165,250,.06));border:1px solid rgba(45,212,191,.25);
            border-radius:14px;padding:14px 18px;margin:4px 0 18px;color:var(--text);font-size:14px;line-height:1.6;}
.fw-insight b{color:var(--teal);}
.fw-foot{color:var(--dim);font-size:11px;text-align:center;margin-top:22px;padding-top:12px;border-top:1px solid var(--line);}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stPlotlyChart"]{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:8px 8px 2px;}
</style>
"""


# Fragments rerun only the page body when a widget on the page changes (not the whole app).
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)


def esc(x) -> str:
    return html.escape(str(x))


def page_header(title, sub="", pill=""):
    pill_html = f'<span class="fw-pill">{esc(pill)}</span>' if pill else ""
    st.markdown(f'<div class="fw-page"><div><div class="fw-page-title">{esc(title)}</div>'
                f'<div class="fw-page-sub">{esc(sub)}</div></div>{pill_html}</div>', unsafe_allow_html=True)


def card_header(title, sub=""):
    sub_html = f'<div class="fw-card-sub">{esc(sub)}</div>' if sub else ""
    st.markdown(f'<div class="fw-card-h"><div class="fw-card-title">{esc(title)}</div>{sub_html}</div>',
                unsafe_allow_html=True)


def kpi(label, value, sub="", tone=""):
    st.markdown(f'<div class="fw-kpi {tone}" title="{esc(label)}: {esc(value)} · {esc(sub)}">'
                f'<div class="fw-kpi-label">{esc(label)}</div><div class="fw-kpi-value">{esc(value)}</div>'
                f'<div class="fw-kpi-sub">{esc(sub)}</div></div>', unsafe_allow_html=True)


def kpi_row(items):
    for col, item in zip(st.columns(len(items)), items):
        with col:
            kpi(*item)


def note(body_html, warn=False):
    st.markdown(f'<div class="fw-note{" warn" if warn else ""}">{body_html}</div>', unsafe_allow_html=True)


def insight(body_html):
    st.markdown(f'<div class="fw-insight">{body_html}</div>', unsafe_allow_html=True)


def chart(fig, height=330, legend=True, **layout):
    """Uniform chart style. Titles sit in card_header (outside the plot), legends get their own top band."""
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=COLORS["text"]), title=None,
        margin=dict(l=8, r=8, t=40 if legend else 14, b=8), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11), title=None),
        hoverlabel=dict(bgcolor="#0E1621", bordercolor="#1E2A3A", font=dict(family="Inter")),
        colorway=[COLORS[c] for c in ("teal", "blue", "amber", "violet", "red")], **layout)
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False, automargin=True)
    st.plotly_chart(fig, config={"displayModeBar": False}, **W)


def _compact(x: float) -> str:
    """18,607 → '18.6k' so ranges fit inside a KPI card."""
    if pd.isna(x):
        return "—"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e4, "k")):
        if abs(x) >= div:
            return f"{x / (div if suf != 'k' else 1e3):,.1f}{suf}"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.1f}"


def table(frame: pd.DataFrame, decimals=1, height=None):
    out = frame.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(decimals)
    st.dataframe(out, hide_index=True, **W, **({"height": height} if height else {}))


def fmt(v, spec=",.1f", unit=""):
    try:
        return f"{v:{spec}}{unit}" if v is not None and np.isfinite(v) else "—"
    except (TypeError, ValueError):
        return "—"


# =============================================================================
# 6. PAGES
# =============================================================================
def page_landing():
    st.markdown('<div class="fw-hero"><div class="fw-eyebrow">IEW 2027 · AI production optimisation for mature fields</div>'
                '<div class="fw-hero-title">More barrels, fewer failures, less energy per barrel.</div>'
                '<p class="fw-hero-text">FIELDWISE AI learns every well from its own operating history — pressures, '
                'choke size, lift gas and pump telemetry — then estimates rates between well tests, forecasts decline, '
                're-allocates lift gas for maximum oil and flags artificial-lift problems before they become failures. '
                'Upload your field data in the sidebar to begin.</p></div>', unsafe_allow_html=True)
    feats = [("🛢️", "AI virtual flow meter", "Daily oil rate for every well from pressures, choke and lift gas — tested on dates it never saw."),
             ("📉", "Decline & forecast", "Arps decline curve per well with a 180-day production forecast."),
             ("⚖️", "Lift-gas optimiser", "Learns each well's gas-lift curve and redistributes gas for maximum field oil."),
             ("🛠️", "Equipment early warning", "Per-well anomaly detection on pump signals with projected days-to-alert.")]
    for col, (icon, title, text) in zip(st.columns(4), feats):
        with col:
            st.markdown(f'<div class="fw-feature"><div class="fw-feature-icon">{icon}</div><div class="fw-feature-title">'
                        f'{title}</div><div class="fw-feature-text">{text}</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        card_header("Get started", "everything stays idle until your data arrives")
        steps = [("Download a template", "Excel or CSV — or use your own export; common header names are recognised."),
                 ("Upload in the sidebar", "CSV, XLSX or XLS. Choose the sheet and declare units (psi, bbl/d, inch chokes …)."),
                 ("Check the quality gate", "See exactly what was mapped, converted, repaired or ignored."),
                 ("Act on the insights", "Models train automatically and every page updates from your data.")]
        st.markdown("".join(f'<div class="fw-step"><div class="fw-step-n">{i}</div><div><div class="fw-step-t">{t}</div>'
                            f'<div class="fw-step-d">{d}</div></div></div>' for i, (t, d) in enumerate(steps, 1)),
                    unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            xlsx = template_xlsx()
            if xlsx is not None:
                st.download_button("⬇ Excel template", xlsx, "fieldwise_template.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **W)
            else:
                st.caption("Excel support not installed — use the CSV template.")
        with c2:
            st.download_button("⬇ CSV template", template_frame().to_csv(index=False).encode(),
                               "fieldwise_template.csv", "text/csv", **W)
    with right:
        card_header("Data columns", "one row = one well on one date · Choke_Size and sensors are optional")
        table(schema_frame(), height=530)


def field_oil_daily(f: Field, vfm) -> tuple:
    """Daily field oil total and how it was obtained.
    Daily-rate files: the measured totals. Well-test files: real test values where measured, the AI estimate
    (from that day's pressures / lift gas / pump signals) on the days in between — never interpolation."""
    if not is_test_mode(f.df):
        return f.df.groupby("Date")["Oil_Rate"].sum(min_count=1).dropna().sort_index(), "measured"
    if not vfm:
        return pd.Series(dtype=float), "unavailable"
    p = vfm["pred"].assign(Best=lambda x: x["Oil_Rate"].fillna(x["Pred"]))
    n = p.groupby("Date")["Well"].nunique()
    full = n[n >= 0.9 * f.df["Well"].nunique()].index            # only days where (almost) every well is covered
    return p[p["Date"].isin(full)].groupby("Date")["Best"].sum().sort_index(), "AI estimate between tests"


def field_briefing(f: Field, ranked, vfm, dca, curves) -> list:
    """Plain-language interpretation of the analytics — what an engineer would say in a morning meeting."""
    out = []
    daily, how = field_oil_daily(f, vfm)
    if f.report.get("test_mode"):
        km = f.report.get("kept_as_measured", {}).get("Oil_Rate", {})
        out.append(f"Oil, water and gas come from <b>{km.get('measured', 0):,} well tests</b> (about every "
                   f"{fmt(f.report.get('test_gap_days'), '.0f')} days per well). They are used exactly as measured — "
                   f"no values were filled in; days between tests are estimated by the AI from the continuous signals.")
    if len(daily) >= 60:
        now, before = daily.tail(14).mean(), daily.iloc[-44:-30].mean()
        chg = (now / before - 1) * 100 if before else 0
        out.append(f"Field oil is <b>{now:,.0f} m³/d</b> (≈ {now * M3_TO_BBL:,.0f} bbl/d), "
                   f"<b>{chg:+.1f}%</b> versus a month earlier.")
    wc = 100 * ranked["Water_Rate"].sum() / max(ranked["Liquid_Rate"].sum(), 1e-9)
    hi_wc = ranked.loc[ranked["Water_Cut"] >= 90, "Well"].tolist()
    out.append(f"Field water cut is <b>{wc:.0f}%</b>" + (f"; {len(hi_wc)} well(s) above 90% ({esc(', '.join(hi_wc[:4]))}"
               f"{'…' if len(hi_wc) > 4 else ''}) are candidates for water shut-off or choke-back." if hi_wc else "."))
    eq_bad = ranked[ranked["Status"].isin(["ALERT", "EARLY WARNING"])]
    if len(eq_bad):
        parts = []
        for _, r in eq_bad.head(4).iterrows():
            days = r["Days to alert"]
            state = "alert" if r["Status"] == "ALERT" else (f"warning (~{days:.0f} d to alert)" if pd.notna(days) else "warning")
            parts.append(f"<b>{esc(r['Well'])}</b> {state}")
        out.append("Artificial-lift equipment needs attention: " + ", ".join(parts) + " — plan inspection before failure.")
    elif f.sensors:
        out.append("No equipment anomalies — all pumps within their own normal range.")
    if curves and f.has_lift:
        res = optimise_lift(curves, sorted(curves), sum(c.x_now for c in curves.values()), 0.5, 1.6)
        if res is not None:
            gain = res["Oil optimised"].sum() - res["Oil now (model)"].sum()
            usd = gain * M3_TO_BBL * 365 * f.report.get("oil_price", 70) / 1e6
            if gain > 0.05:
                out.append(f"Re-allocating the same lift gas could add <b>{gain:+,.1f} m³/d</b> "
                           f"(≈ {gain * M3_TO_BBL:+,.0f} bbl/d, ~${usd:,.2f}M/yr) — see Lift-Gas Optimiser.")
    if not dca.empty:
        worst = dca.sort_values("Decline (%/yr)", ascending=False).head(2)
        out.append(f"Median decline is <b>{dca['Decline (%/yr)'].median():.0f}%/yr</b>; fastest: " +
                   ", ".join(f"{esc(r.Well)} ({r['Decline (%/yr)']:.0f}%/yr)" for _, r in worst.iterrows()) + ".")
    if vfm:
        va = vfm["metrics"].iloc[1]
        out.append(f"The AI rate model is <b>{va['Accuracy (%)']:.1f}%</b> accurate on unseen dates — reliable enough to "
                   f"track every well daily between well tests.")
    if f.report.get("auto_units"):
        out.append("Units auto-detected from headers: " + esc(", ".join(f.report["auto_units"])) + ".")
    return out


def page_overview(f: Field, ranked, vfm, dca, fc, curves=None):
    page_header("Field Command Centre",
                f"{f.name} · {f.df['Date'].min():%d %b %Y} → {f.df['Date'].max():%d %b %Y}",
                f"{f.df['Well'].nunique()} wells · {len(f.df):,} records")
    oil, water = ranked["Oil_Rate"].sum(), ranked["Water_Rate"].sum()
    wc = 100 * water / max(oil + water, 1e-9)
    acc = vfm["metrics"].iloc[1]["Accuracy (%)"] if vfm else np.nan
    alerts, warns = int((ranked["Status"] == "ALERT").sum()), int((ranked["Status"] == "EARLY WARNING").sum())
    tests = bool(f.report.get("test_mode"))
    kpi_row([("Field oil · latest tests" if tests else "Field oil · 7-day avg", f"{oil:,.0f} m³/d", f"≈ {oil * M3_TO_BBL:,.0f} bbl/d"),
             ("Field water cut", f"{wc:.1f}%", "liquid-weighted", "warn" if wc >= 80 else ""),
             ("AI model accuracy", fmt(acc, ".1f", "%"), "blind test on later dates", "good" if acc >= 90 else "warn"),
             ("Equipment", f"{alerts} alert · {warns} warning" if f.sensors else "no sensors", "artificial lift",
              "bad" if alerts else ("warn" if warns else "good")),
             ("Wells needing action", int(ranked["Priority"].isin(["CRITICAL", "HIGH"]).sum()), "critical + high",
              "bad" if (ranked["Priority"] == "CRITICAL").any() else "")])
    if "kWh_per_bbl" in ranked.columns:
        field_kwh_bbl = ranked["Energy_kWh"].sum() / max(oil * M3_TO_BBL, 1e-9)
        kpi_row([("Energy intensity", fmt(field_kwh_bbl, ".2f", " kWh/bbl"), "field average · ESG"),
                 ("Carbon intensity", fmt(field_kwh_bbl * f.report.get("co2", GRID_CO2), ".2f", " kg CO₂/bbl"), "grid electricity"),
                 ("Field energy", fmt(ranked["Energy_kWh"].sum(), ",.0f", " kWh/d"), "7-day avg"),
                 ("Median decline", fmt(dca["Decline (%/yr)"].median() if not dca.empty else np.nan, ".1f", " %/yr"), "Arps"),
                 ("Lift gas in use", fmt(ranked["Injection_Rate"].sum(), ",.0f", f" {f.inj_unit}"), "latest day")])
    top = ranked.iloc[0]
    bullets = field_briefing(f, ranked, vfm, dca, curves or {})
    insight("<b>AI field briefing</b><ul style='margin:8px 0 0 18px;padding:0'>" +
            "".join(f"<li style='margin:3px 0'>{x}</li>" for x in bullets) +
            f"<li style='margin:3px 0'>Top priority: <b>{esc(top['Well'])}</b> ({esc(top['Priority'])}) — {esc(top['Why'])}.</li></ul>")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        fig = go.Figure()
        if tests:
            card_header("Field oil · AI daily estimate", "real tests where measured · AI estimate from continuous "
                        "signals on the days between (no interpolation)")
            daily, _ = field_oil_daily(f, vfm)
            fig.add_trace(go.Scatter(x=daily.index, y=daily.values, name="Field oil", line=dict(color=COLORS["teal"], width=2.2)))
        else:
            card_header("Field production · measured vs AI", "daily totals across all wells")
            t = f.df.groupby("Date")[["Oil_Rate", "Water_Rate"]].sum(min_count=1).reset_index()
            fig.add_trace(go.Scatter(x=t.Date, y=t.Water_Rate, name="Water", line=dict(color=COLORS["blue"], width=1.6)))
            fig.add_trace(go.Scatter(x=t.Date, y=t.Oil_Rate, name="Oil · measured", line=dict(color=COLORS["teal"], width=2.2)))
            if vfm:
                p = vfm["pred"].groupby("Date")["Pred"].sum().reset_index()
                fig.add_trace(go.Scatter(x=p.Date, y=p.Pred, name="Oil · AI", line=dict(color=COLORS["amber"], width=1.6, dash="dot")))
        chart(fig, 370, yaxis_title="m³/d")
    with right:
        card_header("Attention queue", "ranked by water cut, decline and equipment risk")
        table(ranked[["Well", "Priority", "Oil_Rate", "Water_Cut", "Why"]].rename(
            columns={"Oil_Rate": "Oil now m³/d", "Water_Cut": "WC % (flowing)"}), 1, height=388)
    a, b = st.columns(2, gap="large")
    with a:
        card_header("Oil vs water cut", "every well at its latest producing rate · bubble = liquid · "
                    "hollow = shut in now · colour = priority")
        pf = ranked.assign(
            Oil=ranked["Oil while flowing"].fillna(ranked["Oil_Rate"]),
            Liquid=ranked["Liquid while flowing"].fillna(ranked["Liquid_Rate"]).fillna(0).clip(lower=1e-6),
            State=np.where(ranked["Shut in since"].notna(), "Shut in", "Producing"),
        ).dropna(subset=["Water_Cut", "Oil"])
        fig = px.scatter(pf, x="Water_Cut", y="Oil", size="Liquid", color="Priority", symbol="State", text="Well",
                         color_discrete_map=PRIORITY_COLORS, size_max=38,
                         symbol_map={"Producing": "circle", "Shut in": "circle-open"},
                         hover_data={"Why": True, "Liquid": ":.0f", "State": True})
        fig.update_traces(textposition="top center", textfont=dict(size=10), marker=dict(line=dict(width=2)))
        chart(fig, 350, xaxis_title="Water cut while producing (%)", yaxis_title="Oil while producing (m³/d)")
    with b:
        card_header("180-day oil outlook", "sum of per-well decline curves")
        if not fc.empty:
            hist = field_oil_daily(f, vfm)[0].rolling(7, min_periods=1).mean()
            fut = fc.dropna(subset=["Forecast"]).groupby("Date")["Forecast"].sum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History · 7-day avg", line=dict(color=COLORS["teal"])))
            fig.add_trace(go.Scatter(x=fut.index, y=fut.values, name="Forecast", line=dict(color=COLORS["amber"], dash="dash")))
            chart(fig, 350, yaxis_title="m³/d")


@_fragment
def page_model(f: Field, vfm):
    page_header("AI Production Model", "Virtual flow meter — daily oil rate for every well from pressures, choke size, "
                "lift gas and pump signals, between periodic well tests.", "Boosted trees + per-well regression")
    if vfm is None:
        st.error("The model could not be trained — the dataset needs more history per well.")
        return
    m = vfm["metrics"]
    tr, va, base = m.iloc[0], m.iloc[1], m.iloc[2]
    gain = (base["MAE (m³/d)"] - va["MAE (m³/d)"]) / max(base["MAE (m³/d)"], 1e-9) * 100
    kpi_row([("Historical match", f"{vfm['history']['Accuracy (%)']:.1f}%", f"R² {vfm['history']['R²']:.4f} · all history", "good"),
             ("Blind-test accuracy", f"{va['Accuracy (%)']:.1f}%", f"R² {va['R²']:.3f} · unseen later dates",
              "good" if va["Accuracy (%)"] >= 90 else "warn"),
             ("Within ±10 %", f"{va['Within ±10 % (%)']:.0f}%", "of blind-test predictions"),
             ("Mean error", f"{va['MAE (m³/d)']:.2f} m³/d", f"≈ {va['MAE (m³/d)'] * M3_TO_BBL:.1f} bbl/d per well-day"),
             ("vs last well test", f"{gain:+.0f}%", "error reduction vs industry practice", "good" if gain > 0 else "warn")])
    tests = vfm.get("test_mode")
    if tests:
        insight(f"Learned only from <b>{vfm['n_tests']:,} real well tests</b> — no rate was interpolated or filled in. "
                f"Between tests it estimates oil from that day's pressures, lift gas, choke and pump signals, covering "
                f"<b>{vfm['untested_days']:,} untested well-days</b>. Scored on <b>{int(va['Rows']):,}</b> later tests it "
                f"never saw: <b>{va['Accuracy (%)']:.1f}%</b> accurate — and unlike interpolation it does not need the "
                f"next test to exist, so it also covers the days since the latest test.")
    else:
        insight(f"Trained on <b>{int(tr['Rows']):,}</b> measured well-days and tested blind on <b>{int(va['Rows']):,}</b> later "
                f"well-days. With a well test every <b>{f.report.get('interval', 14)}</b> measurements, the AI fills every day "
                f"in between at <b>{va['Accuracy (%)']:.1f}%</b> volumetric accuracy — fewer separator tests, earlier "
                f"detection of rate loss.")
    p = vfm["pred"]
    well = st.selectbox("Show", ["All wells · field total"] + sorted(p["Well"].unique()), key="vfm_well")
    one = not well.startswith("All")
    if one:
        g = p[p["Well"] == well].sort_values("Date")
    elif tests:
        daily, _ = field_oil_daily(f, vfm)
        g = pd.DataFrame({"Date": daily.index, "Oil_Rate": np.nan, "Pred": daily.values, "Pred_Unseen": np.nan})
    else:
        g = p.groupby("Date")[["Oil_Rate", "Pred", "Pred_Unseen"]].sum(min_count=1).reset_index()
    a, b = st.columns([1.45, 1], gap="large")
    with a:
        card_header("Historical match", "dots = real well tests · shaded band = blind-test period, never seen in training"
                    if tests else "shaded band = blind-test period, never seen during training")
        fig = go.Figure()
        fig.add_vrect(x0=vfm["split"], x1=p["Date"].max(), fillcolor=COLORS["amber"], opacity=0.07, line_width=0)
        if tests:
            mt = g.dropna(subset=["Oil_Rate"])
            fig.add_trace(go.Scatter(x=mt.Date, y=mt.Oil_Rate, name="Measured well test", mode="markers",
                                     marker=dict(color=COLORS["teal"], size=8, line=dict(color="#0E1621", width=1))))
        else:
            fig.add_trace(go.Scatter(x=g.Date, y=g.Oil_Rate, name="Measured", line=dict(color=COLORS["teal"], width=2.2)))
        fig.add_trace(go.Scatter(x=g.Date, y=g.Pred, name="AI · full-history model", line=dict(color=COLORS["amber"], width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=g.Date, y=g.Pred_Unseen, name="AI · blind test", line=dict(color=COLORS["violet"], width=2)))
        chart(fig, 390, yaxis_title="Oil (m³/d)")
    with b:
        card_header("Predicted vs measured", "each dot is one real measurement")
        pm = p.dropna(subset=["Oil_Rate"])
        s = pm.sample(min(len(pm), 4000), random_state=1)
        fig = px.scatter(s, x="Oil_Rate", y="Pred", color="Split", opacity=0.55, render_mode="webgl",
                         color_discrete_map={"Train": COLORS["blue"], "Validation": COLORS["amber"]})
        hi = float(max(s["Oil_Rate"].max(), s["Pred"].max()))
        fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=COLORS["muted"], dash="dash"),
                                 showlegend=False))
        chart(fig, 390, xaxis_title="Measured (m³/d)", yaxis_title="AI (m³/d)")
    a, b = st.columns([1, 1.25], gap="large")
    with a:
        card_header("What drives the prediction", "share of the model's decisions using each input")
        imp = vfm["importance"].assign(Feature=lambda x: x["Feature"].str.replace("_", " "))
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[COLORS["teal"]])
        chart(fig, 430, legend=False, xaxis_tickformat=".0%", yaxis_title=None)
    with b:
        card_header("Scorecard", "seen data vs unseen data vs carrying the last well test forward")
        table(m, 3)
        card_header("Accuracy by well")
        table(vfm["per_well"], 1, height=250)
    out = p.assign(Source=np.where(p["Oil_Rate"].notna(), "Measured well test", "AI estimate"))[
        ["Date", "Well", "Source", "Oil_Rate", "Pred"]].rename(columns={"Oil_Rate": "Measured oil (m³/d)",
                                                                     "Pred": "AI estimate (m³/d)"})
    out["AI estimate (bbl/d)"] = out["AI estimate (m³/d)"] * M3_TO_BBL
    st.download_button("⬇ Download daily oil estimates for every well (CSV)", out.round(3).to_csv(index=False).encode(),
                       "fieldwise_daily_oil_estimates.csv", "text/csv",
                       help="Real test values where measured, the AI estimate on every other day — compare with your "
                            "own allocation or truth data.")
    note("<b>Honest accuracy.</b> Field meters carry ~1–3 % uncertainty, so a model that reproduced history at exactly "
         "100 % would be memorising meter noise and fail on new days. FIELDWISE shows both the full-history match and a "
         "blind test on later dates — the number judges and engineers can trust. Same-day oil, water and gas rates are "
         "never used as inputs, and missing rates are never filled in: the model learns only from real measurements.")


def _fc_export(f: Field, results: dict, H: int) -> tuple:
    """(table in standard units, table in the user's own column names + units, range table)."""
    std, own, rng_rows = None, None, None
    mapped = f.report.get("mapped", {})
    for var, r in results.items():
        fc = r["fc"]
        lab = f"{fc_label(var)}" + (f" ({fc_unit(var)})" if fc_unit(var) else "")
        part = fc[["Well", "Date", "P50"]].rename(columns={"P50": lab})
        std = part if std is None else std.merge(part, on=["Well", "Date"], how="outer")
        vals, unit = to_source_units(var, fc["P50"].to_numpy(), f.report)
        head = var[3:] if var.startswith("x::") else mapped.get(var, fc_label(var) + (f" ({unit})" if unit else ""))
        o = pd.DataFrame({"Well": fc["Well"], "Date": fc["Date"], head: vals})
        own = o if own is None else own.merge(o, on=["Well", "Date"], how="outer")
        lo, _ = to_source_units(var, fc["P10"].to_numpy(), f.report)
        hi, _ = to_source_units(var, fc["P90"].to_numpy(), f.report)
        rr = pd.DataFrame({"Well": fc["Well"], "Date": fc["Date"], f"{head} P10": lo, f"{head} P50": vals, f"{head} P90": hi})
        rng_rows = rr if rng_rows is None else rng_rows.merge(rr, on=["Well", "Date"], how="outer")
    date_h, well_h = mapped.get("Date", "Date"), mapped.get("Well", "Well")
    own = own.rename(columns={"Date": date_h, "Well": well_h})[[date_h, well_h] + [c for c in own.columns if c not in ("Well", "Date")]]
    rng_rows = rng_rows.rename(columns={"Date": date_h, "Well": well_h})
    return std.sort_values(["Well", "Date"]), own.sort_values([well_h, date_h]), rng_rows.sort_values([well_h, date_h])


def _xlsx_bytes(sheets: dict) -> bytes | None:
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            for name, frame in sheets.items():
                frame.to_excel(xw, index=False, sheet_name=name[:31])
        return buf.getvalue()
    except ImportError:
        return None


@_fragment
def page_forecast(f: Field):
    code, per = detect_frequency(f.df)
    d, opts = forecastable(f)
    page_header("Forecast", f"Predicts the next {per}s for every well. Each method is tested on recent {per}s it was "
                "not allowed to see; the best ones are blended for the forecast.", f"your data: one value per {per}")
    if not opts:
        st.info("No numeric measurements with enough history to forecast.")
        return
    n_med = int(d.groupby("Well").size().median())
    max_steps = int(max(1, min(365, n_med // 2)))
    c1, c2 = st.columns([2.2, 1])
    with c1:
        default = [v for v in ("Oil_Rate", "Water_Rate", "Gas_Rate") if v in opts][:3] or list(opts)[:1]
        chosen = st.multiselect("What to predict", list(opts), default=default, format_func=lambda v: opts[v],
                                key="fc_vars")
    with c2:
        H = int(st.number_input(f"How many {per}s ahead", min_value=1, max_value=max_steps,
                                value=min(DEFAULT_STEPS.get(code, 12), max_steps, max(1, n_med // 4)), step=1, key="fc_h"))
    if not chosen:
        st.info("Pick at least one measurement to predict.")
        return
    results = {}
    with st.spinner(f"Training and blind-testing {len(chosen)} forecast model(s)…"):
        for v in chosen:
            try:
                r = run_forecast(f, v, H, d)
            except Exception as exc:  # noqa: BLE001 — one bad column must not stop the others
                st.warning(f"{opts[v]}: could not be forecast ({exc}).")
                r = None
            if r is not None and len(r["fc"]):
                results[v] = r
    if not results:
        st.warning("Not enough history to forecast the selected measurements (each well needs at least 3 points).")
        return
    main_v = next(iter(results))
    r0, unit0 = results[main_v], fc_unit(main_v)
    per0, n0 = CODE_LABEL[r0["code"]], r0["steps"]
    complete = r0["density"] >= TEST_MODE_THRESHOLD
    agg = "sum" if (main_v in SUMMABLE and complete) else "mean"
    hist = r0["hist"].groupby("Date")["Value"].agg(agg)
    fut = r0["fc"].groupby("Date")[["P10", "P50", "P90"]].agg(agg)
    recent = float(hist.tail(n0).mean()) if len(hist) else np.nan
    nxt = float(fut["P50"].mean())
    chg = (nxt / recent - 1) * 100 if recent else np.nan
    word = "Field total" if agg == "sum" else ("Field average" if complete else "Average per well")
    acc, base = r0["accuracy"], r0["baseline"]
    kpi_row([(f"{fc_label(main_v)} · next {n0} {per0}s", f"{nxt:,.1f} {unit0}", f"{word.lower()}, mean per {per0}"),
             ("Change vs last period", fmt(chg, "+.1f", "%"), f"vs the last {n0} {per0}s", "warn" if chg < -5 else ""),
             ("Likely range (P10–P90)", f"{_compact(fut['P10'].mean())} – {_compact(fut['P90'].mean())}",
              f"{unit0} · 80% chance the real value lands here"),
             ("Blind-test accuracy", fmt(acc, ".1f", "%"), f"on the last {n0} {per0}s it never saw",
              "good" if acc >= 90 else "warn"),
             ("vs last-value guess", fmt(acc - base, "+.1f", " pts"), f"repeating the last value scores {fmt(base, '.1f', '%')}",
              "good" if acc >= base else "")])
    sparse = [v for v, r in results.items() if r["code"] != code or r["density"] < TEST_MODE_THRESHOLD]
    if sparse:
        note("<b>Measured only on some days:</b> " + ", ".join(
            f"{esc(fc_label(v))} (forecast per {CODE_LABEL[results[v]['code']]}, "
            f"{results[v]['density']:.0%} of periods have a reading)" for v in sparse) +
             ". These are modelled from the real readings only — gaps are left empty, never interpolated, and the "
             "blind test is scored only against periods that were really measured.")
    if np.isfinite(acc):
        insight(f"To prove it works, FIELDWISE hid the last <b>{n0} {per0}s</b> of every well, predicted them, and "
                f"compared with the real readings: <b>{acc:.1f}% accurate</b> for {esc(fc_label(main_v).lower())} "
                f"across {r0['tested_wells']} wells (simply repeating the last value scores {base:.1f}%). It then "
                f"retrained on all data to predict the next {n0} {per0}s."
                + (" <i>Short history: methods were chosen on the same window they were scored on, so this accuracy "
                   "is slightly optimistic.</i>" if r0["optimistic"] else ""))
    else:
        note(f"History is too short for a blind test at {H} {per}s ahead — forecasts use simple trend methods. "
             "Try fewer steps ahead for a tested forecast.", warn=True)

    a, b = st.columns([1.35, 1], gap="large")
    with a:
        card_header(f"{word} · {fc_label(main_v)}", f"history and forecast · shaded = likely range (P10–P90)")
        show = hist.tail(max(4 * n0, 60))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(fut.index) + list(fut.index[::-1]), y=list(fut["P90"]) + list(fut["P10"][::-1]),
                                 fill="toself", fillcolor="rgba(251,191,36,.15)", line=dict(width=0),
                                 name="Likely range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=show.index, y=show.values, name="Actual", mode="lines" if complete else "lines+markers",
                                 line=dict(color=COLORS["teal"], width=2)))
        fig.add_trace(go.Scatter(x=fut.index, y=fut["P50"], name="Forecast", line=dict(color=COLORS["amber"], width=2.4, dash="dash")))
        chart(fig, 380, yaxis_title=unit0 or None)
    with b:
        card_header("Forecast accuracy by measurement", "blind test · most recent periods hidden · scored on real readings")
        acc_rows = []
        for v, r in results.items():
            mix = (r["per_well"]["Method"].str.split(r" \+ ").explode().str.replace(r" \d+%$", "", regex=True)
                   .value_counts().head(2).index.tolist())
            acc_rows.append({"Measurement": fc_label(v), "Forecast per": CODE_LABEL[r["code"]],
                             "Blind accuracy (%)": r["accuracy"], "Last-value guess (%)": r["baseline"],
                             "Wells tested": r["tested_wells"], "Main methods": ", ".join(mix)})
        table(pd.DataFrame(acc_rows), 1, height=min(380, 40 + 36 * len(acc_rows)))

    card_header("Check one well", "the dashed purple line is what the model predicted for the hidden period — "
                "compare it with the actual line")
    c1, c2 = st.columns(2)
    with c1:
        v_sel = st.selectbox("Measurement", list(results), format_func=lambda v: opts[v], key="fc_one_var")
    rv = results[v_sel]
    wells = sorted(rv["fc"]["Well"].unique())
    with c2:
        w_sel = st.selectbox("Well", wells, key="fc_one_well")
    g = rv["hist"][rv["hist"]["Well"] == w_sel].sort_values("Date")
    fw = rv["fc"][rv["fc"]["Well"] == w_sel]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(fw.Date) + list(fw.Date[::-1]), y=list(fw.P90) + list(fw.P10[::-1]), fill="toself",
                             fillcolor="rgba(251,191,36,.15)", line=dict(width=0), name="Likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g.Date, y=g["Value"], name="Actual (real readings)",
                             mode="lines" if rv["density"] >= TEST_MODE_THRESHOLD else "markers",
                             line=dict(color=COLORS["teal"], width=1.8), marker=dict(size=7)))
    if w_sel in rv["blind"]:
        bd, _, bp = rv["blind"][w_sel]
        fig.add_trace(go.Scatter(x=bd, y=bp, name="Blind-test prediction", line=dict(color=COLORS["violet"], width=2.2, dash="dot")))
    fig.add_trace(go.Scatter(x=fw.Date, y=fw.P50, name="Forecast", line=dict(color=COLORS["amber"], width=2.4, dash="dash")))
    chart(fig, 360, yaxis_title=fc_unit(v_sel) or None)
    pw = rv["per_well"][rv["per_well"]["Well"] == w_sel]
    if len(pw):
        st.caption(f"{w_sel}: blind-test accuracy {fmt(pw['Blind-test accuracy (%)'].iloc[0], '.1f', '%')} · "
                   f"method: {pw['Method'].iloc[0]}")

    std_t, own_t, rng_t = _fc_export(f, results, H)
    card_header("Forecast table", f"next {H} {per}s for every well · P50 = most likely value")
    table(std_t, 2, height=360)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇ Forecast · your file's format (CSV)", own_t.to_csv(index=False).encode(),
                           "fieldwise_forecast.csv", "text/csv", **W,
                           help="Same column names and units as your uploaded file — ready to compare with actuals.")
    with c2:
        xb = _xlsx_bytes({"Forecast": own_t, "Range P10-P50-P90": rng_t, "Standard units": std_t,
                          "Accuracy": pd.DataFrame(acc_rows)})
        if xb:
            st.download_button("⬇ Forecast workbook (Excel)", xb, "fieldwise_forecast.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **W)
    with c3:
        st.download_button("⬇ With P10 / P90 range (CSV)", rng_t.to_csv(index=False).encode(),
                           "fieldwise_forecast_range.csv", "text/csv", **W)
    note("<b>Assumptions.</b> Wells keep operating as they have recently: no new wells, workovers or planned "
         "shut-ins; choke size and lift gas stay at current settings. P10–P90 is the range that contained 80% of "
         "the blind-test errors at each step ahead. Methods: recent level, linear trend, damped-trend exponential "
         "smoothing, exponential decline, and a gradient-boosted model trained across all wells.")


@_fragment
def page_decline(f: Field, dca, fc):
    page_header("Decline Curves", "Arps decline fitted to each well's 7-day average oil rate — the classic reservoir-"
                "engineering view (for tested, any-measurement predictions use the Forecast page).", "180-day decline outlook")
    if dca.empty:
        st.info("Not enough history per well to fit decline curves.")
        return
    worst = dca.loc[dca["Decline (%/yr)"].idxmax()]
    kpi_row([("Field rate now", f"{dca['Current rate (m³/d)'].sum():,.0f} m³/d", "decline-curve basis"),
             ("Median decline", f"{dca['Decline (%/yr)'].median():.1f} %/yr", "effective annual"),
             ("Steepest decline", str(worst["Well"]), f"{worst['Decline (%/yr)']:.1f} %/yr", "warn"),
             ("Next 180 days", f"{dca['Next 180 d (m³)'].sum():,.0f} m³", f"≈ {dca['Next 180 d (m³)'].sum() * M3_TO_BBL / 1e3:,.0f}k bbl"),
             ("Median fit R²", f"{dca['Fit R²'].median():.2f}", "curve quality")])
    well = st.selectbox("Well", sorted(dca["Well"]), key="dca_well")
    a, b = st.columns([1.45, 1], gap="large")
    with a:
        g, fw = f.df[f.df["Well"] == well], fc[fc["Well"] == well]
        card_header(f"{well} · decline curve", "measured · Arps fit · forecast")
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=g.Date, y=g.Oil_Rate, name="Measured", mode="markers",
                                 marker=dict(size=4, color=COLORS["teal"], opacity=.55)))
        fig.add_trace(go.Scatter(x=fw.Date, y=fw.Fitted, name="Arps fit", line=dict(color=COLORS["blue"], width=2)))
        fig.add_trace(go.Scatter(x=fw.Date, y=fw.Forecast, name="Forecast", line=dict(color=COLORS["amber"], width=2, dash="dash")))
        chart(fig, 390, yaxis_title="Oil (m³/d)")
    with b:
        card_header("Annual decline by well", "longer bar = faster production loss")
        fig = px.bar(dca.sort_values("Decline (%/yr)"), x="Decline (%/yr)", y="Well", orientation="h",
                     color="Decline (%/yr)", color_continuous_scale=["#2DD4BF", "#FBBF24", "#F87171"])
        fig.update_coloraxes(showscale=False)
        chart(fig, 390, legend=False, yaxis_title=None)
    card_header("Decline parameters")
    table(dca, 3)


@_fragment
def page_lift(f: Field, curves, skipped):
    page_header("Lift-Gas Optimiser", "Each well's gas-lift performance curve is learned from its history; an exact "
                "optimiser redistributes the available gas to maximise total field oil.", f"lift gas in {f.inj_unit}")
    if not f.has_lift:
        st.info("No lift-gas / injection column was found, so allocation optimisation is not available for this dataset.")
        return
    if not curves:
        st.info("Lift-gas rates do not vary enough in the history to learn each well's response.")
        return
    wells = sorted(curves)
    cur_total = float(sum(curves[w].x_now for w in wells))
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        budget = st.number_input(f"Total lift gas available ({f.inj_unit})", min_value=0.0, value=round(cur_total, 0),
                                 step=float(max(1.0, round(cur_total * 0.02))))
    with c2:
        lo = st.slider("Min per well (× current)", 0.0, 1.0, 0.5, 0.05)
    with c3:
        hi = st.slider("Max per well (× current)", 1.0, 3.0, 1.6, 0.1)
    res = optimise_lift(curves, wells, budget, lo, hi) if budget > 0 else None
    if res is None:
        st.warning("No feasible allocation — widen the per-well limits or increase the gas available.")
        return
    now, new = res["Oil now (model)"].sum(), res["Oil optimised"].sum()
    kpi_row([("Oil · current allocation", f"{now:,.1f} m³/d", "model basis"),
             ("Oil · optimised", f"{new:,.1f} m³/d", "same gas limit", "good"),
             ("Extra oil", f"{new - now:+,.1f} m³/d", f"≈ {(new - now) * M3_TO_BBL:+,.0f} bbl/d", "good" if new >= now else "bad"),
             ("Annual value", f"${(new - now) * M3_TO_BBL * 365 * f.report.get('oil_price', 70) / 1e6:+,.2f}M",
              f"at ${f.report.get('oil_price', 70):,.0f}/bbl · {(new / now - 1) * 100 if now else 0:+.1f}% oil", "good" if new >= now else "bad"),
             ("Gas used", f"{res['Optimised lift gas'].sum():,.0f}", f"of {budget:,.0f} {f.inj_unit}")])
    move = (res["Optimised lift gas"] - res["Current lift gas"])
    give, take = res.loc[move.idxmin()], res.loc[move.idxmax()]
    insight(f"Gas moves from low-return to high-return wells until each earns about the same extra oil per unit of gas. "
            f"Largest moves: <b>{esc(give.Well)}</b> {give['Current lift gas']:,.0f} → {give['Optimised lift gas']:,.0f} and "
            f"<b>{esc(take.Well)}</b> {take['Current lift gas']:,.0f} → {take['Optimised lift gas']:,.0f} {f.inj_unit}.")
    pick = st.multiselect("Wells on the curve chart", wells, default=wells[:min(5, len(wells))], key="glpc")
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        card_header("Gas-lift performance curves", "● current setting · ◆ optimised setting")
        fig = go.Figure()
        palette = [COLORS[c] for c in ("teal", "blue", "amber", "violet", "red")] + ["#34D399", "#F472B6", "#E2E8F0"]
        for i, w in enumerate(pick):
            c, col = curves[w], palette[i % len(palette)]
            xs = np.linspace(max(0.0, 0.5 * c.x_min), 1.3 * c.x_max, 60)
            r = res[res.Well == w].iloc[0]
            fig.add_trace(go.Scatter(x=xs, y=c.oil(xs), name=w, line=dict(color=col, width=2)))
            fig.add_trace(go.Scatter(x=[c.x_now], y=[r["Oil now (model)"]], mode="markers", showlegend=False,
                                     marker=dict(color=col, size=9, line=dict(color="#0E1621", width=1))))
            fig.add_trace(go.Scatter(x=[r["Optimised lift gas"]], y=[r["Oil optimised"]], mode="markers", showlegend=False,
                                     marker=dict(color=col, size=12, symbol="diamond", line=dict(color="white", width=1))))
        chart(fig, 410, xaxis_title=f"Lift gas ({f.inj_unit})", yaxis_title="Oil (m³/d)")
    with b:
        card_header("Allocation by well", "current vs optimised")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=res.Well, y=res["Current lift gas"], name="Current", marker_color=COLORS["muted"]))
        fig.add_trace(go.Bar(x=res.Well, y=res["Optimised lift gas"], name="Optimised", marker_color=COLORS["teal"]))
        chart(fig, 410, barmode="group", yaxis_title=f"Lift gas ({f.inj_unit})")
    card_header("Recommendation table", "marginal gain = extra m³/d of oil per additional 1,000 units of lift gas")
    table(res, 3)
    if skipped:
        st.caption("Held at current settings: " + "; ".join(f"{w} ({r})" for w, r in skipped.items()))
    st.download_button("⬇ Download recommendations", res.to_csv(index=False).encode(), "fieldwise_lift_gas_plan.csv", "text/csv")
    note("<b>Engineer in the loop.</b> Apply changes in steps, confirm with a well test and respect compressor, "
         "separator and gas-handling limits. FIELDWISE never sends commands to the field.", warn=True)


@_fragment
def page_equipment(f: Field, rows, summ):
    page_header("Equipment Health", "Each well is compared with its own healthy baseline; an Isolation Forest scores "
                "unusual sensor combinations and the recent trend projects the days left before an alert.", "artificial lift")
    if rows is None:
        st.info("Upload at least two equipment sensor columns (vibration, motor temperature, current, frequency, energy) "
                "to enable equipment screening. No synthetic sensor values are ever generated.")
        return
    n_a, n_w = int((summ.Status == "ALERT").sum()), int((summ.Status == "EARLY WARNING").sum())
    kpi_row([("Alerts", n_a, "anomalous now", "bad" if n_a else "good"),
             ("Early warnings", n_w, "trend reaches alert ≤ 14 days", "warn" if n_w else "good"),
             ("Healthy wells", int((summ.Status == "NORMAL").sum()), f"of {len(summ)}"),
             ("Signals used", len(f.sensors), ", ".join(s.split("_")[-1].lower() for s in f.sensors))])
    a, b = st.columns([1, 1.35], gap="large")
    with a:
        card_header("Fleet status", "highest risk first")
        table(summ.sort_values("Score", ascending=False)[["Well", "Status", "Health", "Days to alert", "Main signal"]], 0,
              height=420)
    with b:
        order = summ.sort_values("Score", ascending=False)["Well"].tolist()
        well = st.selectbox("Well", order, key="eq_well")
        g = rows[rows.Well == well]
        card_header(f"{well} · anomaly score", "dashed line = alert threshold · red = anomalous day")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.Date, y=g.Score, name="Score", line=dict(color=COLORS["teal"], width=2),
                                 fill="tozeroy", fillcolor="rgba(45,212,191,.08)"))
        fig.add_hline(y=summ["Threshold"].iloc[0], line_dash="dash", line_color=COLORS["red"])
        al = g[g.Alert]
        fig.add_trace(go.Scatter(x=al.Date, y=al.Score, mode="markers", name="Anomalous",
                                 marker=dict(color=COLORS["red"], size=6)))
        chart(fig, 350, yaxis_title="Anomaly score")
    for col, s in zip(st.columns(min(4, len(f.sensors))), f.sensors[:4]):
        with col:
            card_header(s.replace("_", " "), STD_UNIT.get(s, ""))
            chart(go.Figure(go.Scatter(x=g.Date, y=g[s], line=dict(color=COLORS["blue"], width=1.6))), 210, legend=False)


@_fragment
def page_well(f: Field, ranked, vfm):
    page_header("Well Explorer", "Everything about one well on one screen.")
    well = st.selectbox("Well (ranked by priority)", ranked["Well"].tolist(), key="explore_well")
    r = ranked[ranked["Well"] == well].iloc[0]
    g = enrich(f.df[f.df["Well"] == well])
    tone = {"CRITICAL": "bad", "HIGH": "warn", "WATCH": "warn", "STABLE": "good"}[r["Priority"]]
    tests = bool(f.report.get("test_mode"))
    basis = "average of last 3 tests" if tests else "7-day avg"
    last_t = r.get("Last test")
    kpi_row([("Oil · latest tests" if tests else "Oil · 7-day avg", f"{r['Oil_Rate']:.1f} m³/d",
              f"last test {last_t:%d %b %Y}" if tests and pd.notna(last_t) else f"≈ {r['Oil_Rate'] * M3_TO_BBL:,.0f} bbl/d"),
             ("Water cut", f"{r['Water_Cut']:.1f}%", basis, "warn" if r["Water_Cut"] >= 80 else ""),
             ("Choke size", fmt(r.get("Choke_Size", np.nan), ".0f", "/64 in"), "current bean"),
             ("Decline", fmt(r.get("Decline (%/yr)", np.nan), ".1f", " %/yr"), "Arps"),
             ("Priority", r["Priority"], f"score {r['Score']}", tone)])
    insight(f"<b>{esc(well)}:</b> {esc(r['Why'])}.")
    a, b = st.columns(2, gap="large")
    with a:
        card_header("Oil, water and AI estimate", "dots = real well tests · dotted line = AI estimate between tests"
                    if tests else "")
        fig = go.Figure()
        mode = "markers" if tests else "lines"
        gw = g.dropna(subset=["Water_Rate"]) if tests else g
        go_ = g.dropna(subset=["Oil_Rate"]) if tests else g
        fig.add_trace(go.Scatter(x=gw.Date, y=gw.Water_Rate, name="Water", mode=mode, line=dict(color=COLORS["blue"], width=1.4),
                                 marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=go_.Date, y=go_.Oil_Rate, name="Oil", mode=mode, line=dict(color=COLORS["teal"], width=2),
                                 marker=dict(size=8)))
        if vfm:
            pw = vfm["pred"][vfm["pred"]["Well"] == well]
            fig.add_trace(go.Scatter(x=pw.Date, y=pw.Pred, name="Oil · AI", line=dict(color=COLORS["amber"], width=1.4, dash="dot")))
        chart(fig, 350, yaxis_title="m³/d")
    with b:
        card_header("Pressures and choke size", "choke on the right axis" if f.has_choke else "")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.Date, y=g.Bottomhole_Pressure, name="Bottom-hole", line=dict(color=COLORS["violet"])))
        fig.add_trace(go.Scatter(x=g.Date, y=g.Wellhead_Pressure, name="Wellhead", line=dict(color=COLORS["blue"])))
        extra = {}
        if f.has_choke:
            fig.add_trace(go.Scatter(x=g.Date, y=g.Choke_Size, name="Choke", yaxis="y2",
                                     line=dict(color=COLORS["amber"], shape="hv", dash="dot")))
            extra = {"yaxis2": dict(title="Choke (1/64 in)", overlaying="y", side="right", showgrid=False)}
        chart(fig, 350, yaxis_title="bar", **extra)
    a, b = st.columns(2, gap="large")
    with a:
        card_header("Water cut", "from well tests" if tests else "")
        gc = g.dropna(subset=["Water_Cut"])
        chart(go.Figure(go.Scatter(x=gc.Date, y=gc.Water_Cut, mode="lines+markers" if tests else "lines",
                                   line=dict(color=COLORS["blue"], width=2), fill=None if tests else "tozeroy",
                                   fillcolor="rgba(96,165,250,.08)")), 280, legend=False, yaxis_title="%")
    with b:
        card_header("Lift gas", f.inj_unit)
        chart(go.Figure(go.Scatter(x=g.Date, y=g.Injection_Rate, line=dict(color=COLORS["teal"], width=1.6))),
              280, legend=False, yaxis_title=f.inj_unit)


def page_quality(f: Field):
    r = f.report
    page_header("Data & Quality Gate", "What was recognised, converted, repaired or ignored — fully auditable.", f.name)
    kpi_row([("Input rows", f"{r['input_rows']:,}", "as uploaded"), ("Clean rows", f"{len(f.df):,}", "used by the models", "good"),
             ("Duplicates merged", f"{r['duplicates']:,}", "exact + same well/date"),
             ("Invalid values", f"{r['invalid']:,}", "negative rates / pressures"),
             ("Outliers capped", f"{r['capped']:,}", "per-well 3×IQR")])
    card_header("Automatic restructuring", f"{r.get('n_sheets', 1)} sheet(s) read · every step FIELDWISE took "
                "to turn your layout into one row per well per date")
    note("<br>".join("• " + esc(x) for x in r.get("structure_log", [])) or "The file was already in tidy form.")
    a, b = st.columns(2, gap="large")
    with a:
        card_header("Before", "first rows of your file, exactly as uploaded")
        if r.get("grid_preview") is not None:
            table(r["grid_preview"], 2, height=300)
    with b:
        card_header("After", "the tidy table FIELDWISE works from")
        if r.get("tidy_preview") is not None:
            table(r["tidy_preview"], 2, height=300)
    if r.get("structure_code"):
        with st.expander("Restructuring code"):
            st.code(r["structure_code"], language="python")
    a, b = st.columns(2, gap="large")
    with a:
        card_header("Column mapping", "your header → FIELDWISE field")
        table(pd.DataFrame([{"Field": FIELD_LABELS.get(k, k), "Your column": v,
                             "How recognised": HOW_TEXT.get(r.get("how", {}).get(k, ""), r.get("how", {}).get(k, ""))}
                            for k, v in r["mapped"].items()]), height=430)
    with b:
        card_header("Cleaning log")
        items = [f"Rows without a valid date or well ID removed: <b>{r['bad_keys']}</b>",
                 f"Ignored columns: <b>{esc(', '.join(r['ignored']) or 'none')}</b>"]
        if r.get("injection_rows_removed"):
            c, n = r["injection_rows_removed"]
            items.append(f"Injector rows set aside (column <b>{esc(c)}</b>): <b>{n:,}</b> — "
                         "production analysis uses producing rows only")
        if r.get("non_producers"):
            items.append(f"Wells with no oil production set aside: <b>{esc(', '.join(r['non_producers']))}</b>")
        if r.get("extra_signals"):
            items.append(f"Extra continuous signals also used by the AI: <b>{esc(', '.join(r['extra_signals']))}</b>")
        for c, n in r.get("dead_gauge", {}).items():
            items.append(f"{FIELD_LABELS.get(c, c)} read exactly 0 while flowing (dead gauge) on <b>{n:,}</b> days "
                         "— treated as missing, not as real pressure")
        km = r.get("kept_as_measured", {})
        if km:
            items.append("Well-test rates kept <b>exactly as measured — never filled in</b>: " + ", ".join(
                f"<b>{esc(FIELD_LABELS.get(k, k))}</b> {v['measured']:,} readings"
                + (f" ({v['not_measured']:,} rows without a test)" if v["not_measured"] else "") for k, v in km.items()))
        if r.get("test_mode"):
            items.append(f"Rates come from periodic well tests (about every <b>{fmt(r.get('test_gap_days'), '.0f')} days</b> "
                         "per well); the AI estimates the days between them from the continuous signals.")
        if r["imputed"]:
            items.append(f"Continuous signals — only gaps of up to {GAP_LIMIT} readings bridged: " +
                         ", ".join(f"<b>{esc(k)}</b> {v}" for k, v in r["imputed"].items() if v))
        if r["conversions"]:
            items.append("Units converted: " + ", ".join(f"<b>{esc(c['Field'])}</b> {esc(c['From'])} → {esc(c['To'])}"
                                                         for c in r["conversions"]))
        if r.get("derived"):
            items.append("Derived using petroleum relationships: " + "; ".join(f"<b>{esc(x)}</b>" for x in r["derived"]))
        if r.get("absent"):
            items.append("Not in file (modules adapt automatically): <b>" + esc(", ".join(r["absent"])) + "</b>")
        if r["dropped_optional"]:
            items.append("Dropped (more than 60 % empty): <b>" + esc(", ".join(r["dropped_optional"])) + "</b>")
        items.append("Choke size: " + ("<b>present</b> — gaps carried forward per well, never interpolated"
                                       if f.has_choke else "<b>not supplied</b> — models run without it"))
        if r["injection_missing"]:
            items.append("No lift-gas column — lift-gas optimisation is disabled.")
        note("<br>".join("• " + i for i in items))
        opt = [c for c in OPTIONAL_COLUMNS if c in f.df.columns]
        note(f"<b>Optional fields found ({len(opt)}/{len(OPTIONAL_COLUMNS)}):</b> {esc(', '.join(opt) or 'none')}")
    card_header("Clean data preview", "first 200 rows")
    table(f.df.head(200), 2, height=360)
    st.download_button("⬇ Download clean data (CSV)", f.df.to_csv(index=False).encode(), "fieldwise_clean.csv", "text/csv")


EXAMPLE_QUESTIONS = ["Predict oil production for the next 30 days", "Total oil produced by each well in the last 30 days",
                     "Which wells are declining?", "Average water cut per well",
                     "Show the oil rate trend of the top-priority well", "What is the energy used per barrel for each well?"]


def _render_result(item: dict):
    st.markdown(f'<div class="fw-card-h"><div class="fw-card-title">❓ {esc(item["q"])}</div>'
                f'<div class="fw-card-sub">{esc(item["engine"])}</div></div>', unsafe_allow_html=True)
    st.markdown(item["answer"])
    res = item.get("result")
    if isinstance(res, pd.Series):
        res = res.reset_index() if res.index.name or isinstance(res.index, pd.MultiIndex) else res.to_frame()
    if isinstance(res, pd.DataFrame) and not res.empty:
        table(res.head(500), 2, height=min(420, 38 + 35 * min(len(res), 11)))
    elif isinstance(res, (int, float, np.integer, np.floating)) and np.isfinite(res):
        kpi("Result", f"{res:,.4g}")
    if item.get("fig") is not None:
        try:
            chart(item["fig"], 360)
        except Exception:  # noqa: BLE001
            st.caption("The chart could not be drawn.")
    if item.get("code"):
        with st.expander("How it was calculated"):
            st.code(item["code"], language="python")


@_fragment
def page_ask(f: Field, ranked, dca):
    page_header("Ask FIELDWISE", "Ask simple questions about your data in plain English. Answers are calculated "
                "instantly on your real numbers — fully offline, free and unlimited.", "Offline engine")
    note("<b>What you can ask</b> — mention a measurement (oil, water, gas, water cut, GOR, pressure, lift gas, "
         "choke, vibration, temperature, current, energy, or any column name from your file), and optionally a well "
         "(e.g. <i>W-003</i>) and a period (<i>last 30 days</i>, <i>2025</i>).<br>"
         "• <b>Totals / averages:</b> “Total oil produced by each well”, “Average water cut per well”<br>"
         "• <b>Highest / lowest:</b> “Which well has the highest GOR?”, “Lowest bottom-hole pressure”<br>"
         "• <b>Trends:</b> “Show the oil rate trend of W-003”<br>"
         "• <b>Declines:</b> “Which wells are declining?”<br>"
         "• <b>Predictions:</b> “Predict oil production for the next 30 days”")
    st.caption("Try one:")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 3]:
            if st.button(ex, key=f"ex_{i}", **W):
                st.session_state["ask_q"] = ex.replace("the top-priority well", str(ranked.iloc[0]["Well"]))
    q = st.text_input("Your question", key="ask_q", placeholder="e.g. Which well has the highest water cut?")
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Calculating…"):
            try:
                out = ask_offline(q, f)
            except Exception as exc:  # noqa: BLE001
                out = {"engine": "Offline engine", "answer": f"Sorry — I could not compute that ({exc}). Try rephrasing "
                       "with a measurement name, e.g. “average oil rate per well”.", "result": None, "fig": None}
        st.session_state.setdefault("ask_history", []).insert(0, {"q": q, **out})
    for item in st.session_state.get("ask_history", [])[:6]:
        _render_result(item)


def page_method():
    page_header("Methodology", "How FIELDWISE AI answers Problem Statement 3.")
    rows = [("Manual data collation", "Ingestion & quality gate",
             "Recognises common header variants, converts units (psi, bbl/d, inch chokes …), repairs and logs every change."),
            ("Rates only measured at well tests", "No imputation of rates",
             "Oil, water and gas are used exactly as measured — blanks stay blank. Continuous signals (pressures, lift gas, "
             "pump data) only have gaps of up to 3 readings bridged. Every model learns and is scored on real readings only."),
            ("Periodic well testing", "AI virtual flow meter",
             "Boosted trees + a per-well regression, learned from real tests only, estimate daily oil from pressures, choke, "
             "lift gas and pump signals — more accurate than interpolating, and it needs no future test."),
            ("Declining reservoir pressure", "Decline curves", "Arps decline per well, 180-day outlook and decline ranking."),
            ("Predicting future production", "Forecast",
             "Recent level, trend, damped exponential smoothing, exponential decline and a boosted-tree model across all "
             "wells compete in a rolling blind test; the best are blended, with a P10–P90 range from the test errors."),
            ("Sub-optimal lift-gas allocation", "Lift-gas optimiser",
             "Per-well gas-lift performance curves + exact dynamic programme under the field gas limit."),
            ("Frequent pump burnouts", "Equipment early warning",
             "Per-well baselines, Isolation Forest scoring and trend projection to days-to-alert."),
            ("Energy per barrel · ESG", "Energy & carbon intensity", "kWh/bbl and kg CO₂/bbl from equipment energy."),
            ("Reactive troubleshooting", "Attention queue", "Explainable ranking with the reason behind every flag.")]
    table(pd.DataFrame(rows, columns=["Pain point in the brief", "FIELDWISE module", "How it works"]), 0)
    card_header("Validation discipline")
    note("Models are always tested on the <b>latest 20 % of dates</b>, which they never see during training (no look-ahead). "
         "Same-day oil, water and gas rates are never model inputs, so the AI works on days without a well test.")
    card_header("Technology")
    st.code("Python · pandas · NumPy · SciPy · scikit-learn · Plotly · Streamlit", language="text")


# =============================================================================
# 7. MAIN
# =============================================================================
PAGES = {"Command Centre": "📊", "Ask FIELDWISE": "💬", "Forecast": "🔮", "AI Production Model": "🧠",
         "Decline Curves": "📉", "Lift-Gas Optimiser": "⚖️",
         "Equipment Health": "🛠️", "Well Explorer": "🔎", "Data & Quality": "🧹", "Methodology": "📘"}


def _file_token(up, data: bytes) -> str:
    """Cheap, stable identity for an upload (avoids hashing the whole file on every rerun)."""
    head = hashlib.md5(data[:1 << 20] + data[-(1 << 20):]).hexdigest()
    return f"{up.name}|{len(data)}|{head}"


@st.cache_resource(show_spinner=False, max_entries=6)
def infer_cached(key: str, _raw: pd.DataFrame):
    return infer_columns(_raw)


FIELD_LABELS = {
    "Date": "Date", "Well": "Well name / ID", "Oil_Rate": "Oil rate", "Water_Rate": "Water rate",
    "Gas_Rate": "Gas rate", "Wellhead_Pressure": "Wellhead pressure", "Bottomhole_Pressure": "Bottom-hole pressure",
    "Injection_Rate": "Lift-gas rate", "Choke_Size": "Choke size", "Pump_Vibration": "Pump vibration",
    "Motor_Temperature": "Motor temperature", "Motor_Current": "Motor current", "Pump_Frequency": "Pump frequency",
    "Energy_kWh": "Energy use", "Water_Cut": "Water cut", "Liquid_Rate": "Gross liquid rate", "GOR": "Gas-oil ratio",
    "Hours_On": "Hours on",
}
FIELD_GROUPS = [
    ("Essential", ["Date", "Well", "Oil_Rate"]),
    ("Production", ["Water_Rate", "Gas_Rate", "Wellhead_Pressure", "Bottomhole_Pressure", "Injection_Rate", "Choke_Size",
                    "Hours_On"]),
    ("Pump / equipment", SENSOR_COLUMNS),
    ("Helpers — used to calculate missing rates", AUX_COLUMNS),
]
HOW_TEXT = {"known name": "name match", "keyword in header": "name contains a keyword"}
NOT_IN_FILE = "—  not in this file  —"


@st.cache_resource(show_spinner=False, max_entries=6)
def column_profile(key: str, _raw: pd.DataFrame):
    """For every column: its kind (date / number / text) and an example value — used to filter the dropdowns."""
    kinds, examples = {}, {}
    sample = _raw.head(300)
    for c in _raw.columns:
        v = sample[c].dropna()
        ex = str(v.iloc[0]) if len(v) else ""
        if pd.api.types.is_datetime64_any_dtype(v):
            kinds[c], ex = "date", (f"{v.iloc[0]:%d %b %Y}" if len(v) else "")
        elif len(v) and pd.to_numeric(v, errors="coerce").notna().mean() > 0.8:
            kinds[c] = "number"
            num = pd.to_numeric(v, errors="coerce").dropna()
            ex = f"{num.iloc[0]:,.4g}" if len(num) else ex
        elif len(v) and v.astype(str).head(50).map(_is_dateish).mean() > 0.8:
            kinds[c] = "date"
        else:
            kinds[c] = "text"
        examples[str(c)] = ex[:16]
    return {str(k): v for k, v in kinds.items()}, examples


def _field_options(field: str, kinds: dict, current) -> list:
    want = "date" if field == "Date" else "text" if field == "Well" else "number"
    cols = [c for c, k in kinds.items() if k == want] or list(kinds)
    if current and current not in cols:
        cols = [current] + cols
    return [NOT_IN_FILE] + cols


def sidebar_columns(token: str, raw: pd.DataFrame, auto: dict, how: dict):
    """Plain-language summary of what each column was recognised as; an optional editor to fix a wrong guess.
    Returns (mapping {field: column | None}, set of fields changed by the user)."""
    store = st.session_state.setdefault(f"ovr_{token}", {})        # user corrections survive page changes
    mapping = {f: store.get(f, auto.get(f)) for f in ALL_FIELDS}
    found = [f for f in ALL_FIELDS if mapping.get(f)]
    missing = [f for f in ("Date", "Well") if not mapping.get(f)]
    if not mapping.get("Oil_Rate") and not mapping.get("Liquid_Rate"):
        missing.append("Oil_Rate")
    label = f"Columns · {len(found)} recognised" + (" · needs attention" if missing else "")
    with st.sidebar.expander(label, expanded=bool(missing)):
        if missing:
            st.warning("Couldn't find: " + ", ".join(FIELD_LABELS[m] for m in missing) +
                       ". Turn on **Fix a column** and pick it from your file.")
        else:
            st.caption("FIELDWISE read your headers and values and recognised these. Nothing to do unless one is wrong.")
        rows = "".join(
            f'<div style="padding:5px 0;border-bottom:1px solid #1E2A3A;line-height:1.35">'
            f'<span style="color:#E2E8F0;font-weight:600">{esc(FIELD_LABELS[f])}</span>'
            f'<span style="color:#64748B"> ← </span><span style="color:#2DD4BF">{esc(mapping[f])}</span>'
            f'<div style="color:#64748B;font-size:11px">'
            f'{"changed by you" if f in store else esc(HOW_TEXT.get(how.get(f, ""), how.get(f, "")))}</div></div>'
            for f in found)
        st.markdown(f'<div style="font-size:13px">{rows}</div>', unsafe_allow_html=True)
        not_found = [FIELD_LABELS[f] for f in ALL_FIELDS if not mapping.get(f) and f not in AUX_COLUMNS]
        if not_found:
            st.caption("Not in this file (the app adapts): " + ", ".join(not_found))

        if st.toggle("Fix a column", value=bool(missing), key=f"edit_{token}"):
            kinds, examples = column_profile(token, raw)
            fmt_opt = lambda c: c if c == NOT_IN_FILE else (f"{c}   ·  e.g. {examples[c]}" if examples.get(c) else c)  # noqa: E731
            for title, fields in FIELD_GROUPS:
                st.markdown(f'<div style="color:#94A3B8;font-size:11px;font-weight:700;letter-spacing:.06em;'
                            f'text-transform:uppercase;margin:10px 0 2px">{esc(title)}</div>', unsafe_allow_html=True)
                for f in fields:
                    opts = _field_options(f, kinds, mapping.get(f))
                    cur = mapping.get(f)
                    sel = st.selectbox(FIELD_LABELS[f] + (" *" if f in REQUIRED else ""), opts,
                                       index=opts.index(cur) if cur in opts else 0, format_func=fmt_opt,
                                       key=f"m_{token}_{f}")
                    val = None if sel == NOT_IN_FILE else sel
                    if val != auto.get(f):
                        store[f] = val
                    else:
                        store.pop(f, None)
                    mapping[f] = val
            used = [c for c in mapping.values() if c]
            dup = sorted({c for c in used if used.count(c) > 1})
            if dup:
                st.warning("Used for two fields: " + ", ".join(dup) + " — only the first will be applied.")
            if store and st.button("Reset to automatic", key=f"reset_{token}"):
                store.clear()
                for k in [k for k in st.session_state if str(k).startswith(f"m_{token}_")]:
                    del st.session_state[k]
                st.rerun()
    return mapping, set(store)


def sidebar_upload():
    st.sidebar.markdown("**Field data**")
    up = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls", "xlsm"])
    if up is None:
        return None, None
    data = up.getvalue()
    token = _file_token(up, data)
    try:
        grids = read_file(token, data, up.name)
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not read {up.name}: {exc}"
    stem = os.path.splitext(up.name)[0]
    try:
        raw, s_log, s_code = structure_cached(token, grids, stem)
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not understand the layout of {up.name}: {exc}"
    if raw is None:
        return None, ("Could not find a time-series table in this file. " + " ".join(s_log[-2:]) +
                      " Check that the file has a date column and one row per well per date (see the template).")
    sheet = "auto"
    auto, how = infer_cached(f"{token}|{sheet}", raw)
    mapping, manual = sidebar_columns(token, raw, auto, how)

    units, auto_units = {}, []
    present = [f for f in UNITS if mapping.get(f)]
    guesses = {c: detect_unit(mapping[c], c) or STD_UNIT[c] for c in present}
    auto_units = [f"{FIELD_LABELS[c]}: {u}" for c, u in guesses.items() if u != STD_UNIT[c]]
    with st.sidebar.expander("Units" + (f" · {len(auto_units)} converted" if auto_units else " · all standard"),
                             expanded=False):
        st.caption("The unit each column is in. Read from the header where it says (e.g. 'psi', 'bbl/d'); "
                   "values are converted automatically. Change one only if it is wrong.")
        for col in present:
            keys = list(UNITS[col])
            units[col] = st.selectbox(f"{FIELD_LABELS[col]}  ({mapping[col]})", keys, index=keys.index(guesses[col]),
                                      key=f"u_{token}_{col}")
    map_key = tuple(sorted(mapping.items()))
    try:
        df, rep = clean_cached(f"{token}|{sheet}", raw, tuple(sorted(units.items())), map_key)
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not process {up.name}: {exc}"
    if df is None:
        if rep.get("missing"):
            need = ", ".join(m.replace("_", " ") for m in rep["missing"])
            return None, (f"Could not find: {need}. Open **Columns** in the sidebar, turn on **Fix a column** and "
                          f"pick it from your file (oil can also be calculated from liquid rate + water cut).")
        return None, rep.get("error", "The data could not be cleaned.")
    first = next(iter(grids.values()))
    rep = {**rep, "auto_units": auto_units, "structure_log": s_log, "structure_code": s_code,
           "n_sheets": len(grids), "grid_preview": first.head(12).dropna(how="all", axis=1).astype(str).replace("nan", ""),
           "tidy_preview": raw.head(12),
           "how": {f: ("set manually" if f in manual else how.get(f, "")) for f in rep["mapped"]}}
    key = f"{token}|{sheet}|{sorted(units.items())}|{map_key}"
    return Field(df=df, name=up.name, report=rep, lift_is_gas=units.get("Injection_Rate", "Sm³/d") in _GAS,
                 key=key, raw=raw), None


def sidebar_settings() -> Settings:
    with st.sidebar.expander("Settings", expanded=False):
        return Settings(test_interval=st.select_slider("Well-test interval (observations)", options=[7, 14, 21, 30], value=14),
                        wc_watch=st.slider("Water cut · watch (%)", 30, 95, 70),
                        wc_high=st.slider("Water cut · high (%)", 50, 99, 90),
                        decline_watch=st.slider("Decline · watch (%/yr)", 5, 50, 15),
                        decline_high=st.slider("Decline · high (%/yr)", 10, 80, 30),
                        co2=st.number_input("Grid CO₂ factor (kg/kWh)", min_value=0.0, max_value=2.0, value=GRID_CO2, step=0.01),
                        oil_price=st.number_input("Oil price (USD/bbl)", min_value=0.0, max_value=300.0, value=70.0, step=1.0))


def main():
    st.set_page_config(page_title=APP, page_icon="🛢️", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown('<div class="fw-brand"><div class="fw-logo">F</div><div><div class="fw-brand-name">FIELDWISE AI</div>'
                        '<div class="fw-brand-sub">Mature-field optimisation</div></div></div>', unsafe_allow_html=True)
    field, err = sidebar_upload()
    footer = f'<div class="fw-foot">designed by {esc(AUTHOR)}</div>'

    if field is None:              # dormant until data is supplied
        if err:
            st.error(err)
        page_landing()
        st.sidebar.markdown(footer, unsafe_allow_html=True)
        return

    settings = sidebar_settings()
    if any(str(x).startswith("⚠") for x in field.report.get("structure_log", [])):
        st.warning("FIELDWISE is not fully sure how this file is organised — see **Data & Quality → Automatic "
                   "restructuring** and check the Before/After preview. If it looks wrong, reshape the file to one row per "
                   "well per date (see the template).")
    field.report = {**field.report, "interval": settings.test_interval, "co2": settings.co2, "oil_price": settings.oil_price}
    page = st.sidebar.radio("Navigate", list(PAGES), format_func=lambda p: f"{PAGES[p]}  {p}")
    st.sidebar.markdown(footer, unsafe_allow_html=True)

    df, k = field.df, field.key
    with st.spinner("Reading the field and training models…"):
        try:
            vfm = train_vfm(k, df, settings.test_interval)
        except ValueError:
            vfm = None
        dca, fc = fit_declines(k, df)
        curves, skipped = fit_lift_curves(k, df) if field.has_lift else ({}, {})
        eq_rows, eq_sum = screen_equipment(k, df, tuple(field.sensors))
        ranked = rank_cached(k, df, dca, eq_sum, settings, astuple(settings))

    if page == "Command Centre":
        page_overview(field, ranked, vfm, dca, fc, curves)
    elif page == "Ask FIELDWISE":
        page_ask(field, ranked, dca)
    elif page == "AI Production Model":
        page_model(field, vfm)
    elif page == "Forecast":
        page_forecast(field)
    elif page == "Decline Curves":
        page_decline(field, dca, fc)
    elif page == "Lift-Gas Optimiser":
        page_lift(field, curves, skipped)
    elif page == "Equipment Health":
        page_equipment(field, eq_rows, eq_sum)
    elif page == "Well Explorer":
        page_well(field, ranked, vfm)
    elif page == "Data & Quality":
        page_quality(field)
    else:
        page_method()


if __name__ == "__main__":
    main()
