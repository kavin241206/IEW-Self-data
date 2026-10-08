
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="FIELDWISE AI", page_icon="⛽", layout="wide")

REQUIRED_COLUMNS = [
    "Date", "Well", "Oil_Rate", "Water_Rate", "Gas_Rate",
    "Wellhead_Pressure", "Bottomhole_Pressure", "Injection_Rate"
]
NUMERIC_COLUMNS = [
    "Oil_Rate", "Water_Rate", "Gas_Rate", "Wellhead_Pressure",
    "Bottomhole_Pressure", "Injection_Rate"
]

# Upload a CSV from the sidebar. The same data powers every dashboard page.
uploaded_file = st.sidebar.file_uploader(
    "Upload field data (CSV)", type=["csv"],
    help="Upload historical well data using the required column names."
)
template_df = pd.DataFrame({
    "Date": ["2026-01-01", "2026-01-02"],
    "Well": ["W-001", "W-001"],
    "Oil_Rate": [85.0, 84.2],
    "Water_Rate": [74.0, 75.0],
    "Gas_Rate": [62.0, 61.5],
    "Wellhead_Pressure": [100.0, 99.5],
    "Bottomhole_Pressure": [172.0, 171.5],
    "Injection_Rate": [55.0, 56.0],
})
st.sidebar.download_button(
    "Download CSV template", template_df.to_csv(index=False).encode("utf-8"),
    file_name="fieldwise_data_template.csv", mime="text/csv",
    use_container_width=True
)

# ---------- Synthetic demo data ----------
@st.cache_data
def make_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    wells = [f"W-{i:03d}" for i in range(1, 13)]
    rows = []
    for wi, well in enumerate(wells):
        base_oil = 145 - wi * 5 + rng.normal(0, 4)
        base_water = 30 + wi * 4
        for t, d in enumerate(dates):
            decline = np.exp(-t / (260 + wi * 15))
            injection = 55 + 10*np.sin(t/25 + wi/3) + rng.normal(0, 3)
            bhp = 190 - wi*2 + 7*np.sin(t/30 + wi) + rng.normal(0, 2)
            whp = 105 - wi + 4*np.sin(t/18) + rng.normal(0, 1.5)
            water = max(5, base_water + 0.055*t + 0.16*injection + rng.normal(0, 3))
            gas = max(20, 65 + 12*np.sin(t/20 + wi) + rng.normal(0, 5))
            oil = max(15, base_oil*decline + 0.26*injection + 0.15*bhp - 20
                       - 0.10*water + rng.normal(0, 4))
            rows.append([d, well, oil, water, gas, whp, bhp, injection])
    return pd.DataFrame(rows, columns=[
        "Date","Well","Oil_Rate","Water_Rate","Gas_Rate",
        "Wellhead_Pressure","Bottomhole_Pressure","Injection_Rate"
    ])

# Use uploaded data when it passes validation; otherwise keep the built-in example dataset.
data_source = "Built-in illustrative dataset"
upload_error = None
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        missing = [c for c in REQUIRED_COLUMNS if c not in uploaded_df.columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        uploaded_df = uploaded_df[REQUIRED_COLUMNS].copy()
        uploaded_df["Date"] = pd.to_datetime(uploaded_df["Date"], errors="coerce")
        uploaded_df["Well"] = uploaded_df["Well"].astype(str).str.strip()
        for col in NUMERIC_COLUMNS:
            uploaded_df[col] = pd.to_numeric(uploaded_df[col], errors="coerce")
        uploaded_df = uploaded_df.replace([np.inf, -np.inf], np.nan)
        if uploaded_df[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError("The file contains blank or non-numeric values in required columns. Please clean them and upload again.")
        if (uploaded_df["Oil_Rate"] < 0).any() or (uploaded_df["Water_Rate"] < 0).any() or (uploaded_df["Injection_Rate"] < 0).any():
            raise ValueError("Oil, water and injection rates must be zero or positive.")
        if len(uploaded_df) < 20:
            raise ValueError("Please upload at least 20 historical rows so the model can be trained and tested.")
        if uploaded_df["Well"].nunique() < 1 or uploaded_df["Date"].nunique() < 2:
            raise ValueError("The dataset must contain at least one well and at least two distinct dates.")
        df = uploaded_df.sort_values(["Date", "Well"]).reset_index(drop=True)
        data_source = f"User-uploaded data · {uploaded_file.name}"
    except Exception as exc:
        upload_error = str(exc)
        df = make_data()
else:
    df = make_data()

features = ["Water_Rate","Gas_Rate","Wellhead_Pressure","Bottomhole_Pressure","Injection_Rate"]

def train_model():
    split_date = df["Date"].quantile(0.80)
    tr = df[df.Date <= split_date]
    te = df[df.Date > split_date]
    if len(tr) < 10 or len(te) < 2:
        raise ValueError("Not enough rows on both sides of the chronological train/test split.")
    model = RandomForestRegressor(
        n_estimators=180, max_depth=10, min_samples_leaf=3, random_state=42
    )
    model.fit(tr[features], tr["Oil_Rate"])
    pred = model.predict(te[features])
    return model, mean_absolute_error(te["Oil_Rate"], pred), te.assign(Predicted_Oil=pred)

try:
    model, mae, test_df = train_model()
except Exception as exc:
    # Keep the app usable if an uploaded file cannot support a train/test split.
    upload_error = str(exc)
    df = make_data()
    data_source = "Built-in illustrative dataset (uploaded data could not be modelled)"
    model, mae, test_df = train_model()

latest = df.sort_values("Date").groupby("Well").tail(1).copy()
latest["Water_Cut"] = latest["Water_Rate"]/(latest["Water_Rate"]+latest["Oil_Rate"]).replace(0, np.nan)*100
latest["Water_Cut"] = latest["Water_Cut"].fillna(0)

# ---------- Helpers ----------
def priority(row):
    wc = row["Water_Cut"]
    if row["Oil_Rate"] < 80 and wc > 40:
        return "CRITICAL"
    if row["Oil_Rate"] < 95 or wc > 35:
        return "HIGH"
    if wc > 28 or row["Oil_Rate"] < 110:
        return "WATCH"
    return "STABLE"

latest["Priority"] = latest.apply(priority, axis=1)

def kpi_card(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def section(title, kicker=""):
    st.markdown(f'<div class="section-kicker">{kicker}</div><h2 class="section-title">{title}</h2>', unsafe_allow_html=True)

# ---------- Styling ----------
st.markdown("""
<style>
 .stApp {
  background:
    radial-gradient(circle at 84% 7%, rgba(73, 174, 158, .13), transparent 25%),
    radial-gradient(circle at 12% 88%, rgba(45, 103, 133, .13), transparent 30%),
    linear-gradient(135deg, #061017 0%, #091720 50%, #071018 100%);
  color: #EAF2F7;
}
.stApp:before {
  content: ""; position: fixed; inset: 0; pointer-events: none; opacity: .18;
  background-image: linear-gradient(rgba(112,171,184,.055) 1px, transparent 1px), linear-gradient(90deg, rgba(112,171,184,.055) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: linear-gradient(to bottom, black, transparent 82%);
}
.block-container { padding-top: 1.2rem; max-width: 1450px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#07131b 0%,#091821 100%); border-right:1px solid #20323d; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
[data-testid="stSidebar"] .stRadio > label { color:#8EA3AE; font-size:11px; text-transform:uppercase; letter-spacing:1.4px; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap:5px; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label { border-radius:9px; padding:7px 9px; }
.hero {
  padding: 34px 38px; border: 1px solid #233541; border-radius: 20px;
  background: linear-gradient(135deg,#0c1d28,#0a141b);
  margin-bottom: 22px;
}
.brand { font-size: 14px; letter-spacing: 4px; font-weight: 800; color:#77D6C8; }
.hero h1 { font-size: 44px; margin: 8px 0 6px; }
.hero p { font-size: 17px; color:#AFC1CB; max-width: 850px; }
.badge { display:inline-block; padding:6px 10px; border-radius:20px; background:#14322f; color:#86E3D2; font-size:12px; font-weight:700; }
.kpi { background:#0d1b24; border:1px solid #223541; border-radius:15px; padding:18px; min-height:112px; }
.kpi-label { color:#8EA3AE; font-size:12px; text-transform:uppercase; letter-spacing:1.2px; }
.kpi-value { font-size:29px; font-weight:800; margin-top:6px; }
.kpi-sub { color:#718792; font-size:12px; margin-top:4px; }
.card { background:#0d1b24; border:1px solid #223541; border-radius:15px; padding:20px; }
.section-kicker { color:#65CFC0; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:800; margin-top:10px; }
.section-title { margin-top:3px; }
.alert { border-left:4px solid #65CFC0; background:#0d2026; padding:14px 16px; border-radius:8px; }
.warn { border-left-color:#E7B85B; background:#251f11; }
.small { color:#8EA3AE; font-size:13px; }
.reco { font-size:20px; font-weight:800; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.markdown("## FIELDWISE AI")
st.sidebar.caption("Mature Field Decision Support")
page = st.sidebar.radio(
    "Navigation",
    ["Command Center","Data Input","Well Intelligence","Intervention Lab","Field Analytics","Explainable AI","Methodology"]
)
st.sidebar.markdown("<div style=\"margin-top:28px; padding-top:10px; border-top:1px solid #20323d; color:#6F858F; font-size:11px; letter-spacing:.5px; text-align:center;\">designed by Kavinkarthick</div>", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
 <div class="brand">FIELDWISE AI</div>
 <span class="badge">MATURE FIELD INTELLIGENCE</span>
 <h1>From Field History to Future Recovery.</h1>
 <p>AI-assisted production intelligence that learns from historical field behaviour,
 rapidly evaluates operating scenarios, and prioritises wells for engineering attention.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Command Center ----------
if page == "Command Center":
    section("Field Command Center","EXECUTIVE VIEW")
    if upload_error:
        st.warning("Uploaded data was not accepted; the dashboard is using the built-in illustrative dataset. Visit Data Input for details.")
    elif uploaded_file is not None and data_source.startswith("User-uploaded"):
        st.success(f"Live dashboard source: {uploaded_file.name}")
    active = latest["Well"].nunique()
    oil = latest["Oil_Rate"].sum()
    wc = np.average(latest["Water_Cut"], weights=latest["Oil_Rate"])
    high = (latest.Priority.isin(["HIGH","CRITICAL"])).sum()
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("Active wells", active, "uploaded field" if data_source.startswith("User-uploaded") else "illustrative field")
    with c2: kpi_card("Current oil", f"{oil:,.0f} m³/d", "field total")
    with c3: kpi_card("Weighted water cut", f"{wc:.1f}%", "production weighted")
    with c4: kpi_card("High-priority wells", high, "AI screening queue")

    left,right = st.columns([1.4,1])
    with left:
        section("Field production trend","PERFORMANCE")
        trend = df.groupby("Date")[["Oil_Rate","Water_Rate"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend.Date,y=trend.Oil_Rate,name="Oil rate",mode="lines"))
        fig.add_trace(go.Scatter(x=trend.Date,y=trend.Water_Rate,name="Water rate",mode="lines"))
        fig.update_layout(template="plotly_dark",height=330,margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        section("AI opportunity queue","PRIORITISATION")
        q = latest.sort_values(["Priority","Water_Cut"], ascending=[True,False])
        order = {"CRITICAL":0,"HIGH":1,"WATCH":2,"STABLE":3}
        q["rank"] = q.Priority.map(order)
        q=q.sort_values(["rank","Water_Cut"],ascending=[True,False]).head(7)
        st.dataframe(q[["Well","Oil_Rate","Water_Cut","Priority"]].rename(columns={
            "Oil_Rate":"Oil","Water_Cut":"Water cut"
        }), hide_index=True, use_container_width=True)

    section("Water-cut evolution","PRODUCTION QUALITY")
    wtrend = df.groupby("Date").apply(
        lambda x: x.Water_Rate.sum()/(x.Water_Rate.sum()+x.Oil_Rate.sum())*100
    ).reset_index(name="Water_Cut")
    fig=px.line(wtrend,x="Date",y="Water_Cut")
    fig.update_layout(template="plotly_dark",height=300,margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig,use_container_width=True)

# ---------- Data Input ----------
elif page == "Data Input":
    section("Bring Your Field Data","DATA IMPORT")
    if upload_error:
        st.error(f"The uploaded file could not be used: {upload_error}")
        st.info("The dashboard is currently using its built-in illustrative dataset. Correct the CSV and upload it again from the left sidebar.")
    elif uploaded_file is not None and data_source.startswith("User-uploaded"):
        st.success(f"Successfully loaded: {uploaded_file.name}")
        st.caption("The uploaded data is now being used across the dashboard and to train the proxy model.")
    else:
        st.info("Upload your historical well data using the CSV uploader in the left sidebar, or download the CSV template there to see the required format.")
    a,b,c = st.columns(3)
    with a: kpi_card("Rows loaded", f"{len(df):,}", "observations")
    with b: kpi_card("Wells found", f"{df['Well'].nunique():,}", "unique well IDs")
    with c: kpi_card("Date range", f"{df['Date'].min():%d %b %Y} – {df['Date'].max():%d %b %Y}", "available history")
    st.markdown("### Required CSV columns")
    requirements = pd.DataFrame({
        "Column": REQUIRED_COLUMNS,
        "Meaning": ["Observation date", "Unique well identifier", "Oil production rate (m³/d)", "Water production rate (m³/d)", "Gas rate (use a consistent unit)", "Wellhead pressure (use a consistent unit)", "Bottom-hole pressure (use a consistent unit)", "Injection rate (use a consistent unit)"],
        "Example": ["2026-01-01", "W-001", "85.0", "74.0", "62.0", "100.0", "172.0", "55.0"]
    })
    st.dataframe(requirements, hide_index=True, use_container_width=True)
    st.caption("Use one row per well per date. Keep units consistent throughout the file. The model predicts Oil_Rate from the five input features shown in the requirements list. At least 20 valid historical rows are required.")
    st.markdown("### Data preview")
    st.dataframe(df.head(20), hide_index=True, use_container_width=True)
    st.caption(f"Active data source: {data_source}")

# ---------- Well Intelligence ----------
elif page == "Well Intelligence":
    section("Well Intelligence","WELL-LEVEL DIAGNOSTICS")
    well = st.selectbox("Select well", latest.Well.tolist())
    hist = df[df.Well==well].copy()
    row = latest[latest.Well==well].iloc[0]
    recent = hist.tail(30)
    oil_change = (recent.Oil_Rate.iloc[-1]/recent.Oil_Rate.iloc[0]-1)*100
    wc = row.Water_Cut

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:kpi_card("Oil rate",f"{row.Oil_Rate:.1f} m³/d","current")
    with c2:kpi_card("Water rate",f"{row.Water_Rate:.1f} m³/d","current")
    with c3:kpi_card("Water cut",f"{wc:.1f}%","current")
    with c4:kpi_card("BHP",f"{row.Bottomhole_Pressure:.1f} bar","current")
    with c5:kpi_card("Priority",row.Priority,"AI screening")

    a,b=st.columns(2)
    with a:
        fig=px.line(hist,x="Date",y="Oil_Rate",title="Oil production history")
        fig.update_layout(template="plotly_dark",height=320)
        st.plotly_chart(fig,use_container_width=True)
    with b:
        hist["Water_Cut"]=hist.Water_Rate/(hist.Water_Rate+hist.Oil_Rate)*100
        fig=px.line(hist,x="Date",y="Water_Cut",title="Water-cut history")
        fig.update_layout(template="plotly_dark",height=320)
        st.plotly_chart(fig,use_container_width=True)

    section("AI diagnosis","ENGINEERING SCREEN")
    reasons=[]
    if oil_change < -5: reasons.append("Recent oil-rate decline is visible.")
    if wc > 35: reasons.append("Water cut is elevated and may limit economic oil production.")
    if row.Bottomhole_Pressure < 180: reasons.append("Current BHP is relatively low within the demonstration field.")
    if not reasons: reasons.append("No major screening trigger detected; continue monitoring.")
    st.markdown('<div class="alert">' + "<br>".join("• "+r for r in reasons) + "</div>", unsafe_allow_html=True)
    note = ("Prioritise a controlled scenario review." if row.Priority in ["HIGH","CRITICAL"]
            else "Continue surveillance and compare against nearby wells.")

# ---------- Intervention Lab ----------
elif page == "Intervention Lab":
    section("Intervention Lab","SCENARIO-BASED OPTIMISATION")
    st.markdown("Test a proposed injection strategy against the current operating point. **Prototype responses are synthetic and illustrative.**")
    well=st.selectbox("Well for scenario", latest.Well.tolist())
    row=latest[latest.Well==well].iloc[0]
    current=row.Injection_Rate
    proposed=st.slider("Proposed injection rate (m³/d)",0.0,120.0,float(np.clip(current,0,120)),1.0)
    run=st.button("RUN AI SCENARIO", type="primary", use_container_width=True)

    if run:
        base=np.array([[row[f] for f in features]])
        base_pred=float(model.predict(base)[0])
        scenarios=np.linspace(0,120,31)
        X=np.repeat(base,31,axis=0)
        X[:,features.index("Injection_Rate")]=scenarios
        oil_preds=model.predict(X)
        # Conservative demo-only water response heuristic
        water_preds=np.maximum(2,row.Water_Rate*(1+0.003*(scenarios-current)))
        wc_preds=water_preds/(water_preds+oil_preds)*100
        idx=int(np.argmin(np.where(wc_preds>55, 1e6, -oil_preds)))
        best_inj=float(scenarios[idx])
        pred_oil=float(model.predict([[row["Water_Rate"],row["Gas_Rate"],row["Wellhead_Pressure"],row["Bottomhole_Pressure"],proposed]])[0])
        pred_water=float(np.maximum(2,row.Water_Rate*(1+0.003*(proposed-current))))
        pred_wc=pred_water/(pred_water+pred_oil)*100

        c1,c2,c3,c4=st.columns(4)
        with c1:kpi_card("Predicted oil",f"{pred_oil:.1f} m³/d","proxy-model estimate")
        with c2:kpi_card("Estimated water",f"{pred_water:.1f} m³/d","demo response")
        with c3:kpi_card("Estimated water cut",f"{pred_wc:.1f}%","demo response")
        with c4:kpi_card("Suggested injection",f"{best_inj:.0f} m³/d","screening optimum")

        fig=go.Figure()
        fig.add_trace(go.Scatter(x=scenarios,y=oil_preds,name="Predicted oil",mode="lines"))
        fig.add_vline(x=proposed,line_dash="dash",annotation_text="Proposed")
        fig.add_vline(x=best_inj,line_dash="dot",annotation_text="Screening optimum")
        fig.update_layout(template="plotly_dark",height=360,xaxis_title="Injection rate (m³/d)",yaxis_title="Predicted oil rate (m³/d)")
        st.plotly_chart(fig,use_container_width=True)

        delta=pred_oil-base_pred
        if proposed > best_inj+5:
            recommendation="Reduce the proposed injection and review a moderate operating point."
        elif proposed < best_inj-5:
            recommendation="A controlled injection increase may be worth evaluating."
        else:
            recommendation="The proposed rate is close to the model's screening optimum."


        st.markdown("### Current vs proposed")
        comp=pd.DataFrame({
            "Metric":["Injection","Oil rate","Water rate","Water cut"],
            "Current":[current,row.Oil_Rate,row.Water_Rate,row.Water_Cut],
            "Proposed":[proposed,pred_oil,pred_water,pred_wc]
        })
        st.dataframe(comp,hide_index=True,use_container_width=True)

    else:
        st.info("Choose a well and proposed injection rate, then run the scenario.")

# ---------- Field Analytics ----------
elif page == "Field Analytics":
    section("Field Analytics","PATTERN DISCOVERY")
    metric=st.selectbox("Visualise metric",["Oil_Rate","Water_Cut","Injection_Rate","Bottomhole_Pressure"])
    if metric=="Water_Cut":
        plot=latest.copy()
        y="Water_Cut"
        title="Current water-cut by well"
    else:
        plot=latest.copy()
        y=metric
        title=f"Current {metric.replace('_',' ')} by well"
    fig=px.scatter(plot,x="Well",y=y,size="Oil_Rate",color="Priority",hover_data=["Oil_Rate","Water_Cut","Injection_Rate"])
    fig.update_layout(template="plotly_dark",height=390)
    st.plotly_chart(fig,use_container_width=True)

    section("Well opportunity map","FIELD VIEW")
    rng=np.random.default_rng(8)
    mapdf=latest.copy()
    mapdf["X"]=rng.uniform(0,100,len(mapdf))
    mapdf["Y"]=rng.uniform(0,60,len(mapdf))
    fig=px.scatter(mapdf,x="X",y="Y",text="Well",color="Priority",size="Oil_Rate",
                   hover_data=["Oil_Rate","Water_Cut","Injection_Rate"])
    fig.update_traces(textposition="top center")
    fig.update_layout(template="plotly_dark",height=450,xaxis_title="Field coordinate X (demo)",
                      yaxis_title="Field coordinate Y (demo)")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Well coordinates are synthetic for demonstration; this is a field-intelligence visualisation, not a geological model.")

# ---------- Explainable AI ----------
elif page == "Explainable AI":
    section("Explainable AI","WHY THE MODEL FLAGS A WELL")
    well=st.selectbox("Select well",latest.Well.tolist())
    row=latest[latest.Well==well].iloc[0]
    importances=pd.Series(model.feature_importances_,index=features).sort_values(ascending=False)
    imp_df=importances.reset_index()
    imp_df.columns=["Feature","Importance"]
    fig=px.bar(imp_df,x="Importance",y="Feature",orientation="h")
    fig.update_layout(template="plotly_dark",height=350)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="card"><b>How to read this</b><br>The chart shows global Random Forest feature importance for the demonstration model. It indicates which input variables were most useful to the model overall; it does not prove physical causality for an individual well.</div>',unsafe_allow_html=True)
    vals=pd.DataFrame({
        "Input":["Water rate","Gas rate","Wellhead pressure","Bottomhole pressure","Injection rate"],
        "Current":[row.Water_Rate,row.Gas_Rate,row.Wellhead_Pressure,row.Bottomhole_Pressure,row.Injection_Rate]
    })
    st.dataframe(vals,hide_index=True,use_container_width=True)

    section("Validation","HELD-OUT DATA")
    st.metric("Test-set MAE",f"{mae:.2f}")
    fig=px.scatter(test_df,x="Oil_Rate",y="Predicted_Oil",opacity=.55)
    lo=min(test_df.Oil_Rate.min(),test_df.Predicted_Oil.min())
    hi=max(test_df.Oil_Rate.max(),test_df.Predicted_Oil.max())
    fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi)
    fig.update_layout(template="plotly_dark",height=380,xaxis_title="Actual oil rate",yaxis_title="Predicted oil rate (m³/d)")
    st.plotly_chart(fig,use_container_width=True)

# ---------- Methodology ----------
elif page == "Methodology":
    section("How FIELDWISE AI Works","TECHNICAL ARCHITECTURE")
    steps=[
        ("01","Historical field data","Production, pressure and injection history form the learning base."),
        ("02","Data quality","Clean, align and prepare time-series observations."),
        ("03","ML proxy model","A Random Forest model learns relationships between operating variables and oil rate."),
        ("04","Well intelligence","Screen wells for decline, water-cut and pressure-related attention."),
        ("05","Scenario engine","Rapidly evaluate alternative injection settings."),
        ("06","Optimisation","Rank scenarios using production response with a water-cut constraint in the demo."),
        ("07","Engineer recommendation","Present the result as decision support—not an autonomous command.")
    ]
    for n,t,d in steps:
        st.markdown(f'<div class="card" style="margin-bottom:10px"><b>{n} · {t}</b><br><span class="small">{d}</span></div>',unsafe_allow_html=True)

    section("Technology stack","IMPLEMENTATION")
    st.code("Python  •  Pandas  •  Scikit-learn  •  Plotly  •  Streamlit",language="text")
    section("Model validation")
    st.write(f"Chronological held-out test evaluation is used. Demonstration test-set MAE: **{mae:.2f} oil-rate units**.")
    section("Innovation statement","WHY THIS IS DIFFERENT")
    st.markdown("""
    <div class="alert">
    <b>FIELDWISE AI is not positioned as a replacement for a physics-based reservoir simulator.</b><br><br>
    It is a decision-support layer that learns from mature-field history, screens well-level opportunities,
    rapidly evaluates operating scenarios and communicates recommendations in an engineer-friendly interface.
    </div>
    """,unsafe_allow_html=True)
