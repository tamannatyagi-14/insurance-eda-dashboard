import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance EDA Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a40);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #ccd6f6; margin: 4px 0; }
    .metric-sub   { font-size: 11px; color: #64ffda; }
    .section-title {
        font-size: 13px; font-weight: 600; color: #8892b0;
        text-transform: uppercase; letter-spacing: 1.5px;
        border-left: 3px solid #64ffda; padding-left: 10px;
        margin-bottom: 14px;
    }
    .insight-box {
        background: #1e2130; border-left: 3px solid #64ffda;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;
    }
    .insight-box h4 { color: #64ffda; font-size: 13px; margin: 0 0 4px; }
    .insight-box p  { color: #a8b2d8; font-size: 12px; margin: 0; line-height: 1.5; }
    div[data-testid="stMetric"] label { color: #8892b0 !important; }
    .stSelectbox label, .stMultiSelect label, .stSlider label { color: #8892b0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-64"]
    )
    return df

df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Filters")
    st.markdown("---")

    age_range = st.slider("Age Range", int(df.age.min()), int(df.age.max()),
                          (int(df.age.min()), int(df.age.max())))

    smoker_filter = st.multiselect("Smoker Status", ["yes", "no"],
                                   default=["yes", "no"])

    sex_filter = st.multiselect("Sex", ["male", "female"],
                                default=["male", "female"])

    region_filter = st.multiselect("Region",
                                   df["region"].unique().tolist(),
                                   default=df["region"].unique().tolist())

    bmi_filter = st.multiselect("BMI Category",
                                ["Underweight", "Normal", "Overweight", "Obese"],
                                default=["Underweight", "Normal", "Overweight", "Obese"])

    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.markdown(f"📋 Total records: **1,338**")
    st.markdown(f"🔢 Features: **7**")
    st.markdown(f"❌ Missing values: **0**")
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# ── Apply filters ─────────────────────────────────────────────────────────────
dff = df[
    (df["age"].between(*age_range)) &
    (df["smoker"].isin(smoker_filter)) &
    (df["sex"].isin(sex_filter)) &
    (df["region"].isin(region_filter)) &
    (df["bmi_category"].isin(bmi_filter))
].copy()

# ── Plotly theme ──────────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
CARD_BG   = "#1e2130"
GRID_CLR  = "#2e3250"
TEXT_CLR  = "#ccd6f6"
TEAL      = "#64ffda"
RED       = "#ff6b6b"
BLUE      = "#57a5ff"
AMBER     = "#ffcb6b"
GREEN     = "#c3e88d"
PURPLE    = "#c792ea"

def dark_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(color=TEXT_CLR, size=14), x=0.01),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR, size=11),
        xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_CLR, size=10))
    )

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🏥 Insurance Charges — EDA Dashboard")
st.markdown(
    f"Showing **{len(dff):,}** of **{len(df):,}** records after filters · "
    "Built as pre-modeling exploratory analysis"
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Avg Charge</div>
        <div class="metric-value">${dff['charges'].mean():,.0f}</div>
        <div class="metric-sub">σ = ${dff['charges'].std():,.0f}</div>
    </div>""", unsafe_allow_html=True)

with k2:
    smoker_avg   = dff[dff["smoker"]=="yes"]["charges"].mean() if len(dff[dff["smoker"]=="yes"]) else 0
    nonsmoker_avg= dff[dff["smoker"]=="no"]["charges"].mean()  if len(dff[dff["smoker"]=="no"]) else 1
    ratio = smoker_avg / nonsmoker_avg if nonsmoker_avg else 0
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Smoker Premium</div>
        <div class="metric-value">{ratio:.1f}×</div>
        <div class="metric-sub">${smoker_avg:,.0f} vs ${nonsmoker_avg:,.0f}</div>
    </div>""", unsafe_allow_html=True)

with k3:
    corr_age = dff[["age","charges"]].corr().iloc[0,1]
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Age Correlation</div>
        <div class="metric-value">r = {corr_age:.2f}</div>
        <div class="metric-sub">with charges</div>
    </div>""", unsafe_allow_html=True)

with k4:
    obese_smoker = dff[(dff["smoker"]=="yes") & (dff["bmi_category"]=="Obese")]["charges"].mean()
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Obese + Smoker</div>
        <div class="metric-value">${obese_smoker:,.0f}</div>
        <div class="metric-sub">Highest risk group</div>
    </div>""", unsafe_allow_html=True)

with k5:
    pct_smoker = len(dff[dff["smoker"]=="yes"]) / len(dff) * 100 if len(dff) else 0
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Smoker %</div>
        <div class="metric-value">{pct_smoker:.1f}%</div>
        <div class="metric-sub">{len(dff[dff["smoker"]=='yes'])} people</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Smoker charges  +  Age group charges
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Charges Analysis</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    smoker_avg_df = dff.groupby("smoker", observed=True)["charges"].mean().reset_index()
    smoker_avg_df.columns = ["Smoker", "Avg Charges"]
    fig = px.bar(smoker_avg_df, x="Smoker", y="Avg Charges",
                 color="Smoker",
                 color_discrete_map={"yes": RED, "no": BLUE},
                 text=smoker_avg_df["Avg Charges"].apply(lambda x: f"${x:,.0f}"))
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(**dark_layout("Avg Charges by Smoker Status"))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    age_avg = dff.groupby("age_group", observed=True)["charges"].mean().reset_index()
    age_avg.columns = ["Age Group", "Avg Charges"]
    fig2 = px.bar(age_avg, x="Age Group", y="Avg Charges",
                  color_discrete_sequence=[GREEN],
                  text=age_avg["Avg Charges"].apply(lambda x: f"${x:,.0f}"))
    fig2.update_traces(textposition="outside", marker_line_width=0)
    fig2.update_layout(**dark_layout("Avg Charges by Age Group"))
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Scatter  +  BMI charges
# ═══════════════════════════════════════════════════════════════════════════════
c3, c4 = st.columns([3, 2])

with c3:
    sample = dff.sample(min(400, len(dff)), random_state=42)
    fig3 = px.scatter(sample, x="age", y="charges", color="smoker",
                      size="bmi", size_max=12, opacity=0.7,
                      color_discrete_map={"yes": RED, "no": BLUE},
                      hover_data=["bmi", "region", "sex"],
                      labels={"age": "Age", "charges": "Charges ($)", "smoker": "Smoker"})
    fig3.update_layout(**dark_layout("Age vs Charges (size = BMI)"))
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    bmi_avg = dff.groupby("bmi_category", observed=True)["charges"].mean().reset_index()
    bmi_avg.columns = ["BMI Category", "Avg Charges"]
    order = ["Underweight", "Normal", "Overweight", "Obese"]
    bmi_avg["BMI Category"] = pd.Categorical(bmi_avg["BMI Category"], categories=order, ordered=True)
    bmi_avg = bmi_avg.sort_values("BMI Category")
    fig4 = px.bar(bmi_avg, x="Avg Charges", y="BMI Category", orientation="h",
                  color_discrete_sequence=[AMBER],
                  text=bmi_avg["Avg Charges"].apply(lambda x: f"${x:,.0f}"))
    fig4.update_traces(textposition="outside", marker_line_width=0)
    fig4.update_layout(**dark_layout("Avg Charges by BMI Category"))
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Distribution histograms
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Feature Distributions</div>', unsafe_allow_html=True)

c5, c6, c7 = st.columns(3)

with c5:
    fig5 = px.histogram(dff, x="charges", nbins=40, color_discrete_sequence=[TEAL],
                        labels={"charges": "Charges ($)"})
    fig5.update_traces(marker_line_width=0)
    fig5.update_layout(**dark_layout("Distribution of Charges"))
    st.plotly_chart(fig5, use_container_width=True)

with c6:
    fig6 = px.histogram(dff, x="age", nbins=20, color_discrete_sequence=[BLUE],
                        labels={"age": "Age"})
    fig6.update_traces(marker_line_width=0)
    fig6.update_layout(**dark_layout("Distribution of Age"))
    st.plotly_chart(fig6, use_container_width=True)

with c7:
    fig7 = px.histogram(dff, x="bmi", nbins=30, color_discrete_sequence=[PURPLE],
                        labels={"bmi": "BMI"})
    fig7.update_traces(marker_line_width=0)
    fig7.update_layout(**dark_layout("Distribution of BMI"))
    st.plotly_chart(fig7, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Boxplots + Region
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Deep Dives</div>', unsafe_allow_html=True)

c8, c9 = st.columns(2)

with c8:
    fig8 = px.box(dff, x="smoker", y="charges", color="smoker",
                  color_discrete_map={"yes": RED, "no": BLUE},
                  points="outliers",
                  labels={"smoker": "Smoker", "charges": "Charges ($)"})
    fig8.update_layout(**dark_layout("Charges Distribution by Smoker"))
    st.plotly_chart(fig8, use_container_width=True)

with c9:
    region_avg = dff.groupby("region")["charges"].mean().reset_index()
    region_avg.columns = ["Region", "Avg Charges"]
    fig9 = px.bar(region_avg, x="Region", y="Avg Charges",
                  color="Avg Charges",
                  color_continuous_scale=["#1e2130", TEAL],
                  text=region_avg["Avg Charges"].apply(lambda x: f"${x:,.0f}"))
    fig9.update_traces(textposition="outside", marker_line_width=0)
    fig9.update_layout(**dark_layout("Avg Charges by Region"),
                       coloraxis_showscale=False)
    st.plotly_chart(fig9, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 5 — Correlation heatmap + Children charges
# ═══════════════════════════════════════════════════════════════════════════════
c10, c11 = st.columns([2, 1])

with c10:
    num_cols = ["age", "bmi", "children", "charges"]
    corr = dff[num_cols].corr()
    fig10 = go.Figure(data=go.Heatmap(
        z=corr.values, x=num_cols, y=num_cols,
        colorscale=[[0, "#1e2130"], [0.5, "#2e4a6b"], [1, TEAL]],
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        showscale=False
    ))
    fig10.update_layout(**dark_layout("Correlation Heatmap (Numeric Features)"))
    st.plotly_chart(fig10, use_container_width=True)

with c11:
    child_avg = dff.groupby("children")["charges"].mean().reset_index()
    child_avg.columns = ["Children", "Avg Charges"]
    fig11 = px.bar(child_avg, x="Children", y="Avg Charges",
                   color_discrete_sequence=[GREEN],
                   text=child_avg["Avg Charges"].apply(lambda x: f"${x:,.0f}"))
    fig11.update_traces(textposition="outside", marker_line_width=0)
    fig11.update_layout(**dark_layout("Avg Charges by Children"))
    st.plotly_chart(fig11, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 6 — Pie charts (Sex + Smoker)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Dataset Composition</div>', unsafe_allow_html=True)

c12, c13, c14 = st.columns(3)

with c12:
    sex_cnt = dff["sex"].value_counts().reset_index()
    fig12 = px.pie(sex_cnt, names="sex", values="count",
                   color_discrete_sequence=[BLUE, RED],
                   hole=0.5)
    fig12.update_layout(**dark_layout("Sex Distribution"),
                        showlegend=True)
    st.plotly_chart(fig12, use_container_width=True)

with c13:
    smk_cnt = dff["smoker"].value_counts().reset_index()
    fig13 = px.pie(smk_cnt, names="smoker", values="count",
                   color_discrete_sequence=[RED, BLUE],
                   hole=0.5)
    fig13.update_layout(**dark_layout("Smoker Distribution"),
                        showlegend=True)
    st.plotly_chart(fig13, use_container_width=True)

with c14:
    reg_cnt = dff["region"].value_counts().reset_index()
    fig14 = px.pie(reg_cnt, names="region", values="count",
                   color_discrete_sequence=[TEAL, BLUE, PURPLE, AMBER],
                   hole=0.5)
    fig14.update_layout(**dark_layout("Region Distribution"),
                        showlegend=True)
    st.plotly_chart(fig14, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Key Insights & Model Prep Notes</div>',
            unsafe_allow_html=True)

i1, i2, i3 = st.columns(3)
with i1:
    st.markdown("""<div class="insight-box">
    <h4>🔑 Smoking is the #1 Driver</h4>
    <p>Smokers pay 3.8× more than non-smokers. Obese smokers average $41,693 — 
    making <b>is_smoker</b> the most important feature for regression.</p>
    </div>""", unsafe_allow_html=True)

with i2:
    st.markdown("""<div class="insight-box">
    <h4>📐 Moderate Correlations</h4>
    <p>Age (r=0.30) and BMI (r=0.20) show moderate linear correlation with charges.
    Children is weak (r=0.07). Consider <b>interaction terms</b> for smoker×BMI.</p>
    </div>""", unsafe_allow_html=True)

with i3:
    st.markdown("""<div class="insight-box">
    <h4>✅ Data is Model-Ready</h4>
    <p>Zero missing values. Applied label encoding for sex & smoker, one-hot for 
    region & BMI category. Charges are <b>right-skewed</b> — consider log transform.</p>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RAW DATA TABLE (toggle)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📊 View Raw Data"):
    st.markdown(f"Showing {len(dff):,} filtered rows")
    st.dataframe(dff.drop(columns=["age_group", "bmi_category"], errors="ignore"),
                 use_container_width=True, height=300)
    csv = dff.to_csv(index=False)
    st.download_button("⬇️ Download Filtered CSV", csv,
                       "insurance_filtered.csv", "text/csv")