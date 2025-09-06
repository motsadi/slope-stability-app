#!/usr/bin/env python
# coding: utf-8

# In[8]:


# app.py — Slope Stability (Classification) — Streamlit
# Run: streamlit run app.py
# CSV: put "slope data.csv" in the same folder (or upload via sidebar)

import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Slope Stability — Classification", page_icon="⛰️", layout="wide")
st.title("⛰️ Slope Stability — Stable / Failure (ML)")

# -------------------- header normalization helpers --------------------
def _norm(name: str) -> str:
    """Normalize a header: strip, lower, replace Greek, drop units/punct/spaces."""
    s = str(name).strip().lower()
    greek = {
        "γ": "gamma", "𝛾": "gamma", "𝜸": "gamma", "𝝲": "gamma",
        "φ": "phi",   "ϕ": "phi",   "𝜑": "phi",   "𝝋": "phi",
        "β": "beta",  "𝛽": "beta",  "𝜷": "beta",
    }
    for g, rep in greek.items():
        s = s.replace(g, rep)
    # remove unit text & symbols
    s = (s.replace("kN/m3".lower(), "")
           .replace("kn/m3", "")
           .replace("kpa", "")
           .replace("(m)", "")
           .replace("°", "")
           .replace("/", " ")
           .replace("(", " ")
           .replace(")", " ")
           .replace("_", " "))
    s = re.sub(r"\s+", " ", s).strip()
    return s

CANON_KEYS = {
    "gamma_kN_m3": ["gamma", "unit weight", "gamma kn m3", "gamma knm3"],
    "c_kPa": ["c", "cohesion"],
    "phi_deg": ["phi", "friction angle", "phi deg"],
    "beta_deg": ["beta", "slope angle", "beta deg"],
    "H_m": ["h", "height", "h m"],
    "ru": ["ru", "r u", "pore pressure ratio", "pore pressure ratio ru"],
    "status": ["status", "class", "label"]
}

def map_headers_by_name(df: pd.DataFrame) -> pd.DataFrame:
    """Try to rename columns to canonical names using normalized header text."""
    rename = {}
    norm_map = {c: _norm(c) for c in df.columns}
    for canon, options in CANON_KEYS.items():
        for c, n in norm_map.items():
            if n in options:
                rename[c] = canon
                break
    out = df.rename(columns=rename).copy()
    # drop obvious index columns
    for c in list(out.columns):
        if _norm(c) in {"no", "#", "index", "id", "sr"}:
            out = out.drop(columns=[c])
    return out

def map_headers_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    If name-mapping failed but column order matches the sample (8 cols or 7 after dropping 'No'),
    assign by position: [gamma, c, phi, beta, H, ru, status]
    """
    out = df.copy()
    # Drop leading index-like column if present (No)
    first_norm = _norm(out.columns[0])
    if first_norm in {"no", "#", "index", "id", "sr"} and out.shape[1] >= 8:
        out = out.drop(columns=[out.columns[0]])
    if out.shape[1] == 7:
        out.columns = ["gamma_kN_m3", "c_kPa", "phi_deg", "beta_deg", "H_m", "ru", "status"]
    return out

# -------------------- CSV loader (robust) --------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    if path_or_buffer is None:
        return None, "No file provided."

    # Try normal read
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception as e:
        # Fallback encodings
        try:
            df = pd.read_csv(path_or_buffer, encoding="utf-8-sig")
        except Exception:
            try:
                df = pd.read_csv(path_or_buffer, encoding="latin-1")
            except Exception as e2:
                return None, f"Could not read CSV: {e2}"

    original_cols = list(df.columns)

    # First, try name-based mapping
    df1 = map_headers_by_name(df)

    needed = ["gamma_kN_m3","c_kPa","phi_deg","beta_deg","H_m","ru","status"]
    if not all(col in df1.columns for col in needed):
        # Try positional mapping if shapes look like the sample
        df2 = map_headers_by_position(df)
        if all(col in df2.columns for col in needed):
            df_use = df2
            mapped_mode = "positional"
        else:
            # Show what we saw to help debug
            return None, (
                "Missing required columns after mapping.\n"
                f"Expected: {needed}\n"
                f"Got: {list(df1.columns)} from headers {original_cols}"
            )
    else:
        df_use = df1
        mapped_mode = "header-names"

    # Coerce numeric types
    for c in ["gamma_kN_m3","c_kPa","phi_deg","beta_deg","H_m","ru"]:
        df_use[c] = pd.to_numeric(df_use[c], errors="coerce")

    # Clean target labels
    status_map = {
        "stable": "stable",
        "failure": "failure",
        "failed": "failure",
        "unstable": "failure",
        "stable ": "stable",
        " failure": "failure",
    }
    df_use["status"] = (
        df_use["status"].astype(str).str.strip().str.lower().map(status_map)
    )

    df_use = df_use.dropna(subset=["gamma_kN_m3","c_kPa","phi_deg","beta_deg","H_m","ru","status"]).copy()
    if df_use.empty:
        return None, "After cleaning, no rows remain. Check numeric cells and the Status column."

    # Binary target: 1 = Stable, 0 = Failure
    df_use["y"] = (df_use["status"] == "stable").astype(int)

    # Final sanity: need both classes for stratified split
    if df_use["y"].nunique() < 2:
        return None, "Dataset contains only one class (all Stable or all Failure). Add more varied rows."

    return df_use, f"Loaded OK via {mapped_mode} mapping."

# -------------------- Sidebar: data --------------------
with st.sidebar:
    st.header("📦 Data")
    choice = st.radio("Source", ["Upload CSV", "Read local file (slope data.csv)"])
    if choice == "Upload CSV":
        up = st.file_uploader("Choose your slope CSV", type=["csv"])
        data, msg = load_csv(up)
    else:
        data, msg = load_csv("slope data.csv")

st.sidebar.caption(msg if msg else "")

if data is None or len(data) < 20:
    st.info("Load your CSV to continue. Expect columns: γ (kN/m3), c (kPa), φ(°), β(°), H (m), ru, Status.")
    if data is None:
        st.stop()

# -------------------- Train classifiers --------------------
@st.cache_data(show_spinner=False)
def train_models(df):
    X = df[["H_m","beta_deg","c_kPa","phi_deg","gamma_kN_m3","ru"]].values
    y = df["y"].values

    # If only one class would break stratify, guard it (we already checked, but safe)
    strat = y if len(np.unique(y)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    models = {
        "Random Forest": Pipeline([
            ("rf", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced"))
        ]),
        "SVC (RBF)": Pipeline([
            ("sc", StandardScaler()),
            ("svc", SVC(C=5.0, gamma="scale", probability=True, class_weight="balanced", random_state=42))
        ]),
        "Neural Net": Pipeline([
            ("sc", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                                  alpha=1e-3, max_iter=1500, random_state=42))
        ])
    }

    trained, metrics = {}, {}
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        # ROC_AUC needs both classes in yte; guard it
        try:
            auc = roc_auc_score(yte, proba)
        except ValueError:
            auc = float("nan")
        metrics[name] = {
            "Accuracy": accuracy_score(yte, pred),
            "F1": f1_score(yte, pred, zero_division=0),
            "ROC_AUC": auc
        }
        trained[name] = (pipe, (Xte, yte))
    met = pd.DataFrame(metrics).T.sort_values("ROC_AUC", ascending=False, na_position="last")
    return trained, met

with st.spinner("Training models…"):
    MODELS, METRICS = train_models(data)

st.subheader("📈 Test metrics")
st.dataframe(
    METRICS.style.format({"Accuracy":"{:.3f}","F1":"{:.3f}","ROC_AUC":"{:.3f}"}),
    use_container_width=True
)

# -------------------- Controls & Prediction --------------------
st.subheader("🎛️ Slope Parameters")
left, right = st.columns([1.0, 1.2])
with left:
    H   = st.slider("Height H (m)", 1.0, float(max(50.0, data["H_m"].max())), float(np.median(data["H_m"])), 0.5)
    beta= st.slider("Slope angle β (°)", 5.0, 80.0, float(np.median(data["beta_deg"])), 0.5)
    c_k = st.slider("Cohesion c (kPa)", 1.0, 200.0, float(np.median(data["c_kPa"])), 0.5)
    phi = st.slider("Friction angle φ (°)", 5.0, 60.0, float(np.median(data["phi_deg"])), 0.5)
    gam = st.slider("Unit weight γ (kN/m³)", 14.0, 28.0, float(np.median(data["gamma_kN_m3"])), 0.1)
    ru  = st.slider("Pore pressure ratio rᵤ (–)", 0.0, 1.0, float(np.median(data["ru"])), 0.01)

Xstar = np.array([[H, beta, c_k, phi, gam, ru]])

def lbl(p):
    return "🟢 Stable" if p >= 0.5 else "🔴 Failure"

with right:
    st.markdown("### 🔮 Predicted stability")
    cols = st.columns(len(MODELS))
    for i, (name, (model, _)) in enumerate(MODELS.items()):
        p = float(model.predict_proba(Xstar)[0, 1])  # prob(Stable)
        with cols[i]:
            st.metric(name, lbl(p), f"P(Stable) = {p:.2f}")

# -------------------- Confusion matrix (best model) --------------------
best_name = METRICS.index[0]
best_m, (Xte, yte) = MODELS[best_name]
proba = best_m.predict_proba(Xte)[:, 1]
pred  = (proba >= 0.5).astype(int)
cm = confusion_matrix(yte, pred)

st.markdown(f"#### Confusion matrix — {best_name}")
st.write(pd.DataFrame(cm, index=["Actual Failure","Actual Stable"], columns=["Pred Failure","Pred Stable"]))

# -------------------- Slope sketch (clean & dynamic) --------------------
st.subheader("🖊️ Slope sketch")

# Bench length only affects the drawing (your dataset doesn't include it)
default_B = float(max(1.0, 0.4 * H))  # ~40% of H by default
B = st.slider("Bench length B (m) — sketch only", 0.0, max(2.0*H, 30.0), default_B, 0.5)

def draw_slope(H, beta_deg, B):
    """
    Geometry:
        (0,0) ───── base origin
        (0,H)       vertical cut (y-axis)
        (B,H)       horizontal bench length B
        (B + H/tanβ, 0)  toe at ground line
    """
    beta = math.radians(beta_deg)
    run = H / max(math.tan(beta), 1e-3)           # horizontal run of the sloping face
    toe_x = B + run

    # polygon (counter-clockwise): base → vertical → bench → slope → back to base
    x = [0.0, 0.0, B, toe_x, 0.0]
    y = [0.0, H,   H,  0.0,   0.0]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    # Fill & outline
    ax.fill(x, y, color="#d6d7db", zorder=1)
    ax.plot(x, y, "k-", linewidth=2, zorder=2)

    # Extend ground line a bit beyond the toe to the right
    ax.plot([0, toe_x*1.06], [0, 0], "k-", linewidth=1)

    # Axes styling (minimalist like the screenshot)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.08*toe_x, toe_x*1.10)
    ax.set_ylim(-0.06*H, H*1.20)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Y "axis" line at x≈0 (like the screenshot)
    ax.plot([0, 0], [0, H*1.05], color="black", linewidth=1.5)

    # Height double-arrow + label (blue)
    ax.annotate("", xy=(-0.05*toe_x, H), xytext=(-0.05*toe_x, 0),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=2))
    ax.text(-0.07*toe_x, H/2, f"H = {H:.1f} m",
            color="#1f77b4", rotation=90, va="center", ha="right", fontsize=10)

    # Angle arc at the toe
    # Draw a small arc with center at the toe to represent β
    arc_r = 0.10 * min(H, run)  # radius
    arc = Arc((toe_x, 0), width=2*arc_r, height=2*arc_r,
              angle=0, theta1=180-beta_deg, theta2=180, color="black", lw=1.5)
    ax.add_patch(arc)
    # Angle label slightly above the arc
    ax.text(toe_x - arc_r*0.80, arc_r*0.50, f"β = {beta_deg:.1f}°", fontsize=10)

    # Bench dimension hint (subtle)
    if B > 0:
        ax.annotate("", xy=(0, H*1.04), xytext=(B, H*1.04),
                    arrowprops=dict(arrowstyle="<->", color="#555", lw=1))
        ax.text(B/2, H*1.06, f"B = {B:.1f} m", color="#555", ha="center", va="bottom", fontsize=9)

    # Light grid baseline for context (very faint)
    ax.grid(False)
    st.pyplot(fig)

draw_slope(H, beta, B)


# -------------------- Data preview --------------------
st.subheader("🧾 Data (first 20)")
st.dataframe(data[["gamma_kN_m3","c_kPa","phi_deg","beta_deg","H_m","ru","status"]].head(20))

st.caption("Notes: ru is a pore-pressure ratio used in slice methods; higher ru generally reduces stability. "
           "Strength uses Mohr–Coulomb (c, φ).")


# In[ ]:




