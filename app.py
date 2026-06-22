"""
GNN Drug-Target Interaction Predictor & Repurposing Engine
Interactive demo — built on top of the trained models documented in this repo's README.
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import re
from pathlib import Path

st.set_page_config(
    page_title="DTI Predictor — Binding Affinity & Repurposing",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling — overriding Streamlit defaults entirely
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #FAFAF7;
    --bg-panel: #FFFFFF;
    --ink: #1A2421;
    --ink-soft: #4A5550;
    --accent: #2D5F4F;
    --accent-soft: #E8F0EC;
    --signal: #C4634A;
    --signal-soft: #FBEEEA;
    --border: #DDD9CE;
    --mono: 'IBM Plex Mono', monospace;
    --display: 'Space Grotesk', sans-serif;
    --body: 'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--body);
    color: var(--ink);
}

.stApp {
    background: var(--bg);
}

/* Hide default streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }

/* Headline system */
h1, h2, h3 {
    font-family: var(--display) !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}

h1 { font-weight: 700 !important; }
h2 { font-weight: 600 !important; }

.eyebrow {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.hero-title {
    font-family: var(--display);
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.08;
    color: var(--ink);
    margin: 0;
}

.hero-sub {
    font-family: var(--body);
    font-size: 1.05rem;
    color: var(--ink-soft);
    max-width: 640px;
    margin-top: 0.9rem;
    line-height: 1.55;
}

/* Bond-line divider — the signature element */
.bond-divider {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1.6rem 0 1.8rem 0;
    height: 14px;
}
.bond-divider svg { display: block; }

/* Metric cards */
.metric-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 1.1rem 1.3rem;
    height: 100%;
}
.metric-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: var(--mono);
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1;
}
.metric-context {
    font-family: var(--body);
    font-size: 0.78rem;
    color: var(--ink-soft);
    margin-top: 0.4rem;
}
.metric-winner {
    border-left: 3px solid var(--accent);
}
.metric-secondary {
    border-left: 3px solid var(--border);
}

/* Panels */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 1.4rem 1.6rem;
}

.section-label {
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--ink-soft);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin-bottom: 1rem;
}

/* Result readout */
.readout {
    font-family: var(--mono);
    font-size: 2.4rem;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
}
.readout-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-soft);
}
.readout-unit {
    font-family: var(--body);
    font-size: 0.85rem;
    color: var(--ink-soft);
}

.badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border-radius: 2px;
    font-weight: 600;
}
.badge-binder {
    background: var(--signal-soft);
    color: var(--signal);
}
.badge-weak {
    background: var(--accent-soft);
    color: var(--accent);
}

.caveat-box {
    background: var(--accent-soft);
    border-left: 3px solid var(--accent);
    padding: 0.9rem 1.1rem;
    border-radius: 2px;
    font-size: 0.85rem;
    color: var(--ink-soft);
    line-height: 1.5;
    margin-top: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
}

/* Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] {
    font-family: var(--mono) !important;
    border-radius: 2px !important;
    border-color: var(--border) !important;
}

.stButton button {
    font-family: var(--display) !important;
    font-weight: 600 !important;
    background: var(--ink) !important;
    color: var(--bg) !important;
    border: none !important;
    border-radius: 2px !important;
    letter-spacing: 0.02em;
}
.stButton button:hover {
    background: var(--accent) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    font-family: var(--mono) !important;
}

footer.app-footer {
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--ink-soft);
    letter-spacing: 0.02em;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def bond_divider(width_pct=100):
    """The signature element: a row of bond-line marks, echoing molecular structure."""
    svg = f"""
    <div class="bond-divider">
        <svg width="{width_pct}%" height="14" viewBox="0 0 800 14" preserveAspectRatio="none">
            <line x1="0" y1="7" x2="800" y2="7" stroke="#DDD9CE" stroke-width="1"/>
            {''.join(f'<circle cx="{x}" cy="7" r="2.5" fill="#2D5F4F"/>' for x in range(20, 800, 60))}
        </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chemistry helpers
# ---------------------------------------------------------------------------

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def render_molecule_svg(smiles, width=320, height=240, accent="#2D5F4F"):
    """Render a real, accurate 2D structure drawing for the given SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg


def get_ecfp(smiles, n_bits=1024, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


AA_LIST = sorted(list("ACDEFGHIKLMNPQRSTVWY") + ["X"])


def parse_mutation(protein_name):
    match = re.search(r'\((\D)(\d+)(\D)\)', protein_name)
    if match:
        return int(match.group(2)), match.group(1), match.group(3)
    return None


def mutation_features(protein_name, max_seq_length=2500):
    result = parse_mutation(protein_name)
    has_mutation = [0]
    original_aa_onehot = [0] * len(AA_LIST)
    mutant_aa_onehot = [0] * len(AA_LIST)
    normalized_position = [0.0]
    if result is not None:
        position, original_aa, mutant_aa = result
        has_mutation = [1]
        if original_aa in AA_LIST:
            original_aa_onehot[AA_LIST.index(original_aa)] = 1
        if mutant_aa in AA_LIST:
            mutant_aa_onehot[AA_LIST.index(mutant_aa)] = 1
        normalized_position = [position / max_seq_length]
    return has_mutation + original_aa_onehot + mutant_aa_onehot + normalized_position


def get_aa_composition_with_mutation(sequence, protein_name):
    counts = np.array([sequence.count(aa) for aa in AA_LIST], dtype=float)
    composition = counts / max(len(sequence), 1)
    mut_feats = np.array(mutation_features(protein_name), dtype=float)
    return np.concatenate([composition, mut_feats])


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

# DATA_DIR = Path(__file__).parent / "data" / "GraphDTA" / "data" / "davis"
DATA_DIR = Path(__file__).parent / "app_data"
MODEL_PATH = Path(__file__).parent / "models" / "xgb_model_v2.json"


@st.cache_resource
def load_davis_data():
    with open(DATA_DIR / "ligands_can.txt") as f:
        ligands = json.load(f)
    with open(DATA_DIR / "proteins.txt") as f:
        proteins = json.load(f)
    return ligands, proteins


@st.cache_resource
def load_model():
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_data
def get_oncology_panel(protein_names):
    known_oncology_kinases = [
        'EGFR', 'ABL1', 'BRAF', 'ALK', 'KIT', 'MET', 'RET', 'ROS1',
        'FLT3', 'JAK2', 'PDGFRA', 'PDGFRB', 'FGFR1', 'FGFR2', 'FGFR3',
        'SRC', 'AKT1', 'AKT2', 'MTOR', 'CDK4', 'PIK3CA', 'BTK',
        'JAK1', 'JAK3', 'AURKA', 'AURKB'
    ]
    panel = sorted(set(
        p for name in known_oncology_kinases for p in protein_names if name in p
    ))
    return panel


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="eyebrow">⬡ DTI Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family: var(--display); font-weight:700; font-size:1.15rem; '
        'line-height:1.3; margin-bottom: 1.6rem;">Drug–Target Interaction<br/>Predictor</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["Predict", "Model comparison", "Repurposing screen", "Methodology"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div style="margin-top: 2rem; font-family: var(--mono); font-size: 0.72rem; '
        'color: var(--ink-soft); line-height: 1.6;">'
        'Trained on the Davis kinase<br/>binding-affinity benchmark<br/>'
        '(68 drugs × 442 targets)<br/><br/>'
        '<a href="https://github.com/JP-Bro/gnn-drug-target-interaction" '
        'style="color: var(--accent);">View source →</a></div>',
        unsafe_allow_html=True,
    )

try:
    ligands, proteins = load_davis_data()
    protein_names = list(proteins.keys())
    data_loaded = True
except Exception:
    data_loaded = False
    protein_names = []

try:
    model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False
    model = None


# ---------------------------------------------------------------------------
# PAGE: Predict
# ---------------------------------------------------------------------------

if page == "Predict":
    st.markdown('<div class="eyebrow">Binding affinity predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-title">Score a drug against a kinase target</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">Enter any molecule as a SMILES string and select a kinase target '
        'from the Davis benchmark panel. The model — ECFP molecular fingerprints combined with '
        'explicit mutation-site features, scored by gradient-boosted trees — returns a predicted '
        'binding affinity (pKd).</p>',
        unsafe_allow_html=True,
    )
    bond_divider()

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        smiles_input = st.text_input(
            "SMILES string",
            value="CC(=O)OC1=CC=CC=C1C(=O)O",
            help="Default shown is aspirin. Try dasatinib: "
                 "CC1=C(C=C(C=C1)NC2=NC(=CC(=N2)N3CCN(CC3)CCO)C)NC(=O)C4=CC=C(C=C4)CN5CCN(C)CC5",
        )

        if protein_names:
            target = st.selectbox("Kinase target (Davis panel)", protein_names,
                                    index=protein_names.index("ABL1") if "ABL1" in protein_names else 0)
        else:
            target = None
            st.warning("Davis dataset not found at `data/GraphDTA/`. See README setup instructions.")

        run = st.button("Predict affinity →", use_container_width=True)

        mol_svg = render_molecule_svg(smiles_input)
        if mol_svg:
            st.markdown('<div class="section-label" style="margin-top:1.6rem;">Structure</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="panel">{mol_svg}</div>', unsafe_allow_html=True)
        elif smiles_input:
            st.error("Could not parse this SMILES string. Check the syntax.")

    with col_result:
        st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)

        if run and mol_svg and target and model_loaded:
            ecfp = get_ecfp(smiles_input)
            sequence = proteins[target]
            comp_mut = get_aa_composition_with_mutation(sequence, target)
            features = np.concatenate([ecfp, comp_mut]).reshape(1, -1)
            pred_pkd = float(model.predict(features)[0])

            is_strong = pred_pkd >= 7.0
            badge_class = "badge-binder" if is_strong else "badge-weak"
            badge_text = "Strong binder (pKd ≥ 7)" if is_strong else "Weak / no binding predicted"

            kd_nM = 10 ** (9 - pred_pkd)

            st.markdown(
                f'<div class="panel">'
                f'<div class="readout-label">Predicted pKd</div>'
                f'<div class="readout">{pred_pkd:.2f}</div>'
                f'<div class="readout-unit">≈ {kd_nM:,.0f} nM dissociation constant</div>'
                f'<div style="margin-top:0.9rem;"><span class="badge {badge_class}">{badge_text}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            mutation = parse_mutation(target)
            if mutation:
                pos, orig, mut = mutation
                st.markdown(
                    f'<div class="caveat-box">This target carries a point mutation: '
                    f'<strong>{orig}{pos}{mut}</strong>. The model uses explicit '
                    f'(position, original residue, mutant residue) features to account for this — '
                    f'added specifically because composition-only features could not distinguish '
                    f'mutant variants (see Methodology).</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="caveat-box" style="margin-top:0.8rem;">'
                'This model achieves Pearson R = 0.71 and 66% precision at its top-5% most '
                'confident predictions on held-out test data — useful as a triage signal, '
                'not a validated binding measurement. See Model comparison for full evaluation.'
                '</div>',
                unsafe_allow_html=True,
            )
        elif run:
            st.info("Provide a valid SMILES string and select a target to run a prediction.")
        else:
            st.markdown(
                '<div class="panel" style="color: var(--ink-soft); font-size:0.9rem;">'
                'Enter a molecule and target, then run a prediction to see the readout here.'
                '</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PAGE: Model comparison
# ---------------------------------------------------------------------------

elif page == "Model comparison":
    st.markdown('<div class="eyebrow">Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">GNN vs. classical baseline</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Two architectures were built and rigorously compared: a from-scratch '
        'graph neural network with attention pooling, and an ECFP fingerprint + gradient-boosted '
        'tree baseline. The baseline wins — reported here as the primary finding, not omitted.</p>',
        unsafe_allow_html=True,
    )
    bond_divider()

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            '<div class="metric-card metric-winner">'
            '<div class="metric-label">ECFP + XGBoost — Pearson R</div>'
            '<div class="metric-value">0.714</div>'
            '<div class="metric-context">Best-performing model</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="metric-card metric-secondary">'
            '<div class="metric-label">GNN (5-seed mean) — Pearson R</div>'
            '<div class="metric-value">0.546 <span style="font-size:1rem;">± 0.013</span></div>'
            '<div class="metric-context">After LR-schedule fix (was 0.472 ± 0.079)</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="metric-card metric-secondary">'
            '<div class="metric-label">Concordance Index</div>'
            '<div class="metric-value">0.837 <span style="font-size:1rem;">vs 0.78</span></div>'
            '<div class="metric-context">XGBoost vs. GNN</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 1.6rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Precision at top-ranked candidates</div>', unsafe_allow_html=True)

    pr_data = pd.DataFrame({
        "Top fraction screened": ["5%", "10%", "20%"],
        "XGBoost — precision": [0.660, 0.485, 0.326],
        "XGBoost — recall": [0.396, 0.583, 0.784],
        "GNN — precision": [0.520, 0.385, 0.278],
        "GNN — recall": [0.312, 0.463, 0.669],
    })
    st.dataframe(pr_data, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="caveat-box">'
        '<strong>Why the baseline wins:</strong> ECFP fingerprints are a mature, dense molecular '
        'representation; the GNN had to learn its own representation from ~25K examples on a '
        'CPU-constrained architecture. This is a well-documented pattern on small-to-medium tabular '
        'biological datasets — gradient boosting is frequently competitive with or superior to deep '
        'learning in this regime. Verified consistent across pair-split, drug-holdout, and '
        'protein-holdout evaluation strategies, and across 5 independent random seeds.'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PAGE: Repurposing screen
# ---------------------------------------------------------------------------

elif page == "Repurposing screen":
    st.markdown('<div class="eyebrow">Inference at scale</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">1,497-drug oncology repurposing screen</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Every FDA-approved small-molecule drug with a known structure was '
        'scored against a curated panel of 83 oncology-relevant kinases (including known '
        'resistance-mutation variants). Results below are reported with the same scrutiny used '
        'throughout this project.</p>',
        unsafe_allow_html=True,
    )
    bond_divider()

    known_kinase_hits = pd.DataFrame({
        "Drug": ["Dasatinib", "Dasatinib", "Bosutinib", "Sunitinib"],
        "Target": ["ABL1(H396P)", "ABL1(F317L)", "ABL1(F317L)", "KIT(V559D)"],
        "Predicted pKd": [9.31, 9.27, 9.26, 9.07],
        "Status": ["Already approved for this target", "Already approved for this target",
                    "Already approved for this target", "Already approved for this target"],
    })

    novel_candidates = pd.DataFrame({
        "Drug": ["Josamycin", "Eribulin", "Tacrolimus", "Nevirapine"],
        "Target": ["FLT3(N841I)", "FLT3(D835H)", "FLT3(N841I)", "ABL1(F317L)"],
        "Predicted pKd": [8.66, 8.36, 8.22, 8.18],
        "Status": ["No literature corroboration found", "No literature corroboration found",
                    "No literature corroboration found", "No literature corroboration found"],
    })

    st.markdown('<div class="section-label">Top hits — sanity check (model recovers known biology)</div>', unsafe_allow_html=True)
    st.dataframe(known_kinase_hits, hide_index=True, use_container_width=True)

    st.markdown('<div style="height:1.2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">After excluding known kinase inhibitors</div>', unsafe_allow_html=True)
    st.dataframe(novel_candidates, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="caveat-box">'
        '<strong>Honest conclusion — no validated discovery.</strong> The top-ranked hits across '
        'the full screen were dominated by drugs already approved as kinase inhibitors for those '
        'exact targets (dasatinib → ABL1, sunitinib → KIT) — a useful sanity check confirming the '
        'model learned real chemistry, not a novel finding. The four candidates remaining after '
        'excluding known inhibitors showed no corroborating evidence of kinase activity in a '
        'literature check. Given the baseline\'s own measured false-positive rate (~34% even at '
        'its best precision threshold), these are reported as unvalidated speculation, not leads.'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PAGE: Methodology
# ---------------------------------------------------------------------------

elif page == "Methodology":
    st.markdown('<div class="eyebrow">How this was built</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">Pipeline & evaluation methodology</p>', unsafe_allow_html=True)
    bond_divider()

    st.markdown(
        """
<div class="panel">
<strong>Dataset</strong> — Davis kinase binding-affinity benchmark: 68 drugs × 442 kinase targets,
~30,056 measured pairs, sourced via the GraphDTA repository. 69.6% of labels sit at a single
censoring value (Kd = 10,000 nM); all evaluation reports this subset separately from genuine
measured binders.
<br/><br/>
<strong>Drug encoder</strong> — SMILES → RDKit → molecular graph (atoms as nodes, bonds as edges)
→ 3-layer GCN with learned attention pooling → fixed-size embedding.
<br/><br/>
<strong>Protein encoder</strong> — amino acid sequence → learned embedding → 1D CNN → max pooling
→ fixed-size embedding.
<br/><br/>
<strong>Baseline</strong> — ECFP molecular fingerprints (1024-bit, radius 2) + amino-acid
composition, extended with explicit (position, original residue, mutant residue) features for
point-mutant targets → XGBoost regressor.
<br/><br/>
<strong>Rigor checks performed</strong>
<ul>
<li>5 independent random seeds, reporting mean ± std rather than single-run numbers</li>
<li>Generalization tested via three strategies: standard pair-split, strict drug-level holdout
(14 unseen drugs), strict protein-level holdout (88 unseen kinases) — all consistent</li>
<li>Real precision/recall computed at multiple thresholds, not just correlation metrics</li>
<li>Training instability identified (σ=0.079 across seeds) and resolved via learning-rate
scheduling (σ reduced to 0.013)</li>
<li>Repurposing candidates checked against literature rather than reported at face value</li>
</ul>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="caveat-box" style="margin-top:1.2rem;">'
        '<strong>Known limitations:</strong> CPU-only training constrained GNN architecture size '
        'and epoch budget. No pretrained protein language model (e.g. ESM) was used. Repurposing '
        'candidates were checked via general web search, not a systematic bioactivity database '
        'query. Full details, including null results, are in the project README.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<footer class="app-footer">DTI Predictor — built end-to-end as a learning and evaluation '
    'exercise · See full methodology and source on GitHub</footer>',
    unsafe_allow_html=True,
)
