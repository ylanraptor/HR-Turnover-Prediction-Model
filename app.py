import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

# --- Load model and features ---
@st.cache_resource
def load_model():
    pipeline = joblib.load('turnover_rf_pipeline.joblib')
    with open('features.json') as f:
        features = json.load(f)
    return pipeline, features

pipeline, FEATURES = load_model()


DEFAULT_FORM_STATE = {
    "gender": 0,
    "married": 0,
    "perf_score": 3,
    "diversity": 0,
    "dept_id": 5,
    "position_id": 19,
    "engagement": 3.5,
    "satisfaction": 3,
    "special_projects": 0,
    "days_late": 0,
    "absences": 5,
    "tenure_years": 5.0,
    "age_mid": 35.0,
    "salary_ratio": 1.0,
    "tenure_ratio": 1.0,
    "manager_turnover": 0.2,
}

for key, value in DEFAULT_FORM_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "sync_lock" not in st.session_state:
    st.session_state.sync_lock = False

if "csv_sync_error" not in st.session_state:
    st.session_state.csv_sync_error = ""

if "last_action" not in st.session_state:
    st.session_state.last_action = None


# ---------------------------------------------------------------------------
# Parameter categories
# ---------------------------------------------------------------------------

# Parameters the company CAN actively influence
# Both `best_val` and `description` / `action_label` are callables: (current_value) -> value
ACTIONABLE_CONFIG = {
    "satisfaction": {
        "label": "Employee Satisfaction",
        "best_val": lambda v: 5,
        "description": lambda v: (
            "Critical situation: overhaul work conditions, team culture, and management relationship"
            if int(v) <= 2 else
            "Enhance recognition programmes, flexibility, and team dynamics"
            if int(v) == 3 else
            "Fine-tune benefits and growth opportunities to reach top satisfaction"
        ),
        "action_label": lambda v: (
            "Critical overhaul" if int(v) <= 2 else
            "Targeted improvement" if int(v) == 3 else
            "Recognition boost"
        ),
        "fmt": lambda v: str(int(v)),
    },
    "engagement": {
        "label": "Engagement Survey Score",
        "best_val": lambda v: 5.0,
        "description": lambda v: (
            "Urgent: launch a 1-on-1 re-engagement plan, career path discussion, and meaningful work assignment"
            if float(v) <= 2.5 else
            "Team culture programme, regular feedback cycles, and employee involvement in decisions"
            if float(v) <= 3.5 else
            "Stretch assignments and mentoring to sustain high engagement"
        ),
        "action_label": lambda v: (
            "Urgent re-engagement" if float(v) <= 2.5 else
            "Culture initiative" if float(v) <= 3.5 else
            "Incremental boost"
        ),
        "fmt": lambda v: f"{v:.1f}",
    },
    "salary_ratio": {
        "label": "Salary vs. Dept. Average",
        "best_val": lambda v: (
            round(max(float(v) + 0.30, 1.0), 2) if float(v) < 0.80 else
            round(min(float(v) + 0.20, 2.0), 2) if float(v) < 0.95 else
            round(min(float(v) + 0.10, 2.0), 2)
        ),
        "description": lambda v: (
            "Severely underpaid: significant raise to at least department average (ratio ≥ 1.0)"
            if float(v) < 0.80 else
            "Below competitive range: raise compensation to align with market peers"
            if float(v) < 0.95 else
            "Merit raise to reward contribution and signal retention intent"
        ),
        "action_label": lambda v: (
            "Market-parity raise" if float(v) < 0.80 else
            "Competitive raise" if float(v) < 0.95 else
            "Merit raise"
        ),
        "fmt": lambda v: f"{float(v):.2f}",
    },
    "special_projects": {
        "label": "Special Projects Count",
        "best_val": lambda v: 5 if int(v) == 0 else min(int(v) + 3, 10),
        "description": lambda v: (
            "No projects assigned at all: launch a major visibility initiative — assign 5 projects immediately"
            if int(v) == 0 else
            "Expand the project portfolio to strengthen engagement and career growth"
        ),
        "action_label": lambda v: (
            "Major initiative" if int(v) == 0 else
            "Portfolio expansion"
        ),
        "fmt": lambda v: str(int(v)),
    },
    "manager_turnover": {
        "label": "Manager Turnover Rate",
        "best_val": lambda v: (
            0.0 if float(v) > 0.5 else
            round(max(float(v) - 0.25, 0.0), 2) if float(v) > 0.3 else
            round(max(float(v) - 0.15, 0.0), 2)
        ),
        "description": lambda v: (
            "Critical instability: replace the manager to provide stable, long-term leadership"
            if float(v) > 0.5 else
            "High turnover rate: launch a targeted management retention and intervention programme"
            if float(v) > 0.3 else
            "Provide coaching and support to stabilise the management chain"
        ),
        "action_label": lambda v: (
            "Replace manager" if float(v) > 0.5 else
            "Management intervention" if float(v) > 0.3 else
            "Coaching & support"
        ),
        "fmt": lambda v: f"{float(v):.2f}",
    },
    "perf_score": {
        "label": "Performance Score",
        "best_val": lambda v: min(int(v) + 2, 4) if int(v) == 1 else min(int(v) + 1, 4),
        "description": lambda v: (
            "Critical underperformance: enrol in an intensive development programme with dedicated mentoring"
            if int(v) == 1 else
            "Provide structured coaching, clear goals, and regular performance reviews"
            if int(v) == 2 else
            "Assign stretch goals and leadership opportunities to unlock the next performance level"
        ),
        "action_label": lambda v: (
            "Intensive programme" if int(v) == 1 else
            "Structured coaching" if int(v) == 2 else
            "Stretch assignments"
        ),
        "fmt": lambda v: str(int(v)),
    },
}

# Parameters the company CANNOT directly change
NON_ACTIONABLE_CONFIG = {
    "gender":       {"label": "Gender",                   "fmt": lambda v: "Female" if int(v) == 0 else "Male"},
    "married":      {"label": "Marital Status",           "fmt": lambda v: "No" if int(v) == 0 else "Yes"},
    "age_mid":      {"label": "Age",                      "fmt": lambda v: f"{float(v):.0f}"},
    "diversity":    {"label": "Hired via Diversity Fair", "fmt": lambda v: "No" if int(v) == 0 else "Yes"},
    "dept_id":      {"label": "Department ID",            "fmt": lambda v: str(int(v))},
    "position_id":  {"label": "Position ID",              "fmt": lambda v: str(int(v))},
    "tenure_years": {"label": "Tenure (years)",           "fmt": lambda v: f"{float(v):.1f}"},
    "days_late":    {"label": "Days Late (last 30)",      "fmt": lambda v: str(int(v))},
    "absences":     {"label": "Total Absences",           "fmt": lambda v: str(int(v))},
    "tenure_ratio": {"label": "Tenure vs. Dept. Average","fmt": lambda v: f"{float(v):.2f}"},
}


# ---------------------------------------------------------------------------
# Derived-feature helpers
# ---------------------------------------------------------------------------

def _compute_derived_from_state():
    absences_per_year = st.session_state.absences / (st.session_state.tenure_years + 0.5)
    low_sat_high_abs = int(st.session_state.satisfaction <= 2 and st.session_state.absences > 10)
    risk_score = (
        (5 - st.session_state.engagement)
        + (5 - st.session_state.satisfaction)
        + (min(st.session_state.days_late, 10) / 2)
        + (st.session_state.absences / 10)
        + (2.0 if st.session_state.perf_score <= 2 else 0.0)
        + (0.5 if st.session_state.special_projects == 0 else 0.0)
    )
    risk_score = max(risk_score, 0.0)
    return risk_score, absences_per_year, low_sat_high_abs


def _compute_derived(state: dict):
    absences_per_year = state["absences"] / (state["tenure_years"] + 0.5)
    low_sat_high_abs = int(state["satisfaction"] <= 2 and state["absences"] > 10)
    risk_score = (
        (5 - state["engagement"])
        + (5 - state["satisfaction"])
        + (min(state["days_late"], 10) / 2)
        + (state["absences"] / 10)
        + (2.0 if state["perf_score"] <= 2 else 0.0)
        + (0.5 if state["special_projects"] == 0 else 0.0)
    )
    risk_score = max(risk_score, 0.0)
    return risk_score, absences_per_year, low_sat_high_abs


def _build_feature_values_from_state():
    risk_score, absences_per_year, low_sat_high_abs = _compute_derived_from_state()
    return [
        float(st.session_state.gender),
        float(st.session_state.married),
        float(st.session_state.perf_score),
        float(st.session_state.diversity),
        float(st.session_state.dept_id),
        float(st.session_state.position_id),
        float(st.session_state.engagement),
        float(st.session_state.satisfaction),
        float(st.session_state.special_projects),
        float(st.session_state.days_late),
        float(st.session_state.absences),
        float(st.session_state.tenure_years),
        float(st.session_state.age_mid),
        float(st.session_state.salary_ratio),
        float(risk_score),
        float(st.session_state.tenure_ratio),
        float(st.session_state.manager_turnover),
        float(absences_per_year),
        float(low_sat_high_abs),
    ]


def _build_row_from_params(overrides: dict) -> pd.DataFrame:
    """Build a prediction DataFrame row, applying overrides on top of current session state."""
    state = {key: st.session_state[key] for key in DEFAULT_FORM_STATE}
    state.update(overrides)

    risk_score, absences_per_year, low_sat_high_abs = _compute_derived(state)

    return pd.DataFrame([[
        float(state["gender"]),
        float(state["married"]),
        float(state["perf_score"]),
        float(state["diversity"]),
        float(state["dept_id"]),
        float(state["position_id"]),
        float(state["engagement"]),
        float(state["satisfaction"]),
        float(state["special_projects"]),
        float(state["days_late"]),
        float(state["absences"]),
        float(state["tenure_years"]),
        float(state["age_mid"]),
        float(state["salary_ratio"]),
        float(risk_score),
        float(state["tenure_ratio"]),
        float(state["manager_turnover"]),
        float(absences_per_year),
        float(low_sat_high_abs),
    ]], columns=FEATURES)


# ---------------------------------------------------------------------------
# CSV sync helpers
# ---------------------------------------------------------------------------

def _format_csv_value(v):
    return f"{float(v):.6f}".rstrip("0").rstrip(".")


def _sync_csv_from_form():
    if st.session_state.sync_lock:
        return
    st.session_state.sync_lock = True
    try:
        values = _build_feature_values_from_state()
        st.session_state.csv_input = ",".join(_format_csv_value(v) for v in values)
        st.session_state.csv_sync_error = ""
    finally:
        st.session_state.sync_lock = False


def _enforce_age_tenure():
    """Clamp tenure so that age - tenure >= 18 (can't work before age 18)."""
    max_tenure = max(0.0, float(st.session_state.age_mid) - 18.0)
    if float(st.session_state.tenure_years) > max_tenure:
        st.session_state.tenure_years = round(max_tenure * 2) / 2  # round to nearest 0.5
    _sync_csv_from_form()


def _sync_form_from_csv():
    if st.session_state.sync_lock:
        return

    raw = st.session_state.csv_input.strip()
    if not raw:
        st.session_state.csv_sync_error = ""
        return

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != len(FEATURES):
        st.session_state.csv_sync_error = (
            f"CSV sync: expected {len(FEATURES)} values, got {len(parts)}."
        )
        return

    try:
        vals = list(map(float, parts))
    except ValueError:
        st.session_state.csv_sync_error = "CSV sync: values must be numeric."
        return

    st.session_state.sync_lock = True
    try:
        st.session_state.gender = int(vals[0])
        st.session_state.married = int(vals[1])
        st.session_state.perf_score = int(vals[2])
        st.session_state.diversity = int(vals[3])
        st.session_state.dept_id = int(vals[4])
        st.session_state.position_id = int(vals[5])
        st.session_state.engagement = float(vals[6])
        st.session_state.satisfaction = int(vals[7])
        st.session_state.special_projects = int(vals[8])
        st.session_state.days_late = int(vals[9])
        st.session_state.absences = int(vals[10])
        st.session_state.tenure_years = float(vals[11])
        st.session_state.age_mid = float(vals[12])
        st.session_state.salary_ratio = float(vals[13])
        st.session_state.tenure_ratio = float(vals[15])
        st.session_state.manager_turnover = float(vals[16])
        st.session_state.csv_sync_error = ""
    finally:
        st.session_state.sync_lock = False


if "csv_input" not in st.session_state:
    st.session_state.csv_input = ",".join(
        _format_csv_value(v) for v in _build_feature_values_from_state()
    )


# ---------------------------------------------------------------------------
# Prediction display
# ---------------------------------------------------------------------------

def display_prediction(row):
    """Run the model and display results."""
    proba = float(pipeline.predict_proba(row)[0][1])
    proba = max(0.0, min(1.0, proba))
    prediction = pipeline.predict(row)[0]

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Turnover probability", f"{proba:.1%}")

    with col2:
        if prediction == 1:
            st.error("Result: AT RISK OF LEAVING")
        else:
            st.success("Result: STABLE")

    st.progress(proba)

    with st.expander("View input details"):
        detail = pd.DataFrame({
            "Feature": FEATURES,
            "Value": row.iloc[0].values,
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

    return proba


# ---------------------------------------------------------------------------
# Advice engine
# ---------------------------------------------------------------------------

def compute_advice():
    """
    For each actionable parameter, compute individual impact on turnover probability
    when set to its best feasible value.

    Only changes that actually reduce risk are returned as recommendations.
    Counter-productive or neutral changes are silently discarded.

    Returns:
        baseline_proba  – current turnover probability
        beneficial      – list of impact dicts (risk-reducing only), sorted desc
        final_proba     – cumulative probability after applying all beneficial changes
    """
    baseline_row = _build_row_from_params({})
    baseline_proba = float(pipeline.predict_proba(baseline_row)[0][1])

    all_impacts = []

    for param, cfg in ACTIONABLE_CONFIG.items():
        current_val = st.session_state[param]
        best_val = cfg["best_val"](current_val)

        # Skip if the recommended value equals the current value
        if isinstance(current_val, int):
            if int(best_val) == int(current_val):
                continue
        else:
            if abs(float(best_val) - float(current_val)) < 1e-6:
                continue

        test_row = _build_row_from_params({param: best_val})
        test_proba = float(pipeline.predict_proba(test_row)[0][1])
        impact = baseline_proba - test_proba  # positive = reduces risk

        all_impacts.append({
            "param": param,
            "label": cfg["label"],
            "action_label": cfg["action_label"](current_val),
            "description": cfg["description"](current_val),
            "old_val": current_val,
            "new_val": best_val,
            "fmt": cfg["fmt"],
            "impact": impact,
            "individual_proba": test_proba,
        })

    # Keep only changes that genuinely reduce risk; discard neutral / counter-productive ones
    beneficial = [item for item in all_impacts if item["impact"] > 0.001]

    # Sort by individual impact, highest first
    beneficial.sort(key=lambda x: x["impact"], reverse=True)

    # Greedy cumulative: apply only the beneficial changes one by one
    combined = {}
    for item in beneficial:
        combined[item["param"]] = item["new_val"]
        row = _build_row_from_params(combined)
        item["cumulative_proba"] = float(pipeline.predict_proba(row)[0][1])

    final_proba = beneficial[-1]["cumulative_proba"] if beneficial else baseline_proba

    return baseline_proba, beneficial, final_proba


def display_advice():
    """Display the parameter categories and retention recommendations."""
    baseline_proba, impacts, final_proba = compute_advice()

    st.divider()
    st.subheader("Retention Advice")

    # ---- Parameter categories ----
    with st.expander("Parameter categories", expanded=True):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**What the company can influence**")
            rows = []
            for param, cfg in ACTIONABLE_CONFIG.items():
                val = st.session_state[param]
                rows.append({
                    "Parameter": cfg["label"],
                    "Current value": cfg["fmt"](val),
                    "Potential action": cfg["action_label"](val),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        with col_b:
            st.markdown("**What the company cannot directly change**")
            rows = []
            for param, cfg in NON_ACTIONABLE_CONFIG.items():
                val = st.session_state[param]
                rows.append({
                    "Parameter": cfg["label"],
                    "Current value": cfg["fmt"](val),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.caption(
            "Derived features (risk score, absences per year, low-satisfaction flag) "
            "are computed automatically and will improve as the actionable parameters change."
        )

    # ---- Individual recommendations (beneficial only) ----
    if not impacts:
        st.info(
            "None of the actionable parameters produce a risk reduction for this employee "
            "profile. The main drivers of their turnover risk appear to be fixed factors "
            "(demographics, tenure, absence history). Consider a direct conversation to "
            "understand their specific concerns."
        )
        return

    # ---- Summary metrics (only when there is at least one beneficial change) ----
    reduction = baseline_proba - final_proba
    st.markdown("**Projected impact of all recommended changes combined:**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current risk", f"{baseline_proba:.1%}")
    c2.metric(
        "Projected risk",
        f"{final_proba:.1%}",
        delta=f"{final_proba - baseline_proba:+.1%}",
        delta_color="inverse",
    )
    c3.metric("Total reduction", f"{reduction:.1%}")

    st.progress(min(final_proba, 1.0))

    if final_proba < 0.5:
        st.success("With all recommended changes, the employee is projected to be **STABLE**.")
    else:
        st.warning(
            "Even with all recommended changes the employee remains **AT RISK**. "
            "Consider additional retention strategies beyond these levers."
        )

    st.markdown("**Recommended actions** (sorted by individual impact on risk):")

    # Build a summary table — only beneficial rows
    table_rows = []
    for item in impacts:
        old_fmt = item["fmt"](item["old_val"])
        new_fmt = item["fmt"](item["new_val"])
        table_rows.append({
            "Parameter": item["label"],
            "Action": item["action_label"],
            "Current": old_fmt,
            "Recommended": new_fmt,
            "Risk reduction": f"▼ {item['impact']:.1%}",
            "Risk if only this changes": f"{item['individual_proba']:.1%}",
            "Cumulative risk": f"{item['cumulative_proba']:.1%}",
        })

    st.dataframe(
        pd.DataFrame(table_rows),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Action details:**")
    for item in impacts:
        old_fmt = item["fmt"](item["old_val"])
        new_fmt = item["fmt"](item["new_val"])
        st.markdown(
            f"- **{item['label']}** — *{item['action_label']}*: "
            f"`{old_fmt}` → `{new_fmt}` &nbsp; **-{item['impact']:.1%} risk**  \n"
            f"  {item['description']}"
        )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.title("Employee Turnover Prediction")
st.markdown("Predict the risk of an employee leaving using the Random Forest model.")


# ==================== FORM TAB ====================
tab_form, tab_csv = st.tabs(["Form input", "Paste CSV"])

with tab_form:
    st.subheader("Employee information")

    # --- Demographics ---
    st.markdown("**Demographics**")
    dem1, dem2, dem3, dem4 = st.columns(4)
    with dem1:
        st.selectbox(
            "Gender",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            key="gender",
            on_change=_sync_csv_from_form,
        )
    with dem2:
        st.selectbox(
            "Married",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key="married",
            on_change=_sync_csv_from_form,
        )
    with dem3:
        st.number_input(
            "Age",
            min_value=18.0,
            max_value=70.0,
            step=1.0,
            key="age_mid",
            on_change=_enforce_age_tenure,
        )
    with dem4:
        st.selectbox(
            "From diversity job fair",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key="diversity",
            on_change=_sync_csv_from_form,
        )

    # --- Job ---
    st.markdown("**Job details**")
    job1, job2, job3 = st.columns(3)
    with job1:
        st.number_input(
            "Department ID",
            min_value=1,
            max_value=10,
            step=1,
            key="dept_id",
            on_change=_sync_csv_from_form,
        )
    with job2:
        st.number_input(
            "Position ID",
            min_value=1,
            max_value=30,
            step=1,
            key="position_id",
            on_change=_sync_csv_from_form,
        )
    with job3:
        st.number_input(
            "Tenure (years)",
            min_value=0.0,
            max_value=30.0,
            step=0.5,
            key="tenure_years",
            on_change=_enforce_age_tenure,
        )

    max_tenure_allowed = max(0.0, float(st.session_state.age_mid) - 18.0)
    if float(st.session_state.tenure_years) > max_tenure_allowed:
        st.warning(
            f"Tenure cannot exceed age − 18 (max {max_tenure_allowed:.1f} yrs). "
            "Value has been clamped automatically."
        )
    else:
        st.caption(
            f"Max tenure allowed: {max_tenure_allowed:.1f} yrs "
            f"(age {int(st.session_state.age_mid)} − 18)"
        )

    # --- Performance & engagement ---
    st.markdown("**Performance & engagement**")
    perf1, perf2, perf3 = st.columns(3)
    with perf1:
        st.select_slider(
            "Performance score",
            options=[1, 2, 3, 4],
            key="perf_score",
            on_change=_sync_csv_from_form,
        )
    with perf2:
        st.slider(
            "Engagement survey",
            min_value=1.0,
            max_value=5.0,
            step=0.1,
            key="engagement",
            on_change=_sync_csv_from_form,
        )
    with perf3:
        st.select_slider(
            "Satisfaction",
            options=[1, 2, 3, 4, 5],
            key="satisfaction",
            on_change=_sync_csv_from_form,
        )

    # --- Activity ---
    st.markdown("**Activity**")
    act1, act2, act3 = st.columns(3)
    with act1:
        st.number_input(
            "Special projects count",
            min_value=0,
            max_value=10,
            step=1,
            key="special_projects",
            on_change=_sync_csv_from_form,
        )
    with act2:
        st.number_input(
            "Days late (last 30 days)",
            min_value=0,
            max_value=30,
            step=1,
            key="days_late",
            on_change=_sync_csv_from_form,
        )
    with act3:
        st.number_input(
            "Absences (total)",
            min_value=0,
            max_value=50,
            step=1,
            key="absences",
            on_change=_sync_csv_from_form,
        )

    # --- Computed / contextual ---
    st.markdown("**Contextual ratios**")
    ctx1, ctx2, ctx3 = st.columns(3)
    with ctx1:
        st.number_input(
            "Salary / dept. average",
            min_value=0.0,
            max_value=3.0,
            step=0.05,
            key="salary_ratio",
            on_change=_sync_csv_from_form,
        )
    with ctx2:
        st.number_input(
            "Tenure / dept. average",
            min_value=0.0,
            max_value=5.0,
            step=0.05,
            key="tenure_ratio",
            on_change=_sync_csv_from_form,
        )
    with ctx3:
        st.number_input(
            "Manager's turnover rate",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="manager_turnover",
            on_change=_sync_csv_from_form,
        )

    # Auto-computed features
    risk_score, absences_per_year, low_sat_high_abs = _compute_derived_from_state()

    with st.expander("Auto-computed features"):
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Risk score", f"{risk_score:.2f}")
        ac2.metric("Absences / year", f"{absences_per_year:.2f}")
        ac3.metric("Low sat. & high abs.", str(low_sat_high_abs))

    # --- Buttons ---
    btn1, btn2 = st.columns([1, 1])
    with btn1:
        if st.button("Predict", key="form_predict", use_container_width=True):
            st.session_state.last_action = "predict"
    with btn2:
        if st.button("Get Retention Advice", key="form_advice", use_container_width=True, type="secondary"):
            st.session_state.last_action = "advice"

    # --- Results ---
    if st.session_state.last_action in ("predict", "advice"):
        _sync_csv_from_form()
        row = pd.DataFrame([[
            float(st.session_state.gender), float(st.session_state.married),
            float(st.session_state.perf_score), float(st.session_state.diversity),
            float(st.session_state.dept_id), float(st.session_state.position_id),
            float(st.session_state.engagement), float(st.session_state.satisfaction),
            float(st.session_state.special_projects), float(st.session_state.days_late),
            float(st.session_state.absences), float(st.session_state.tenure_years),
            float(st.session_state.age_mid), float(st.session_state.salary_ratio),
            float(risk_score), float(st.session_state.tenure_ratio),
            float(st.session_state.manager_turnover), float(absences_per_year),
            float(low_sat_high_abs),
        ]], columns=FEATURES)

        display_prediction(row)

        if st.session_state.last_action == "advice":
            display_advice()


# ==================== CSV TAB ====================
with tab_csv:
    st.subheader("Paste a CSV line")

    with st.expander("View expected features (in order)"):
        st.code(",".join(FEATURES), language="text")
        st.markdown(
            "| # | Feature | Description |\n"
            "|---|---------|-------------|\n"
            "| 1 | GenderID | 0 = Female, 1 = Male |\n"
            "| 2 | MarriedID | 0 = Not married, 1 = Married |\n"
            "| 3 | PerfScoreID | Performance score (1-4) |\n"
            "| 4 | FromDiversityJobFairID | 0 = No, 1 = Yes |\n"
            "| 5 | DeptID | Department ID |\n"
            "| 6 | PositionID | Position ID |\n"
            "| 7 | EngagementSurvey | Engagement score (1.0-5.0) |\n"
            "| 8 | EmpSatisfaction | Satisfaction (1-5) |\n"
            "| 9 | SpecialProjectsCount | Number of special projects |\n"
            "| 10 | DaysLateLast30 | Days late (last 30 days) |\n"
            "| 11 | Absences | Total number of absences |\n"
            "| 12 | tenure_years | Tenure (years) |\n"
            "| 13 | age_mid | Age (midpoint of bracket) |\n"
            "| 14 | salary_ratio_dept | Salary / department average |\n"
            "| 15 | risk_score | Composite risk score |\n"
            "| 16 | tenure_ratio_dept | Tenure / department average |\n"
            "| 17 | manager_turnover_rate | Manager's turnover rate |\n"
            "| 18 | absences_per_year | Absences per year |\n"
            "| 19 | low_sat_high_abs | 1 if satisfaction <= 2 and absences > 10 |"
        )

    csv_input = st.text_area(
        "Enter a CSV line (19 comma-separated values):",
        key="csv_input",
        on_change=_sync_form_from_csv,
        height=100,
        placeholder="1,0,3,0,5,19,4.6,3,0,0,1,5.48,37.5,0.98,3.0,1.12,0.33,0.17,0",
    )

    if st.session_state.csv_sync_error:
        st.warning(st.session_state.csv_sync_error)

    if st.button("Predict", key="csv_predict"):
        if not csv_input.strip():
            st.error("Please enter a CSV line.")
        else:
            try:
                values = [v.strip() for v in csv_input.strip().split(",")]

                if len(values) != len(FEATURES):
                    st.error(
                        f"Incorrect number of values: {len(values)} "
                        f"(expected: {len(FEATURES)})"
                    )
                else:
                    row = pd.DataFrame(
                        [list(map(float, values))],
                        columns=FEATURES,
                    )
                    display_prediction(row)

            except ValueError as e:
                st.error(f"Format error: all values must be numeric. ({e})")
