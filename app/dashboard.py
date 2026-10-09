"""Streamlit dashboard for the credit-risk model.

Run locally::

    streamlit run app/dashboard.py

The dashboard mirrors the exact inference path used by the API: it feeds raw
applicant features through the persisted :class:`CreditRiskModel` and reports the
calibrated probability, the risk tier, the decision, and an ablation-based local
explanation. It is a decision-support surface, not an automated decision-maker.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from credit_risk.config import get_config
from credit_risk.models.predict import load_model

st.set_page_config(page_title="Credit Risk Scoring", page_icon=None, layout="wide")

DISCLAIMER = (
    "Decision support only. This tool estimates a statistical probability of default "
    "from historical data and does not establish causation. Every adverse credit "
    "decision requires qualified human review and must comply with applicable fair-lending law."
)


@st.cache_resource(show_spinner=False)
def _load_model():
    return load_model()


def _load_importance() -> pd.DataFrame | None:
    path = get_config().reports_dir / "feature_importance.csv"
    if path.exists():
        frame = pd.read_csv(path)
        for column in ("feature", "importance", "label"):
            if column not in frame.columns:
                return None
        return frame
    return None


def main() -> None:
    st.title("Credit Risk Scoring")
    st.caption("Probability-of-default model for credit-card clients — calibrated and auditable.")

    try:
        model = _load_model()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python -m credit_risk.models.train` and reload."
        )
        return

    meta = model.metadata
    st.info(
        f"Model **{meta.get('model_name', 'n/a')}** | version `{meta.get('model_version', 'n/a')}` | "
        f"calibration **{meta.get('calibration', {}).get('method', 'n/a')}** | "
        f"operating threshold **{model.approve_threshold:.2f}**"
    )

    with st.sidebar:
        st.header("Applicant details")
        limit_bal = st.number_input("Credit limit (NT$)", min_value=0.0, value=120000.0, step=10000.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=33)
        education = st.selectbox(
            "Education", options=[1, 2, 3, 4],
            format_func=lambda v: {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}[v],
        )
        st.subheader("Repayment status (-2 paid duly … 9 = severe delay)")
        pay = {}
        for month in ("pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"):
            pay[month] = st.slider(month, min_value=-2, max_value=9, value=-1)
        st.subheader("Statement balances (NT$)")
        bill = {f"bill_amt{i}": st.number_input(f"Month {i} balance", min_value=-200000.0, value=5000.0, step=500.0)
                for i in range(1, 7)}
        st.subheader("Amounts repaid (NT$)")
        paid = {f"pay_amt{i}": st.number_input(f"Month {i} repaid", min_value=0.0, value=2000.0, step=100.0)
                for i in range(1, 7)}

    record = {"limit_bal": limit_bal, "age": int(age), "education": int(education), **pay, **bill, **paid}

    if st.button("Score applicant", type="primary"):
        result = model.score_record(record)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Default probability", f"{result['default_probability']:.1%}")
        col2.metric("Risk tier", result["risk_tier"])
        col3.metric("Decision", result["decision"])
        col4.metric("Risk score", f"{result['risk_score']}/1000")

        st.subheader("Key factors (ablation vs. portfolio median)")
        factors = model.explain_record(record)
        left, right = st.columns(2)
        with left:
            st.markdown("**Risk-increasing**")
            if factors["risk_increasing"]:
                st.dataframe(pd.DataFrame(factors["risk_increasing"]), hide_index=True, use_container_width=True)
            else:
                st.write("None above the reference.")
        with right:
            st.markdown("**Protective**")
            if factors["protective"]:
                st.dataframe(pd.DataFrame(factors["protective"]), hide_index=True, use_container_width=True)
            else:
                st.write("None below the reference.")

        st.caption(
            "Contributions are the change in the model's probability when each feature is reset to the "
            "portfolio median. They explain the model's behaviour for this applicant and are not causal."
        )
    else:
        st.write("Enter applicant details in the sidebar and click **Score applicant**.")

    st.divider()
    with st.expander("Model metadata"):
        st.json({
            "model_version": meta.get("model_version"),
            "trained_at": meta.get("trained_at"),
            "git_commit": meta.get("git_commit"),
            "dataset_fingerprint": meta.get("dataset", {}).get("fingerprint"),
            "operating_threshold": model.approve_threshold,
            "decline_threshold": model.decline_threshold,
            "test_metrics": meta.get("test_metrics"),
        })
    importance = _load_importance()
    if importance is not None:
        with st.expander("Global feature importance"):
            top = importance.sort_values("importance", ascending=False).head(20)
            st.bar_chart(top.set_index("label")["importance"])

    st.warning(DISCLAIMER)


if __name__ == "__main__":
    main()
