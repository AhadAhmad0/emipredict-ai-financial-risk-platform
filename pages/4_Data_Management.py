"""
EMIPredict AI - Data Management page
Session-based CRUD interface for reviewing/managing prediction records made
during this session. Not backed by a persistent database -- records reset
when the Streamlit session ends. This is documented explicitly below rather
than silently implied to be permanent storage.
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Management — EMIPredict AI", page_icon="\U0001F5C2\uFE0F", layout="wide")

st.title("\U0001F5C2\uFE0F Data Management")

st.warning(
    "**Storage note:** records here are held in this browser session only (Streamlit "
    "`session_state`), not a persistent database. Refreshing the page or starting a new "
    "session clears this table. Use **Export to CSV** below before ending your session "
    "if you want to keep a record."
)

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader(f"Session Records ({len(st.session_state.history)})")

if not st.session_state.history:
    st.info("No records yet. Go to the **Predict** page and submit an assessment — it will appear here automatically.")
else:
    history_df = pd.DataFrame(st.session_state.history)
    history_df.insert(0, "Record #", range(1, len(history_df) + 1))

    st.dataframe(history_df, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "\U0001F4E5 Export to CSV", data=csv, file_name="emi_prediction_records.csv", mime="text/csv"
        )
    with col2:
        if st.button("\U0001F5D1\uFE0F Clear All Records", type="secondary"):
            st.session_state.history = []
            st.rerun()
    with col3:
        record_to_delete = st.number_input(
            "Delete record #", min_value=1, max_value=max(len(history_df), 1), value=1, step=1
        )

    if st.button("Delete selected record"):
        idx = record_to_delete - 1
        if 0 <= idx < len(st.session_state.history):
            st.session_state.history.pop(idx)
            st.success(f"Record #{record_to_delete} deleted.")
            st.rerun()

    st.divider()
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        elig_counts = history_df['predicted_eligibility'].value_counts()
        st.metric("Eligible", int(elig_counts.get('Eligible', 0)))
    with col2:
        st.metric("High Risk", int(elig_counts.get('High_Risk', 0)))
    with col3:
        st.metric("Not Eligible", int(elig_counts.get('Not_Eligible', 0)))

st.divider()
st.caption(
    "For a production deployment, this page would connect to a real database "
    "(e.g. PostgreSQL, Firebase) for persistent, multi-user record storage. "
    "This session-based version is scoped for demo/portfolio purposes."
)
