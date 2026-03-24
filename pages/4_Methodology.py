"""Page 3: Methodology — mathematical explanations with LaTeX.

Full implementation in Phase 6.
"""

import streamlit as st

st.set_page_config(page_title="Methodology", page_icon="📐", layout="wide")

from ui.theme import inject_css  # noqa: E402

inject_css()
st.markdown(
    "<h2 style='font-size:1.4rem;font-weight:700;letter-spacing:-0.3px;"
    "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;margin-bottom:0.1rem;'>"
    "📐 Methodology</h2>",
    unsafe_allow_html=True,
)
st.info("Strategy mathematical explanations will appear here in Phase 6.")
