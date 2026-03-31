"""QuantScope — entry point. Defines navigation and launches the active page."""

import streamlit as st

pg = st.navigation(
    [
        st.Page("Portfolio_Hub.py",                  title="Portfolio Hub",          icon="🏗"),
        st.Page("pages/1_Strategy_Lab.py",           title="Strategy Lab",           icon="⚡"),
        st.Page("pages/2_Strategy_Race.py",          title="Strategy Race",          icon="🏁"),
        st.Page("pages/4_Risk_Analytics.py",         title="Risk Analytics",         icon="📊"),
        st.Page("pages/5_Portfolio_Construction.py", title="Portfolio Construction", icon="📐"),
        st.Page("pages/6_Macro_Dashboard.py",        title="Macro Dashboard",        icon="🌐"),
        st.Page("pages/7_ML_Lab.py",                 title="ML Lab",                 icon="🤖"),
    ],
)
pg.run()
