"""Streamlit demo UI. Run: streamlit run ui/app.py"""

import streamlit as st

from medtrace.pipeline import MedTrace

st.set_page_config(page_title="MedTrace", page_icon=None, layout="wide")
st.title("MedTrace")
st.caption(
    "Answers only from cited sources, and says so when the evidence is weak. "
    "For research and education, not medical advice."
)


@st.cache_resource
def engine() -> MedTrace:
    return MedTrace.from_settings()


question = st.text_input("Ask a question about breast cancer", placeholder="What does HER2-positive mean?")
strict = st.toggle("Strict mode (drop unsupported sentences)", value=True)

if question:
    ans = engine().ask(question, strict=strict)
    if ans.abstained:
        st.warning(f"{ans.answer}  \nReason: `{ans.reason}`")
    else:
        st.markdown(ans.answer)
        st.metric("Unsupported sentences in draft", f"{ans.unsupported_ratio:.0%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Citations")
        for c in ans.citations:
            loc = c.source + (f", p.{c.page}" if c.page else "")
            with st.expander(f"[{c.marker}] {loc}"):
                st.write(c.snippet + "...")
    with col2:
        st.subheader("Claim checks")
        for chk in ans.checks:
            icon = "Supported" if chk.supported else "Unsupported"
            st.write(f"**{icon}** ({chk.support_score:.2f}, KG: {chk.kg_status}) - {chk.sentence}")

    with st.expander("Retrieved chunks"):
        for r in ans.retrieved:
            st.write(
                f"relevance {r.relevance:.2f} | bm25 rank {r.bm25_rank} | dense rank {r.dense_rank} | {r.chunk.source}"
            )
            st.caption(r.chunk.text[:300] + "...")
