"""
Optional Streamlit web UI for the AI Customer Support Bot.

Run:  streamlit run app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.config import CLASSIFIER_SAVE_DIR
from src.pipeline import CustomerSupportPipeline, build_pipeline, load_pipeline

st.set_page_config(
    page_title="AI Customer Support Bot",
    page_icon="🤖",
    layout="centered",
)

# ── Load pipeline (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model …")
def get_pipeline() -> CustomerSupportPipeline:
    if CLASSIFIER_SAVE_DIR.exists() and any(CLASSIFIER_SAVE_DIR.iterdir()):
        try:
            return load_pipeline()
        except Exception:
            pass
    return build_pipeline()


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🤖 AI Customer Support Bot")
st.caption("Powered by SentenceTransformers · Rule-based Escalation · Demo Data")

st.markdown("---")

pipeline = get_pipeline()

message = st.text_area(
    "Customer Message",
    placeholder="Type your message here … e.g. 'My refund hasn't arrived!'",
    height=120,
)

if st.button("🔍 Analyse", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Processing …"):
            result = pipeline.handle(message)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Intent", result["intent"].replace("_", " ").title())
            st.metric("Confidence", f"{result['confidence']:.4f}")
        with col2:
            if result["should_escalate"]:
                st.error("⚠️ **ESCALATED**")
            else:
                st.success("✅ **Bot Handles**")
            st.metric("Processing Time", f"{result['processing_time_ms']:.0f} ms")

        if result["should_escalate"]:
            st.warning(f"**Escalation Reason:** {result['escalation_reason']}")
            st.info(f"**Rule Triggered:** {result['triggered_rule']}")

        if result.get("anger_signals"):
            with st.expander("🔥 Anger Signals Detected"):
                for sig in result["anger_signals"]:
                    st.write(f"• {sig}")

        st.markdown("### 💬 Bot Reply")
        st.info(result["reply"])

        with st.expander("🔧 Debug Details"):
            st.json({
                "processed_message": result["processed_message"],
                "intent": result["intent"],
                "confidence": result["confidence"],
                "should_escalate": result["should_escalate"],
                "escalation_reason": result["escalation_reason"],
                "triggered_rule": result["triggered_rule"],
                "anger_signals": result["anger_signals"],
            })

st.markdown("---")

# ── Quick demo buttons ────────────────────────────────────────────────────────
st.markdown("#### 🧪 Quick Demo Messages")
demos = {
    "Billing (Angry)": "This is RIDICULOUS! I was charged twice!!!",
    "Technical Issue": "App keeps crashing every time I open it",
    "Account Access": "I forgot my password and can't login",
    "Feature Request": "Could you please add dark mode?",
    "Positive": "Love your service, thank you so much!",
}

cols = st.columns(len(demos))
for col, (label, demo_msg) in zip(cols, demos.items()):
    with col:
        if st.button(label, use_container_width=True):
            with st.spinner("Processing …"):
                r = pipeline.handle(demo_msg)
            st.markdown(f"**Message:** _{demo_msg}_")
            st.write(f"**Intent:** `{r['intent']}` | **Confidence:** `{r['confidence']:.3f}`")
            esc_str = "✅ Bot handles" if not r["should_escalate"] else f"⚠️ Escalated: {r['escalation_reason']}"
            st.write(f"**Escalation:** {esc_str}")
            st.write(f"**Reply:** {r['reply']}")
