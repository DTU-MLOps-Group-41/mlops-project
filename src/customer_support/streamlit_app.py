"""Streamlit web application for customer support ticket classification."""

import os
from pathlib import Path

import streamlit as st
import torch
from loguru import logger
from transformers import DistilBertTokenizer

from customer_support.model import TicketClassificationModule

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for branding
st.markdown(
    """
    <style>
    :root {
        --primary-color: #1f77b4;
        --background-color: #ffffff;
        --secondary-background-color: #f0f2f6;
        --text-color: #262730;
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }

    .priority-high {
        color: #d62728;
        font-weight: bold;
    }

    .priority-medium {
        color: #ff7f0e;
        font-weight: bold;
    }

    .priority-low {
        color: #2ca02c;
        font-weight: bold;
    }

    .badge {
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        text-align: center;
        font-weight: bold;
        color: white;
    }

    .badge-high {
        background-color: #d62728;
    }

    .badge-medium {
        background-color: #ff7f0e;
    }

    .badge-low {
        background-color: #2ca02c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🎟️ Ticket Classifier")
    st.markdown("---")

    st.subheader("About")
    st.write(
        """
        This application classifies customer support tickets by priority level
        using a fine-tuned **DistilBERT** model.

        **Priority Levels:**
        - 🟢 **Low**: Non-urgent tickets
        - 🟡 **Medium**: Standard priority
        - 🔴 **High**: Urgent, requires immediate attention
        """
    )

    st.markdown("---")

    st.subheader("Documentation")
    st.markdown(
        """
        - [📊 Data](https://dtu-mlops-group-41.github.io/mlops-project/data/)
        - [🤖 Model](https://dtu-mlops-group-41.github.io/mlops-project/model/)
        - [🚀 API](https://dtu-mlops-group-41.github.io/mlops-project/api/)
        - [🐳 Training](https://dtu-mlops-group-41.github.io/mlops-project/training/)
        """
    )

    st.markdown("---")

    st.subheader("Model Info")
    st.info(
        """
        **Model**: DistilBERT Base Multilingual Cased
        - **Parameters**: ~66M
        - **Task**: Sequence Classification
        - **Languages**: Multilingual support
        """
    )

# ============================================================================
# SESSION STATE & CACHING
# ============================================================================


@st.cache_resource
def load_model() -> tuple[TicketClassificationModule, DistilBertTokenizer]:
    """Load the trained model and tokenizer."""
    model_path = Path(os.getenv("MODEL_PATH", "models/model.ckpt"))

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    logger.info(f"Loading model from {model_path}")

    # Load model
    model = TicketClassificationModule.load_from_checkpoint(
        model_path,
        local_files_only=True,
    )
    model.eval()
    model.freeze()

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-multilingual-cased",
        local_files_only=True,
    )

    return model, tokenizer


def predict(text: str, model: TicketClassificationModule, tokenizer: DistilBertTokenizer) -> dict:
    """Predict ticket priority.

    Args:
        text: Ticket body text
        model: Loaded TicketClassificationModule
        tokenizer: DistilBERT tokenizer

    Returns:
        Dictionary with priority, priority_id, and confidence
    """
    # Tokenize input
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Get predictions
    logits = outputs[0]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()

    # Map class ID to priority name
    priority_map = {0: "low", 1: "medium", 2: "high"}
    priority = priority_map[predicted_class]

    return {
        "priority": priority,
        "priority_id": predicted_class,
        "confidence": confidence,
    }


def get_badge_html(priority: str, confidence: float) -> str:
    """Generate HTML badge for priority display.

    Args:
        priority: Priority level (low, medium, high)
        confidence: Confidence score (0-1)

    Returns:
        HTML string for the badge
    """
    emoji_map = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    badge_class = f"badge badge-{priority}"
    emoji = emoji_map.get(priority, "")

    return f'<div class="{badge_class}">{emoji} {priority.upper()} ({confidence * 100:.1f}%)</div>'


# ============================================================================
# MAIN APP
# ============================================================================

st.title("🎟️ Customer Support Ticket Classifier")
st.markdown(
    "Classify customer support tickets by priority using AI-powered DistilBERT model.",
    help="Enter a ticket description and get an instant priority classification.",
)

st.markdown("---")

# Load model
try:
    model, tokenizer = load_model()
    st.session_state.model_loaded = True
except FileNotFoundError as e:
    st.error(f"❌ {str(e)}")
    st.stop()

# Main input section
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📝 Enter Ticket Details")

with col2:
    st.markdown("##### Examples")

# Ticket input
ticket_text = st.text_area(
    label="Ticket Body",
    placeholder="Describe the customer support issue...",
    height=150,
    label_visibility="collapsed",
    help="Enter the customer's support ticket text for classification",
)

# Example tickets sidebar
with col2:
    with st.expander("📋 Sample Tickets", expanded=False):
        examples = {
            "Low": "I'd like to know how to change my profile picture. Is there a help section?",
            "Medium": "I'm unable to log into my account. I've tried resetting my password but it's not working.",
            "High": "The system is completely down and I have an important presentation in 1 hour. This is critical!",
        }

        for priority, example_text in examples.items():
            if st.button(f"Use {priority} Example", key=f"example_{priority}"):
                ticket_text = example_text
                st.rerun()

# Predict button
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    predict_button = st.button("🔍 Classify Ticket", use_container_width=True, type="primary")

# Prediction results
if predict_button:
    if not ticket_text.strip():
        st.warning("⚠️ Please enter a ticket description first.")
    else:
        with st.spinner("Classifying ticket..."):
            prediction = predict(ticket_text, model, tokenizer)

            # Store in session state for history
            if "predictions" not in st.session_state:
                st.session_state.predictions = []
            st.session_state.predictions.append(
                {"text": ticket_text, "prediction": prediction},
            )

        st.markdown("---")
        st.subheader("✅ Classification Result")

        # Display results in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Priority Level**")
            st.markdown(get_badge_html(prediction["priority"], prediction["confidence"]), unsafe_allow_html=True)

        with col2:
            st.metric("Confidence Score", f"{prediction['confidence'] * 100:.1f}%")

        with col3:
            priority_descriptions = {
                "low": "Non-urgent ticket",
                "medium": "Standard priority",
                "high": "Urgent attention needed",
            }
            st.info(f"**Note:** {priority_descriptions[prediction['priority']]}")

        # Additional info
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Ticket Text:**")
            st.text(ticket_text[:300] + ("..." if len(ticket_text) > 300 else ""))

        with col2:
            st.write("**Model Used:**")
            st.caption("DistilBERT Base Multilingual Cased")

# History section (collapsible)
if "predictions" in st.session_state and len(st.session_state.predictions) > 0:
    st.markdown("---")
    with st.expander(f"📜 Prediction History ({len(st.session_state.predictions)} predictions)", expanded=False):
        for i, pred in enumerate(reversed(st.session_state.predictions), 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"Ticket {len(st.session_state.predictions) - i + 1}")
                st.text(pred["text"][:100] + "..." if len(pred["text"]) > 100 else pred["text"])
            with col2:
                st.markdown(
                    get_badge_html(pred["prediction"]["priority"], pred["prediction"]["confidence"]),
                    unsafe_allow_html=True,
                )

        if st.button("🗑️ Clear History"):
            st.session_state.predictions = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.85rem; padding: 20px;">
        <p>Customer Support Ticket Classifier | DTU MLOps Project | Built with Streamlit 🎈</p>
    </div>
    """,
    unsafe_allow_html=True,
)
