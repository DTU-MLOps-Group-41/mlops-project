"""Streamlit web application for customer support ticket classification with hybrid inference."""

import os
from pathlib import Path
from typing import Any

import requests  # type: ignore
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
        using a fine-tuned **DistilBERT** model via FastAPI backend.

        **Priority Levels:**
        - 🟢 **Low**: Non-urgent tickets
        - 🟡 **Medium**: Standard priority
        - 🔴 **High**: Urgent, requires immediate attention
        """
    )

    st.markdown("---")

    st.subheader("API Configuration")
    api_url = st.text_input(
        "FastAPI URL",
        value=os.getenv("API_URL", "http://localhost:8080"),
        help="FastAPI backend endpoint",
    )
    st.session_state.api_url = api_url

    if st.button("🔗 Check API Connection", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connected to API")
            else:
                st.error(f"❌ API returned status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")

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
# HELPER FUNCTIONS
# ============================================================================


@st.cache_resource
def load_local_model() -> tuple[TicketClassificationModule | None, DistilBertTokenizer]:
    """Load model and tokenizer for local inference fallback.

    Returns None for model if checkpoint not found.
    """
    model_path = Path(os.getenv("MODEL_PATH", "models/model.ckpt"))
    model = None

    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        try:
            model = TicketClassificationModule.load_from_checkpoint(
                model_path,
                local_files_only=True,
            )
            model.eval()
            model.freeze()
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            model = None
    else:
        logger.warning(f"Model checkpoint not found at {model_path}")

    # Load tokenizer
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-multilingual-cased",
            local_files_only=True,
        )
    except Exception:
        logger.warning("Downloading tokenizer from HuggingFace")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    return model, tokenizer


@st.cache_data
def predict_via_api(text: str, api_url: str) -> dict[str, Any] | None:
    """Call FastAPI endpoint for prediction.

    Args:
        text: Ticket body text
        api_url: FastAPI backend URL

    Returns:
        Prediction dictionary or None if error
    """
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"text": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        logger.error(f"Failed to connect to API at {api_url}")
        return None
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return None
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return None


def predict_locally(
    text: str, model: TicketClassificationModule | None, tokenizer: DistilBertTokenizer
) -> dict[str, Any]:
    """Perform local prediction using the model.

    Args:
        text: Ticket body text
        model: Loaded TicketClassificationModule or None for demo mode
        tokenizer: DistilBERT tokenizer

    Returns:
        Prediction dictionary
    """
    # Demo mode: keyword-based predictions
    if model is None:
        keywords_high = ["urgent", "critical", "emergency", "down", "broken", "crash", "fail"]
        keywords_medium = ["help", "issue", "problem", "error", "not working", "unable"]

        text_lower = text.lower()
        if any(kw in text_lower for kw in keywords_high):
            priority = "high"
            priority_id = 2
            confidence = 0.85
        elif any(kw in text_lower for kw in keywords_medium):
            priority = "medium"
            priority_id = 1
            confidence = 0.72
        else:
            priority = "low"
            priority_id = 0
            confidence = 0.68

        return {
            "priority": priority,
            "priority_id": priority_id,
            "confidence": confidence,
        }

    # Real inference
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

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    logits = outputs[0]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()

    priority_map = {0: "low", 1: "medium", 2: "high"}
    priority = priority_map[predicted_class]

    return {
        "priority": priority,
        "priority_id": predicted_class,
        "confidence": confidence,
    }


def predict_with_fallback(
    text: str,
    api_url: str,
    local_model: TicketClassificationModule | None,
    tokenizer: DistilBertTokenizer,
) -> tuple[dict[str, Any], str]:
    """Try API first, then fall back to local inference.

    Args:
        text: Ticket body text
        api_url: FastAPI backend URL
        local_model: Local model for fallback
        tokenizer: Tokenizer for fallback

    Returns:
        Tuple of (prediction dict, inference_mode)
    """
    # Try API first
    prediction = predict_via_api(text, api_url)
    if prediction is not None:
        return prediction, "API"

    # Fall back to local inference
    logger.info("API unavailable, using local inference")
    prediction = predict_locally(text, local_model, tokenizer)
    return prediction, "Local"


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

# Initialize API URL in session state
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", "http://localhost:8080")

# Load local model for fallback
local_model, tokenizer = load_local_model()

# Main input section
st.subheader("📝 Enter Ticket Details")

# Ticket input
ticket_text = st.text_area(
    label="Ticket Body",
    placeholder="Describe the customer support issue...",
    height=150,
    label_visibility="collapsed",
    help="Enter the customer's support ticket text for classification",
)

# Example tickets table
st.markdown("**Example Tickets:**")
examples = {
    "Low": "I'd like to know how to change my profile picture. Is there a help section?",
    "Medium": "I'm unable to log into my account. I've tried resetting my password but it's not working.",
    "High": "The system is completely down and I have an important presentation in 1 hour. This is critical!",
}

example_cols = st.columns([1, 4, 1])
with example_cols[0]:
    st.markdown("**Priority**")
with example_cols[1]:
    st.markdown("**Example Ticket**")
with example_cols[2]:
    st.markdown("")

for priority, example_text in examples.items():
    emoji_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.caption(f"{emoji_map[priority]} {priority}")
    with col2:
        st.caption(example_text)
    with col3:
        if st.button("Use", key=f"example_{priority}", use_container_width=True):
            st.session_state.example_ticket = example_text
            st.rerun()

# Check if example was selected
if "example_ticket" in st.session_state:
    ticket_text = st.session_state.example_ticket
    del st.session_state.example_ticket

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
            # Try API first, fall back to local inference
            prediction, inference_mode = predict_with_fallback(
                ticket_text,
                st.session_state.api_url,
                local_model,
                tokenizer,
            )

            # Store in session state for history
            if "predictions" not in st.session_state:
                st.session_state.predictions = []
            st.session_state.predictions.append(
                {"text": ticket_text, "prediction": prediction, "mode": inference_mode},
            )

            st.markdown("---")
            st.subheader("✅ Classification Result")

            # Display inference mode
            if inference_mode == "API":
                mode_badge = "🌐 **Via API**"
            else:
                mode_badge = "🖥️ **Local Inference**"

            # Display results in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Priority Level**")
                st.markdown(
                    get_badge_html(prediction["priority"], prediction["confidence"]),
                    unsafe_allow_html=True,
                )

            with col2:
                st.metric("Confidence Score", f"{prediction['confidence'] * 100:.1f}%")

            with col3:
                st.metric("Inference Mode", mode_badge)

            # Additional info
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Ticket Text:**")
                st.text(ticket_text[:300] + ("..." if len(ticket_text) > 300 else ""))

            with col2:
                st.write("**Model Used:**")
                if inference_mode == "API":
                    st.caption("DistilBERT via FastAPI Backend")
                elif local_model is not None:
                    st.caption("DistilBERT Local Inference")
                else:
                    st.caption("Demo Mode (Keywords)")

# History section (collapsible)
if "predictions" in st.session_state and len(st.session_state.predictions) > 0:
    st.markdown("---")
    with st.expander(
        f"📜 Prediction History ({len(st.session_state.predictions)} predictions)",
        expanded=False,
    ):
        for i, pred in enumerate(reversed(st.session_state.predictions), 1):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.text(f"Ticket {len(st.session_state.predictions) - i + 1}")
                st.text(pred["text"][:100] + "..." if len(pred["text"]) > 100 else pred["text"])
            with col2:
                st.markdown(
                    get_badge_html(pred["prediction"]["priority"], pred["prediction"]["confidence"]),
                    unsafe_allow_html=True,
                )
            with col3:
                mode_emoji = "🌐" if pred.get("mode") == "API" else "🖥️" if pred.get("mode") == "Local" else "🎯"
                st.caption(f"{mode_emoji} {pred.get('mode', 'Unknown')}")

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
