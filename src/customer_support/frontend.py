"""Streamlit web application for customer support ticket classification via FastAPI."""

import os
from typing import Any

import requests  # type: ignore
import streamlit as st
from loguru import logger

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
            prediction = predict_via_api(ticket_text, st.session_state.api_url)

            if prediction is None:
                st.error(
                    f"❌ Failed to connect to API at {st.session_state.api_url}\n\n"
                    "Please ensure:\n"
                    "1. FastAPI server is running: `uv run uvicorn customer_support.api:app --port 8080`\n"
                    "2. API URL is correct in the sidebar\n"
                    "3. Model checkpoint is available at `models/model.ckpt`"
                )
            else:
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
                    st.markdown(
                        get_badge_html(prediction["priority"], prediction["confidence"]),
                        unsafe_allow_html=True,
                    )

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
                    st.caption("DistilBERT Base Multilingual Cased (via FastAPI)")

# History section (collapsible)
if "predictions" in st.session_state and len(st.session_state.predictions) > 0:
    st.markdown("---")
    with st.expander(
        f"📜 Prediction History ({len(st.session_state.predictions)} predictions)",
        expanded=False,
    ):
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
