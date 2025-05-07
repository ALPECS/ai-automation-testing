import streamlit as st
import pandas as pd
import logging
import io
import os

# Import the refactored test runner function and constants
from chatgpt_testing import run_test, RESULTS_CSV_PATH

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Calculus Problem Tester", layout="wide")
st.title("Calculus Problem Tester using OpenRouter")
st.markdown(f"""
This app runs the calculus problem testing script (`chatgpt_testing.py`).
It processes problems defined in `problems.csv`, sends images from the directory
to the specified AI model via OpenRouter, evaluates the responses using SymPy.
""")

# --- Logging Handler for Streamlit ---
# Create a string buffer to capture logs
log_stream = io.StringIO()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a custom handler that writes to the string stream
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setLevel(logging.INFO)
# Optional: Add formatting to the handler if desired
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
# Add the custom handler to the logger from chatgpt_testing
logger.addHandler(stream_handler)
# Ensure logs also go to console (Streamlit's default behavior might suppress this otherwise)
# logger.addHandler(logging.StreamHandler()) # Uncomment if console logs disappear


# --- UI Elements ---
log_placeholder = st.empty()
results_placeholder = st.empty()

# Display initial log area
log_placeholder.text_area("Logs", "", height=300)

if st.button("▶️ Start Testing"):
    # Clear previous logs and results
    log_stream.seek(0)
    log_stream.truncate(0)
    results_placeholder.empty()
    log_placeholder.text_area("Logs", "Starting test run...", height=300)

    st.info("Test run initiated. Please wait... Logs will appear below.")

    try:
        # --- Execute the Test ---
        with st.spinner("Processing problems... This might take a while."):
            results_df = run_test()

        # --- Display Logs ---
        log_stream.seek(0) # Rewind stream to read its content
        log_content = log_stream.read()
        log_placeholder.text_area("Logs", log_content, height=400)
        st.success("Test run finished!")

        # --- Display Results ---
        if not results_df.empty:
            st.subheader("Test Results")
            results_placeholder.dataframe(results_df)
            st.markdown(f"Results have been saved to `{RESULTS_CSV_PATH}`")

            # Provide download button for the CSV
            try:
                # Ensure results_df is used for download
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results CSV",
                    data=csv_data,
                    file_name=os.path.basename(RESULTS_CSV_PATH),
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Could not prepare results file for download: {e}")

        else:
            results_placeholder.warning("No results were generated.")

    except Exception as e:
        st.error(f"An error occurred during the test run: {e}")
        # Display any logs captured before the error
        log_stream.seek(0)
        log_content = log_stream.read()
        log_placeholder.text_area("Logs", log_content + f"\nERROR: {e}", height=400)

# --- Cleanup ---
# It's good practice to remove the handler if the app reruns or stops,
# though Streamlit's execution model might make this tricky to place perfectly.
# Consider context managers if needed for more complex scenarios.
# logger.removeHandler(stream_handler) # This might cause issues on reruns if not handled carefully 