# streamlit_app.py

import streamlit as st
import os
import sys
import shutil
import pandas as pd
import subprocess
import time
import uuid
import io
from pathlib import Path
from dotenv import dotenv_values

# --- Configuration ---
from config import (
    CRAWL_CACHE_DIR, KNOWLEDGE_CACHE_DIR, RACE_INPUT_FILE, OUTPUT_DIR
)
from gemini import DEFAULT_INPUT_FOLDER

# --- App Directories (Defined at the top for reliability) ---
TEMP_DIR = Path("./temp_streamlit_files")
TEMP_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("./streamlit_logs")
LOG_DIR.mkdir(exist_ok=True)

# --- App State Management ---
if 'agent_status' not in st.session_state:
    st.session_state.agent_status = "Idle"  # Idle, Running, Finished, Terminated, Error
if 'active_process' not in st.session_state:
    st.session_state.active_process = None
if 'log_file' not in st.session_state:
    st.session_state.log_file = None
if 'final_log_content' not in st.session_state:
    st.session_state.final_log_content = ""
if 'output_files' not in st.session_state:
    st.session_state.output_files = []
if 'files_before_run' not in st.session_state:
    st.session_state.files_before_run = set()
if 'env_vars' not in st.session_state:
    st.session_state.env_vars = {}


# --- Helper Functions ---
def read_log_file():
    """Reads the current log file's content safely."""
    if st.session_state.log_file and Path(st.session_state.log_file).exists():
        try:
            with open(st.session_state.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return "Reading log file..."
    return "Process has not started yet."


def get_process_environment():
    """Prepares the environment variables for the subprocess."""
    env = os.environ.copy()
    if st.session_state.env_vars:
        env.update(st.session_state.env_vars)
    return env


def get_files_in_dir(directory):
    """Returns a set of all CSV files in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        dir_path.mkdir(exist_ok=True)
    return {str(f) for f in dir_path.glob("*.csv")}


# --- UI Layout ---
st.set_page_config(page_title="Crawl4AI Agent System", layout="wide")
st.title("Crawl4AI Agent System")

# --- Sidebar ---
with st.sidebar:
    st.header("Global Configuration")
    st.info("Upload a single `.env` file for API keys.")
    env_file = st.file_uploader("Upload .env File", type=['env'])
    if env_file:
        try:
            env_contents = env_file.getvalue().decode('utf-8')
            st.session_state.env_vars = dotenv_values(stream=io.StringIO(env_contents))
            st.success(f"Loaded {len(st.session_state.env_vars)} API keys.")
        except Exception as e:
            st.error(f"Failed to parse .env file: {e}")

    with st.expander("Utilities"):
        if st.button("Clear Crawl Cache"):
            try:
                if Path(CRAWL_CACHE_DIR).exists(): shutil.rmtree(CRAWL_CACHE_DIR)
                Path(CRAWL_CACHE_DIR).mkdir(exist_ok=True)
                st.success("Cleared Crawl Cache.")
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Clear Knowledge Cache"):
            try:
                if Path(KNOWLEDGE_CACHE_DIR).exists(): shutil.rmtree(KNOWLEDGE_CACHE_DIR)
                Path(KNOWLEDGE_CACHE_DIR).mkdir(exist_ok=True)
                st.success("Cleared Knowledge Cache.")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Main Interface ---
col1, col2 = st.columns(2)

with col1:
    st.header("Agent Control Panel")
    agent_tabs = st.tabs(["Mistral Web Analyst", "Gemini Image Processor"])

    with agent_tabs[0]:
        st.subheader("1. Inputs")
        input_file = st.file_uploader("Upload Data File (JSON/CSV/XLSX)", type=['json', 'csv', 'xlsx'])
        st.caption(f"Output files will be saved to: **{os.path.abspath(OUTPUT_DIR)}**")

        st.subheader("2. Execution")
        is_running = st.session_state.agent_status == "Running"
        if st.button("▶️ Start Mistral Agent", disabled=is_running):
            if not st.session_state.env_vars:
                st.error("Please upload a .env file first.")
            elif not input_file:
                st.error("Please upload an input data file.")
            else:
                st.session_state.output_files = []
                st.session_state.log_file = LOG_DIR / f"run_{uuid.uuid4()}.log"
                st.session_state.final_log_content = ""
                temp_input_path = TEMP_DIR / RACE_INPUT_FILE

                try:
                    file_contents = input_file.getvalue()
                    if input_file.name.lower().endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(file_contents))
                        df.to_json(temp_input_path, orient='records', indent=4)
                    elif input_file.name.lower().endswith('.xlsx'):
                        df = pd.read_excel(io.BytesIO(file_contents))
                        df.to_json(temp_input_path, orient='records', indent=4)
                    else:
                        with open(temp_input_path, "wb") as f:
                            f.write(file_contents)

                    st.session_state.files_before_run = get_files_in_dir(OUTPUT_DIR)
                    command = [sys.executable, "-u", "main.py", "--output-dir", OUTPUT_DIR, "--input-file",
                               str(temp_input_path)]

                    with open(st.session_state.log_file, 'w', encoding='utf-8') as log_f:
                        process = subprocess.Popen(command, stdout=log_f, stderr=subprocess.STDOUT,
                                                   env=get_process_environment(), text=True, encoding='utf-8')

                    st.session_state.active_process = process
                    st.session_state.agent_status = "Running"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during agent setup: {e}")

        st.subheader("3. Results")
        if st.session_state.agent_status == "Finished":
            if st.session_state.output_files:
                st.success("Processing finished! You can now download your files.")
                for file_path in st.session_state.output_files:
                    file = Path(file_path)
                    with open(file, "rb") as fp:
                        st.download_button(label=f"Download {file.name}", data=fp, file_name=file.name, mime="text/csv")
            else:
                st.warning("Process finished, but no new output files were generated.")
        else:
            st.info("Results from the current run will appear here after completion.")

    with agent_tabs[1]:
        st.subheader("1. Inputs")
        uploaded_images = st.file_uploader("Upload Image Files", type=['png', 'jpg', 'jpeg'],
                                           accept_multiple_files=True)
        st.caption(f"Image files will be temporarily saved to: **{os.path.abspath(DEFAULT_INPUT_FOLDER)}**")
        st.caption(f"Result CSV will be saved to: **{os.path.abspath(OUTPUT_DIR)}**")

        st.subheader("2. Execution")
        is_running = st.session_state.agent_status == "Running"
        if st.button("▶️ Start Gemini Processor", disabled=is_running):
            if not st.session_state.env_vars:
                st.error("Please upload a .env file first.")
            elif not uploaded_images:
                st.error("Please upload at least one image.")
            else:
                st.session_state.output_files = []
                st.session_state.log_file = LOG_DIR / f"run_{uuid.uuid4()}.log"
                st.session_state.final_log_content = ""
                image_input_path = Path(DEFAULT_INPUT_FOLDER)
                image_input_path.mkdir(exist_ok=True)
                for image in uploaded_images:
                    with open(image_input_path / image.name, "wb") as f: f.write(image.getbuffer())

                st.session_state.files_before_run = get_files_in_dir(OUTPUT_DIR)
                command = [sys.executable, "-u", "gemini.py", "--output-dir", OUTPUT_DIR, "--input-dir",
                           str(image_input_path)]

                with open(st.session_state.log_file, 'w', encoding='utf-8') as log_f:
                    process = subprocess.Popen(command, stdout=log_f, stderr=subprocess.STDOUT,
                                               env=get_process_environment(), text=True, encoding='utf-8')
                st.session_state.active_process = process
                st.session_state.agent_status = "Running"
                st.rerun()

with col2:
    st.header("Live Status & Logs")

    status = st.session_state.agent_status
    if status == "Idle":
        st.info("**Status:** Waiting to start a process.")
    elif status == "Running":
        st.warning(f"**Status:** Running... Please wait.")
    elif status == "Finished":
        st.success("**Status:** Process finished successfully.")
    elif status == "Terminated":
        st.error("**Status:** Process stopped by user.")
    elif status == "Error":
        st.error("**Status:** Process failed with an error. Check logs for details.")

    if st.session_state.agent_status == "Running":
        if st.button("⏹️ Stop Active Process"):
            if st.session_state.active_process:
                st.session_state.active_process.terminate()
                st.session_state.agent_status = "Terminated"
                st.session_state.active_process = None
                st.rerun()

    log_content = st.session_state.final_log_content if status != "Running" else read_log_file()
    st.text_area("Log Output", value=log_content, height=500, disabled=True, key="log_area")
    st.caption(
        "Note: 'Telemetry' and 'huggingface_hub' messages are harmless warnings from libraries and can be safely ignored.")

# --- Background Loop to Update UI State ---
if st.session_state.agent_status == "Running":
    process = st.session_state.active_process
    if process and process.poll() is not None:
        time.sleep(0.5)
        st.session_state.final_log_content = read_log_file()

        return_code = process.poll()
        if return_code == 0:
            st.session_state.agent_status = "Finished"
            files_after_run = get_files_in_dir(OUTPUT_DIR)
            st.session_state.output_files = sorted(list(files_after_run - st.session_state.files_before_run))
        else:
            st.session_state.agent_status = "Error"

        st.session_state.active_process = None
        st.rerun()
    else:
        time.sleep(1.5)
        st.rerun()