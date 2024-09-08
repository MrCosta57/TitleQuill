"""
This script serves as a GUI to explore TitleQuill
"""

import os
import pathlib
import base64
import platform
import subprocess, tempfile
from typing import Dict

import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from utils.evaluator import Evaluator

MODEL_DIR = "output"

MODEL_DOUBLE_TASK: Dict[str, bool] = {
    "combined_tasks_10": False,
}


def get_num_devices() -> int:
    """
    Get the number of available devices
    Returns one if only CPU and more than one if GPU is available
    """
    return 1 + torch.cuda.device_count() if torch.cuda.is_available() else 1


def get_available_models() -> list[str]:
    """
    Returns the list of available models
    """

    paths = [p.name for p in pathlib.Path(MODEL_DIR).iterdir() if p.is_dir()]
    paths.sort()
    return paths


def generate_pdf(abstract: str, title: str, keywords: str):

    match platform.system():

        case "Windows": newline = f'\\\\'
        case "Linux":   newline = r'\\\\'
        case "Darwin":  newline = r'\\\\'
        case _:         newline = ''

    TEMPLATE = f"""
    \\documentclass[12pt]{{article}}
    \\usepackage[utf8]{{inputenc}}
    \\usepackage{{amsmath}}
    \\usepackage{{geometry}}
    \\geometry{{a4paper, margin=1in}}

    \\title{{{title}}}
    \\date{{\\today}}

    \\begin{{document}}

    \\maketitle

    \\section*{{Abstract}}
    \\noindent
    {abstract}
    """ + newline + f"""

    \\vspace{{0.5cm}}
    \\noindent
    \\small{{\\textbf{{Keywords:}} {keywords}}}

    \\end{{document}}
    """

    pdf_data = ""
    with tempfile.TemporaryDirectory() as temp_dir:

        tex_path = os.path.join(temp_dir, "document.tex")
        # Write the LaTeX source to a file
        with open(tex_path, "w") as f:
            f.write(TEMPLATE)

        original_dir = os.getcwd()
        os.chdir(temp_dir)
        try:
            # Compile the LaTeX file to PDF
            subprocess.run(
                ["pdflatex", "document.tex"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            pdf_path = os.path.join(temp_dir, "document.pdf")
            with open(pdf_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()

        except subprocess.CalledProcessError as e:
            raise RuntimeError("`pdflatex` error occurred") from e
        finally:
            os.chdir(original_dir)

    return pdf_data


def display_pdf(pdf_data: bytes):
    """Display the PDF in the app"""
    base64_pdf = base64.b64encode(pdf_data).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# Load the model and tokenizer with device selection
@st.cache_resource
def load_model(model_main_dir: str | None, device: torch.device):
    """Load the model and tokenizer from the given directory"""

    # Return None if the model directory is not provided
    if model_main_dir is None:
        return None, None

    # Load the model and tokenizer
    model_name = os.path.join(model_main_dir, "model")
    tokenizer_name = os.path.join(model_main_dir, "tokenizer")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Move the model to the selected device
    model = model.to(device)

    return model, tokenizer


# Tokenize input and check token count
def tokenize_input(
    abstract: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_tokens: int = 512,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the input text and check the token count

    :param abstract: The input text to tokenize
    :param tokenizer: The tokenizer to use
    :param max_tokens: The maximum number of tokens to use
    """

    tokens = tokenizer(
        abstract, return_tensors="pt", truncation=True, max_length=max_tokens
    )
    return tokens  # type: ignore


def prediction(
    model: AutoModelForSeq2SeqLM,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    input_text: str,
    device: torch.device,
):
    """
    Perform the prediction using the model and tokenizer

    :param model: The model to use for prediction
    :param tokenizer: The tokenizer to use for prediction
    :param input_text: The input text to generate the output from
    """

    tokenized_input = tokenize_input(input_text, tokenizer)
    inputs = {k: v.to(device) for k, v in tokenized_input.items()}

    output = model.generate(  # type: ignore
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_token_length,
    )

    generated_text = tokenizer.decode(
        output[0],
        clean_up_tokenization_spaces=True,
        skip_special_tokens=True,
    )

    return generated_text


# Initialize the Streamlit app
st.title(
    "🪶TitleQuill: Unified Framework for Titles and Keywords Generation using Pre-Trained Model🪶"
)
st.markdown(
    """
**Authors**:  
👨‍💻 Giovanni Costa
👨‍💻 Sebastiano Quintavalle
👨‍💻 Nicola Aggio
👩‍💻 Martina Novello
"""
)

st.write("Welcome to **TitleQuill**!")

st.caption(
    "This app generates precise title and suitable keywords from a given abstract using a state-of-the-art model. 🚀"
)

st.sidebar.header("Configuration")

# Device selection dropdown in the sidebar
device_option = st.sidebar.selectbox(
    "Select the device",
    options=range(get_num_devices()),
    format_func=lambda idx: (
        "CPU" if idx == 0 else f"GPU ({torch.cuda.get_device_name(idx-1)})"
    ),
)

# Model selection dropdown in the sidebar
model_options = st.sidebar.selectbox("Select the model", get_available_models())

# Load the model and tokenizer
device = torch.device(f"cuda:{device_option-1}" if device_option > 0 else "cpu")

# Load the model and tokenizer
model_name = None if not model_options else os.path.join(MODEL_DIR, model_options)
model, tokenizer = load_model(model_name, device)

# Max token length slider in the sidebar
max_token_length = st.sidebar.slider(
    "Max token length for the output", min_value=10, max_value=150, value=50
)

# Text input for the abstract (limited to 512 tokens) in the main section
abstract_input = st.text_area("📃 Enter the abstract (max 512 tokens)", height=200)

# Title and keywords session state initialization
if "title" not in st.session_state:
    st.session_state["title"] = ""
if "keywords" not in st.session_state:
    st.session_state["keywords"] = ""

# Title generation button
if (
    st.button("Generate Title and Keywords 𓂃🖊")
    and model is not None
    and tokenizer is not None
):
    if abstract_input:
        # Generate the title and keywords using the model
        with st.spinner("Generating... ✨"):
            with torch.inference_mode():
                # Double task
                if MODEL_DOUBLE_TASK[model_options]:
                    pred = {}

                    for prefix, target in [
                        ("generate title", "Title"),
                        ("generate keywords", "Keywords"),
                    ]:
                        prefix_abstract = f"{prefix}: {abstract_input}"
                        pred[target] = prediction(
                            model, tokenizer, prefix_abstract, device
                        )

                    st.session_state["title"], st.session_state["keywords"] = (
                        pred["Title"],
                        pred["Keywords"],
                    )
                # Single task
                else:
                    prefix_abstract = f"generate title and keywords: {abstract_input}"
                    generated_text = prediction(
                        model, tokenizer, prefix_abstract, device
                    )
                    # generated_text = "Title: Title of the document. Keywords: keyword1, keyword2, keyword3"
                    st.session_state["title"], st.session_state["keywords"] = (
                        Evaluator.split_title_keywords([generated_text])[0]
                    )
    else:
        st.warning("Please enter an abstract to generate the title and keywords.")


# Display the generated title and keywords from session state, even after PDF generation
if st.session_state["title"] or st.session_state["keywords"]:

    title = st.session_state["title"].capitalize()
    if title.endswith("."):
        title = title[:-1]

    keywords = ", ".join(st.session_state["keywords"])

    st.subheader("🎯 Generated Title and Keywords")
    st.write(f"**Title**: {title}")
    st.write(f"**Keywords**: {keywords}")

    # Button to generate the PDF
    if st.button("Generate PDF 𓂃📄"):
        try:
            pdf_data = generate_pdf(abstract_input, title, keywords)
            # Display the PDF preview in the app
            display_pdf(pdf_data)

            # Provide a download button
            st.download_button(
                "Download PDF",
                pdf_data,
                file_name="paper.pdf",
                mime="application/octet-stream",
            )
        except Exception as e:
            st.warning(
                "Impossible to generate PDF. Please check your LaTeX installation."
            )

# Footer
st.markdown("---")
st.write(
    "Demo built with [Streamlit](https://streamlit.io/) and [Hugging Face](https://huggingface.co/)."
)
