import os

import pathlib, base64, subprocess
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import torch
from utils.evaluator import Evaluator

model_dir = "output"

MODEL_DOUBLE_TASK = {
    "combined_tasks_10": False,
}


def get_num_devices():
    return 1 + torch.cuda.device_count() if torch.cuda.is_available() else 1


def get_available_models():
    paths = [p.name for p in pathlib.Path(model_dir).iterdir() if p.is_dir()]
    paths.sort()
    return paths


def generate_pdf(abstract, title, keywords, out_dir=".", out_name="document"):

    latex_template = f"""
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
    {abstract}

    \\vspace{{0.5cm}}
    \\hspace{{0.15cm}}
    \\small{{\\textbf{{Keywords:}} {keywords}}}


    \\end{{document}}
    """

    tex = os.path.join(out_dir, f"{out_name}.tex")
    # Write the LaTeX source to a file
    with open(tex, "w") as f:
        f.write(latex_template)

    try:
        # Compile the LaTeX file to PDF
        subprocess.run(["pdflatex", tex], check=True)
        for ext in ["aux", "log", "tex"]:
            os.remove(os.path.join(out_dir, f"{out_name}.{ext}"))

        with open(os.path.join(out_dir, f"{out_name}.pdf"), "rb") as pdf_file:
            pdf_data = pdf_file.read()
            return pdf_data

    except subprocess.CalledProcessError as e:
        raise RuntimeError("pdflatex error")


def display_pdf(pdf_data):
    base64_pdf = base64.b64encode(pdf_data).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# Load the model and tokenizer with device selection
@st.cache_resource
def load_model(model_main_dir: str | None, device: torch.device):
    if model_main_dir is None:
        return None, None
    model_name = os.path.join(model_main_dir, "model")
    tokenizer_name = os.path.join(model_main_dir, "tokenizer")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = model.to(device)
    return model, tokenizer


# Tokenize input and check token count
def tokenize_input(
    abstract: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_tokens: int = 512,
):
    tokens = tokenizer(
        abstract, return_tensors="pt", truncation=True, max_length=max_tokens
    )
    return tokens


def prediction(model, tokenizer, input_text):

    tokenized_input = tokenize_input(input_text, tokenizer)
    inputs = {k: v.to(device) for k, v in tokenized_input.items()}
    output = model.generate(
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

# Sidebar options
st.sidebar.header("Configuration")

# Device selection dropdown in the sidebar
device_option = st.sidebar.selectbox(
    "Select the device",
    options=range(get_num_devices()),
    format_func=lambda idx: (
        "CPU" if idx == 0 else f"GPU ({torch.cuda.get_device_name(idx-1)})"
    ),
)
model_options = st.sidebar.selectbox("Select the model", get_available_models())

device = torch.device(f"cuda:{device_option-1}" if device_option > 0 else "cpu")
model_name = None if not model_options else os.path.join(model_dir, model_options)
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

if (
    st.button("Generate Title and Keywords 𓂃🖊")
    and model is not None
    and tokenizer is not None
):
    if abstract_input:
        # Generate the title and keywords using the model
        with st.spinner("Generating... ✨"):
            with torch.inference_mode():

                if MODEL_DOUBLE_TASK[model_options]:

                    pred = {}

                    for prefix, target in [
                        ("generate title", "Title"),
                        ("generate keywords", "Keywords"),
                    ]:
                        prefix_abstract = f"{prefix}: {abstract_input}"
                        pred[target] = prediction(model, tokenizer, prefix_abstract)

                    st.session_state["title"], st.session_state["keywords"] = (
                        pred["Title"],
                        pred["Keywords"],
                    )

                else:

                    prefix_abstract = f"generate title and keywords: {abstract_input}"

                    generated_text = prediction(model, tokenizer, prefix_abstract)

                    # generated_text = "Title: Title of the document\nKeywords: keyword1, keyword2, keyword3"

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
                "Impossible to generate PDF, missing LaTeX installation in your system."
            )

# Footer
st.markdown("---")
st.write(
    "Demo built with [Streamlit](https://streamlit.io/) and [Hugging Face](https://huggingface.co/)."
)
