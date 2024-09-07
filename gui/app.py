import os, glob, pathlib
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import torch

model_dir = "output"


def get_num_devices():
    return 1 + torch.cuda.device_count() if torch.cuda.is_available() else 1


def get_available_models():
    paths = [p.name for p in pathlib.Path(model_dir).iterdir() if p.is_dir()]
    paths.sort()
    return paths


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


# Initialize the Streamlit app
st.title("🪶TitleQuill: Title & Keywords generation with Pre-trained Models🪶")
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

if (
    st.button("Generate Title and Keywords 𓂃🖊")
    and model is not None
    and tokenizer is not None
):
    if abstract_input:
        # Generate the title and keywords using the model
        with st.spinner("Generating... ✨"):
            with torch.inference_mode():
                tokenized_input = tokenize_input(abstract_input, tokenizer)
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

            # Display the generated title and keywords
            st.subheader("🎯 Generated Title and Keywords:")
            st.write(generated_text)
    else:
        st.warning("Please enter an abstract to generate the title and keywords.")

# Footer
st.markdown("---")
st.write(
    "Demo built with [Streamlit](https://streamlit.io/) and [Hugging Face](https://huggingface.co/)."
)
