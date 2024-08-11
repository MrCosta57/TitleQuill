import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from utils.general_utils import seed_everything


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "Qwen/Qwen2-0.5B-Instruct"
    data_dir = "data/OAGKX"
    seed_everything(123)

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForCausalLM.from_pretrained(model_type)

    data_files = glob.glob(f"{data_dir}/*.jsonl")
    dataset = load_dataset("json", data_files=data_files, split="train", streaming=True)

    # Encode input prompt
    template_prompt = "Generate the title and the keywords from the below abstract. Do not add any other information.\
        Output must be in the format:\nTitle: [title]\nKeywords: [keywords]\n\
        The abstract is:"

    for i, data in enumerate(dataset):
        prompt = template_prompt + data["abstract"]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"Example {i + 1}")
        print(f"Generated text:\n{response}")
        print(f"TRUE title:\n{data['title']}")
        print(f"TRUE keywords:\n{data['keywords']}")

        if i >= 1:
            break


if __name__ == "__main__":
    main()
