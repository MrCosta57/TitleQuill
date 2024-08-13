from functools import partial
import re
from typing import Dict, List, Tuple

from src.datamodule.custom_training_loop import Batch, CustomTrainingLoop, DatasetItem, ModelOut, get_collate_function_with_preprocessing, load_oagkx_dataset

from torch import Tensor
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DATASET_DIR = "data/OAGKX"
MODEL_TYPE  = "google/flan-t5-small"
LR          = 1e-4

tokenizer_input_args  = {'padding': True, 'max_length': 512, 'truncation': True} # NOTE: return_tensors = "pt" is already included
tokenizer_label_args  = {'padding': True, 'max_length': 128, 'truncation': True} # NOTE: return_tensors = "pt" is already included
dataloader_args       = {'batch_size': 2, 'shuffle': False}

# Filter function
def at_least_one_keyword(items: Dict[str, List[str]]) -> bool:
    
    return [bool(re.match(r"\w+", elem)) for elem in items["keywords"]]

# Define input and labels
def title_and_keywords_together(batch: List[DatasetItem], task: str = 'title') -> Tuple[List[str], List[str]]:
    
    def title_and_keywords_together_aux(item: DatasetItem) -> Tuple[str, str]:
        
        input_format = "{task}: {a}"
        label_format = "Title: {t}, Keywords: {k}"
        
        inputs = input_format.format(task=task, a=item["abstract"])
        labels = label_format.format(t=item["title"], k=item["keywords"])
        
        return inputs, labels
    
    inputs, labels = zip(*[title_and_keywords_together_aux(item) for item in batch])
    
    return inputs, labels

# Loss function
def loss_fn(batch: Batch, outputs: ModelOut) -> Tensor:
    
    loss = outputs.loss
    
    return loss

if __name__ == "__main__":
    
    tokenizer = AutoTokenizer        .from_pretrained(MODEL_TYPE)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TYPE)
    optimizer = AdamW(model.parameters(), lr=LR)
    
    dataset = load_oagkx_dataset(
        data_dir=DATASET_DIR,
        split="train",
        streaming=True,
        filter_fn=at_least_one_keyword,
    )
    
    collate_fn = get_collate_function_with_preprocessing(
        input_labels_fn=partial(title_and_keywords_together, task='title'),
        tokenizer=tokenizer,
        tokenizer_input_args=tokenizer_input_args,
        tokenizer_target_args=tokenizer_label_args,
    )
    
    training_loop = CustomTrainingLoop(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        collate_fn=collate_fn,
        loss_fn=loss_fn,
        log_fn=lambda model, batch, outputs: print("Here"),
        epochs=1,
        dataloader_args=dataloader_args,
    )
    
    training_loop.train()
    
    
