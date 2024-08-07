# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import Tensor

from titlequill.dataset      import OAGKXRawDataset, OAGKXItem
from titlequill.utils.logger import Logger, SilentLogger
from script.utils.settings   import DATASET_DIR
from typing import Any, Dict, Literal, Tuple, Type


T5ModelNames = Literal['flan-t5-base', 'flan-t5-large', 'flan-t5-xxl']

class T5FlanTitleKeywordGenerator:
    
    _KEYWORDS_PROMPT = 'keywords list'
    _TITLE_PROMPT    = 'title_summary'
    
    GENERATION_TYPE = Literal['title', 'keywords']
    
    def __init__(
        self, 
        name: T5ModelNames | str,
        logger: Logger = SilentLogger()
    ):
        
        self._tokenizer = T5Tokenizer               .from_pretrained(name)
        self._generator = T5ForConditionalGeneration.from_pretrained(name)
        
        self._logger = logger
    
    def __str__(self) -> str: return f'T5FlanTitleKeywordGenerator[{self._generator.name_or_path}]'
    def __repr__(self) -> str: return str(self)
    
    # --- Pipeline ---
    
    def _tokenize(self, item: OAGKXItem, gen_type: GENERATION_TYPE, **args) -> Dict[str, Tensor]:
        
        match gen_type:
            
            case 'title'    : input_text = f'{self._TITLE_PROMPT}:    {item.title}'
            case 'keywords' : input_text = f'{self._KEYWORDS_PROMPT}: {item.keywords}'
            case _          : raise ValueError(f'Invalid generation type: {gen_type}. Must be either {self.GENERATION_TYPE}.')
        
        return self._tokenizer(input_text, return_tensors='pt', **args)
    
    def _generate(self, tokens: Dict[str, Tensor], **args) -> Dict[str, Tensor]:
        
        return self._generator.generate(
            input_ids      = tokens["input_ids"],
            attention_mask = tokens["attention_mask"],
            **args
        )
    
    def _decode(self, tokens: Dict[str, Tensor], **args) -> str:
        
        return self._tokenizer.decode(tokens, **args)
    
    def generate(self, item: OAGKXItem, gen_type: GENERATION_TYPE, **kwargs) -> str:
        
        token_kwargs  = kwargs.get('token_kwargs',  {})
        gen_kwargs    = kwargs.get('gen_kwargs',    {})
        decode_kwargs = kwargs.get('decode_kwargs', {})
        
        
        tokens     = self._tokenize(item  =item,       **token_kwargs,  gen_type=gen_type)
        tokens_gen = self._generate(tokens=tokens,     **gen_kwargs   )[0]
        out        = self._decode  (tokens=tokens_gen, **decode_kwargs)
        
        return out
    
    def __call__(self, item: OAGKXItem, **kwargs) -> Tuple[str, str]:
        
        return (
            self.generate(item=item, gen_type='title',    **kwargs),
            self.generate(item=item, gen_type='keywords', **kwargs)
        )


def main():
    
    ARGS = {
        'token_kwargs' : {},
        'gen_kwargs'   : {
            'max_length'     : 128,
            'num_beams'      : 4,
            'early_stopping' : True
        },
        'decode_kwargs': {'skip_special_tokens' : True}
    }
    N = 10
    
    
    logger    = Logger()
    dataset   = OAGKXRawDataset(dataset_dir=DATASET_DIR)
    generator = T5FlanTitleKeywordGenerator(name='google/flan-t5-base', logger=logger)

    for i in range(N):
        
        print(f'ITEM {i}')
    
        item = dataset[i]
        
        print("   TRUE")
        print(f'    - title:    {item.title}')
        print(f'    - keywords: {item.keywords}')
        
        title, keywords = generator(item)
        
        print("   PRETICTED")
        print(f'    - title:    {title}')
        print(f'    - keywords: {keywords}')
        print()

if __name__ == "__main__": main()   


