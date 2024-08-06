from dataclasses import dataclass
from functools import partial
from typing import Dict
from torch import Tensor
import torch
from transformers import (
    PreTrainedModel, PreTrainedTokenizer, 
    AlbertModel,     AlbertTokenizer,
    BertModel,       BertTokenizer
)

# Type Alias
Tokens     = Dict[str, Tensor]
Embeddings = Tensor

class Encoder:
    '''
    Class incorporating tokenization and embedding process 
    '''
    
    class EncoderTokenizer:
        ''' 
        Class to wrap around a Hugging Face tokenizer
        The class sets default hyperparameters for the tokenizer
            to perform tokenization on arbitrary text
        '''
        
        # Tokenizer default hyperparameters
        RETURN_TENSORS = 'pt'    # PyTorch tensors
        TRUNCATION     = True    # Truncate sequences
        PADDING        = True    # Pad sequences
        MAX_LENGTH     = 512     # Maximum sequence length
        
        def __init__(self, name: str, tokenizer: 'PreTrainedTokenizer'):
            '''
            Initialize the EncoderTokenizer class

            :param name: Name of the tokenizer
            :type name: str
            :param tokenizer: Hugging Face tokenizer type to be wrapped
            :type tokenizer: PreTrainedTokenizer
            '''
            
            self._name: str = name
            self._tokenizer: PreTrainedTokenizer = tokenizer.from_pretrained(name)
            
        def __str__ (self) -> str: return f'EncoderTokenizer[{self._name}]'
        def __repr__(self) -> str: return str(self)
        
        @property
        def tokenizer(self) -> PreTrainedTokenizer:
            ''' 
            Returns tokenizer with macro hyperparameters set
            
            :return: Tokenizer with macro hyperparameters set
            :rtype: PreTrainedTokenizer
            '''
            
            return partial(
                self._tokenizer, 
                return_tensors = self.RETURN_TENSORS,
                truncation     = self.TRUNCATION,
                padding        = self.PADDING,
                max_length     = self.MAX_LENGTH
            )
        
        def __call__(self, text: str) -> Tokens:
            ''' 
            Tokenize the input text
            :param text: Input text to tokenize
            :type text: str
            :return: Tokenized text
            :rtype: Tokens
            '''
            
            return self.tokenizer(text)
    
    class EncoderModel:
        '''
        Class to wrap around a Hugging Face model.
        
        The class transforms tokenized text into embeddings,
        with the option to disable gradient computation.
        '''
        
        def __init__(self, name: str, model: 'PreTrainedModel'):
            '''
            Initializes an instance of the EncoderModel class.
            
            :param name: The name of the encoder model.
            :type name: str
            :param model: Hugging Face model type to be wrapped.
            :type model: PreTrainedModel
            '''
            
            self._name  : str             = name
            self._model : PreTrainedModel = model.from_pretrained(name)
            
        def __str__ (self) -> str: return f'EncoderModel[{self._name}]'
        def __repr__(self) -> str: return str(self)
        
        @property
        def model(self) -> PreTrainedModel: 
            '''
            Get the wrapped Hugging Face model.
            
            :return: The wrapped Hugging Face model.
            :rtype: PreTrainedModel
            '''
            
            return self._model
        
        def __call__(self, tokens: Tokens, no_grad: bool = True) -> Embeddings:
            '''
            Transforms tokenized text into embeddings using the wrapped Hugging Face model.
            
            :param tokens: The tokenized text.
            :type tokens: Tokens
            :param no_grad: Whether to disable gradient computation, defaults to True
            :type no_grad: bool, optional
            :return: The embeddings of the tokenized text.
            :rtype: Embeddings
            '''
            
            if no_grad:
                with torch.no_grad(): outputs = self.model(**tokens)
            
            else: 
                outputs = self.model(**tokens)
            
            embeddings = outputs.last_hidden_state
            
            return embeddings
    
    def __init__(self, name: str, tokenizer='PreTrainedTokenizer', model='PreTrainedModel'):
        
        self._name      : str = name
        self._tokenizer : self.EncoderTokenizer = self.EncoderTokenizer(name=name, tokenizer=tokenizer)
        self._model     : self.EncoderModel     = self.EncoderModel    (name=name, model=model)
    
    def __str__ (self) -> str: return f'Encoder[{self._tokenizer}; {self._model}]'
    def __repr__(self) -> str: return str(self)
    
    @property
    def tokenizer(self) -> EncoderTokenizer : return self._tokenizer
    
    @property
    def model(self)     -> EncoderModel     : return self._model
    
    def __call__(self, text: str, no_grad: bool = True) -> Tensor:
        
        tokens     = self._tokenizer(text)
        embeddings = self._model(tokens, no_grad)
        
        return embeddings


ENCODERS: Dict[str, Encoder]= {
    
    'albert': Encoder(
        name      = 'albert-base-v2',
        model     = AlbertModel    ,
        tokenizer = AlbertTokenizer
    ),
    
    'bert': Encoder(
        name      = 'bert-base-uncased',
        model     = BertModel    ,
        tokenizer = BertTokenizer
    )

}