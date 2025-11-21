import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        nl_path = os.path.join(data_folder, f'{split}.nl')
        with open(nl_path, 'r') as f:
            self.nl_queries = [line.strip() for line in f.readlines()]
        
        # Load SQL queries (not available for test set)
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            with open(sql_path, 'r') as f:
                self.sql_queries = [line.strip() for line in f.readlines()]
        else:
            self.sql_queries = None
        
        # Tokenize encoder inputs (natural language)
        self.encoder_inputs = []
        for nl in self.nl_queries:
            tokens = tokenizer(nl, max_length=512, truncation=True, 
                             padding=False, return_tensors=None)
            self.encoder_inputs.append(tokens['input_ids'])
        
        # Tokenize decoder inputs/targets (SQL)
        self.decoder_inputs = []
        self.decoder_targets = []
        
        if split != 'test':
            for sql in self.sql_queries:
                tokens = tokenizer(sql, max_length=512, truncation=True,
                                 padding=False, return_tensors=None)
                
                # Decoder input: start with pad token (BOS), then SQL tokens (excluding last)
                decoder_input = [tokenizer.pad_token_id] + tokens['input_ids'][:-1]
                
                # Decoder target: SQL tokens as-is
                decoder_target = tokens['input_ids']
                
                self.decoder_inputs.append(decoder_input)
                self.decoder_targets.append(decoder_target)
    
    def __len__(self):
        # TODO
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        # TODO
        if self.split == 'test':
            return {
                'encoder_input': self.encoder_inputs[idx],
                'decoder_input': [],
                'decoder_target': []
            }
        else:
            return {
                'encoder_input': self.encoder_inputs[idx],
                'decoder_input': self.decoder_inputs[idx],
                'decoder_target': self.decoder_targets[idx]
            }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # return [], [], [], [], []
    encoder_inputs = [torch.tensor(item['encoder_input']) for item in batch]
    decoder_inputs = [torch.tensor(item['decoder_input']) for item in batch]
    decoder_targets = [torch.tensor(item['decoder_target']) for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    # Initial decoder input (first token for generation)
    initial_decoder_inputs = torch.tensor([[item['decoder_input'][0]] for item in batch])
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # return [], [], []
    encoder_inputs = [torch.tensor(item['encoder_input']) for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder input (start token)
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    initial_decoder_inputs = torch.tensor([[tokenizer.pad_token_id]] * len(batch))
    
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_x, train_y, dev_x, dev_y, test_x