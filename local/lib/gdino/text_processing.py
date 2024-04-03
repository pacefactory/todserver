#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Modified/simplified from Grounding DINO:
@ https://github.com/IDEA-Research/GroundingDINO
'''


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
from torch import nn

import transformers as HFace

from .model_datatypes import TextEncoding


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes


HFace.logging.set_verbosity_info()

class TextEncoder(nn.Module):
    
    '''
    This is a simplified version of the text processing stages from the Grounded DINO model
    See the .forward(...) method for details about what is actually happening during text processing
    '''
    
    # .................................................................................................................
    
    def __init__(self, output_size = 256, text_encoder_type = "bert-base-uncased", *, verbose = False):
        
        # Inherit from parent class
        super().__init__()
        
        # Change logging if needed
        HFace.logging.set_verbosity_error()
        if verbose:
            HFace.logging.set_verbosity_info()
        
        # Set up pretrained models
        self._output_size = output_size
        self._text_encoder_type = text_encoder_type
        self.tokenizer_model = HFace.AutoTokenizer.from_pretrained(text_encoder_type)
        self.bert = HFace.BertModel(HFace.BertConfig(), add_pooling_layer = False)
        # self.bert = HFace.BertModel.from_pretrained(text_encoder_type) # ALT LOADING - Uses cached model weights
        
        # Set up extra mapping layer for bert output to later stage inputs
        bert_output_size = self.bert.config.hidden_size
        self.feat_map = torch.nn.Linear(bert_output_size, output_size, bias=True)
        
        # Pre-compute special tokens for text processing
        self._special_tokens_list = []
        with torch.no_grad():
            self._special_tokens_list = self.tokenizer_model.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        
        # Assume we're running inference only
        self.eval()
    
    # .................................................................................................................
    
    def forward(self, input_text_str: str) -> TextEncoding:
        
        # Convert text to tokens and then encode with bert
        text_token_mask, input_ids, attention_mask, token_type_ids, position_ids = self.run_tokenizer(input_text_str)
        bert_output = self.run_bert(input_ids, attention_mask, token_type_ids, position_ids)
        encoded_text = self.feat_map.forward(bert_output)
        
        # Generate reverse-lookup of token ids to text
        # -> This is used to associate text labels with bbox detections
        input_ids = input_ids.squeeze()
        idx_to_text_lookup = [self.tokenizer_model.decode(tokid) for tokid in input_ids]
        
        # Build final text output
        encoding = TextEncoding(
            text_prompt=input_text_str,
            features=encoded_text,
            token_mask=text_token_mask,
            attn_mask=attention_mask,
            position_ids=position_ids,
            input_ids=input_ids,
            idx_to_text_lookup=idx_to_text_lookup,
        )
        
        return encoding
    
    # .................................................................................................................
    
    def run_tokenizer(self, text_input):
        
        '''
        Function which performs text-to-token processing
        Returns tuple:
            text_token_mask, input_ids, text_self_attention_masks, token_type_ids, position_ids
        
        Sizing info, assuming we get N tokens:
            text_token_mask:            bool tensor of shape [1, N]
            input_ids:                  int64 tensor of shape [1, N]
            text_self_attention_masks:  bool tensor of shape [1, N, N]
            token_type_ids:             int64 tensor of shape [1, N]  (mostly zeros)
            position_ids:               int64 tensor of shape [1, N]  (counting: [0, 0, 1, 2, 3, 4, ..., 0])
        '''
        
        # Clean text & add trailing period if needed
        clean_text_input = text_input.strip().lower()
        clean_text_input = clean_text_input if clean_text_input.endswith(".") else clean_text_input + "."
        
        # Process input text
        # -> Returns a dictionary with key: 'input_ids'. 'token_type_ids', 'attention_mask'
        # -> Each value is a vector of the same length, equal to the number of tokens (roughly 2 + number of words)
        with torch.no_grad():
            text_tokens = self.tokenizer_model(clean_text_input, padding="longest", return_tensors="pt")
        
        # Generate positional encodings + attention masks (for special characters?)
        text_self_attention_masks, position_ids = \
            generate_masks_with_special_tokens_and_transfer_map(text_tokens, self._special_tokens_list)
        
        # Extract text embeddings (??? Taken directly from original code)
        text_token_mask = text_tokens.attention_mask.bool()  # bs, num_tokens
        token_type_ids = text_tokens["token_type_ids"]
        input_ids = text_tokens["input_ids"]
        
        # Error if we get too much data
        num_tokens = position_ids.shape[1]
        need_truncate_text = num_tokens > self._output_size
        if need_truncate_text:
            raise IOError("Error! Text token length exceeded maximum capability: {} tokens".format(self._output_size))
        
        return text_token_mask, input_ids, text_self_attention_masks, token_type_ids, position_ids
    
    # .................................................................................................................
    
    def run_bert(self, input_ids, attention_mask, token_type_ids, position_ids):
        
        '''
        The bert model consists of 3 major blocks:
            1. Embedding
            2. Encoder
            3. Pooler
            
        - The embedding block takes in text tokens and generates per-word embeddings
        - The encoder block takes in the embedded text and generates another set of embeddings
          which take into account the entire sentence (i.e. context-aware or sentence-aware embeddings)
        - The pooler takes in the encoder output and generates an embedding for the entire sentence
          -> note, the pooler may be optional, depending on how BERT is initialized!
        
        For use in the GDINO model, we only care about the encoder outputs, so we don't run the pooler
        -> This function assumes that the input text has already passed through a tokenizer
        '''
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        position_ids = position_ids.to(device)
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        input_shape = input_ids.size()
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_shape)
        # -> This seems to do something very simple. See 'ModuleUtilsMixin' for exact implementation
        # -> Just does:
        # 1. output = attention_mask[:, None, :, :]
        # 2. output = output.to(dtype=dtype)  # fp16 compatibility
        # 3. output = (1.0 - output) * torch.finfo(dtype).min
        
        # Generate token embeddings (?). Gives tensor of shape: [1, num_input_ids, 768]
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds = None,
            past_key_values_length = 0,
        )
        
        # Calculate encoded outputs
        #head_mask = [None] * self.bert.config.num_hidden_layers
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # Bert model output
        sequence_output = encoder_outputs[0]
        
        return sequence_output
    
    # .................................................................................................................
    
    def decode_text(self, logit_01, text_token_ids, text_threshold = 0.25) -> str:
        
        '''
        Function used to map from text-to-image logits back to text
        (i.e. to assign text labels with bounding boxes)
        
        Expects a single logit (with values between 0.0 and 1.0) along
        with the encoded text token ids
        
        Returns:
            text_label (for the given logit)
        '''
        
        text_filter_mask = logit_01 > text_threshold
        non_zero_idx = text_filter_mask.nonzero(as_tuple=True)[0].tolist()
        
        label = self.tokenizer_model.decode([text_token_ids[i] for i in non_zero_idx])
        if label == "":
            label = "unknown"
        
        return label
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def generate_masks_with_special_tokens_and_transfer_map(text_tokens, special_tokens_list):
    
    """
    Taken from:
        https://github.com/IDEA-Research/GroundingDINO
        groundingdino/model/GroundingDINO/bertwarper.py
    
    Generate attention mask between each pair of special tokens
    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    
    input_ids = text_tokens["input_ids"]
    in_device = input_ids.device
    bs, num_token = input_ids.shape
    
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token)).bool()
    for token in special_tokens_list:
        special_tokens_mask |= (input_ids == token)

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = torch.eye(num_token, device=in_device).bool().unsqueeze(0).repeat(bs, 1, 1)
    position_ids = torch.zeros((bs, num_token), device=in_device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(0, col - previous_col)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)

# .....................................................................................................................

