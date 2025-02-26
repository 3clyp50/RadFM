from torch import nn
from transformers.models.llama import LlamaForCausalLM
from transformers import AutoConfig
from .my_embedding_layer import MyEmbedding
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import tqdm.auto as tqdm
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import numpy as np
import gc
import os

class MultiLLaMAForCausalLM(nn.Module):
    def __init__(self, lang_model_path):  
        super(MultiLLaMAForCausalLM, self).__init__()  
        
        # Check if we're trying to load from Language_files
        if os.path.basename(lang_model_path) == "Language_files":
            print("Detected Language_files path - using config-only initialization")
            # Always use config-only initialization for Language_files
            # This avoids trying to load weights from Language_files
            try:
                print("Loading config from:", lang_model_path)
                config = AutoConfig.from_pretrained(lang_model_path)
                self.lang_model = LlamaForCausalLM(config)
                print("Model initialized from config only - weights will be loaded separately")
            except Exception as e:
                print(f"Error loading config: {e}")
                raise ValueError(f"Failed to load config from {lang_model_path}. Make sure config.json exists.")
        else:
            # For other paths, try normal loading
            try:
                # Use memory-efficient loading with low_cpu_mem_usage=True
                print("Attempting to load LlamaForCausalLM with memory optimization...")
                self.lang_model = LlamaForCausalLM.from_pretrained(
                    lang_model_path,
                    low_cpu_mem_usage=True,  # Enable memory optimization
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use half precision if GPU available
                )
                print("LlamaForCausalLM loaded successfully with memory optimization")
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                print("Falling back to config-only initialization")
                # If loading fails, initialize from config only
                config = AutoConfig.from_pretrained(lang_model_path)
                self.lang_model = LlamaForCausalLM(config)
                print("Model initialized from config only")
        
        # Enable memory optimization techniques
        self.lang_model.gradient_checkpointing_enable()
        self.lang_model.enable_input_require_grads()
        
        # Initialize embedding layer after language model to avoid memory duplication
        self.embedding_layer = MyEmbedding()
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight
        self.hidden_dim = 5120
        self.voc_size = 32000
        
        # Force garbage collection after initialization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def forward(self,lang_x, vision_x, attention_mask, labels, loss_reweight,key_words_query):
        if labels.shape == lang_x.shape:
            self.embedding_layer.flag = 'Text'
            # lang_x = lang_x.to(vision_x.dtype)
            # lang_x = lang_x + torch.zeros(1, dtype=lang_x.dtype, device=lang_x.device, requires_grad=True)
            # vision_x = vision_x + torch.zeros(1, dtype=vision_x.dtype, device=vision_x.device, requires_grad=True) 
            # input_embedding = checkpoint(self.embedding_layer, lang_x, vision_x)
            input_embedding,loss_match= self.embedding_layer(lang_x, vision_x,key_words_query)   # ,loss_matching
            output = self.lang_model(inputs_embeds = input_embedding,attention_mask = attention_mask, labels = labels)
            logits = output['logits']

            loss_reg = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_loss_reweight = loss_reweight[...,1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction = 'none')
                shift_logits = shift_logits.view(-1, self.voc_size)
                shift_labels = shift_labels.view(-1)
                shift_loss_reweight = shift_loss_reweight.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                shift_loss_reweight = shift_loss_reweight.to(shift_logits.device) 
                loss_reg = loss_fct(shift_logits, shift_labels)
                loss_reg = torch.sum(shift_loss_reweight*loss_reg)/torch.sum(shift_loss_reweight)
            loss = loss_reg
            if loss_match!= None:
                loss = 0.8*loss + 0.2*loss_match
            
            logits = output['logits'][..., :-1, :].contiguous().detach()
            total = len(labels)
            predictions = torch.argmax(logits, dim=-1)
            labels = labels[..., 1:].contiguous()
            Acc = torch.sum(torch.all(torch.logical_or(predictions == labels, labels == -100),dim = -1))       
            Accuracy = Acc /total      
            
            return dict(
                # loss_reg = loss_reg,
                # loss_matching = loss_matching,
                logits =  Accuracy,
                loss = output['loss'],
            )
        ### useless for now ignore the folowing codes ###
        # if labels.shape == vision_x.shape:
        #    self.embedding_layer.flag = 'Seg'
        #    input_embedding = self.embedding_layer(lang_x, vision_x)
    
    def generate(self, lang_x, vision_x):
        self.embedding_layer.flag = 'Text'
        with torch.no_grad():
            # Clear cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            input_embedding, _ = self.embedding_layer(lang_x, vision_x) 
            generation = self.lang_model.generate(
                inputs_embeds=input_embedding, 
                max_new_tokens=200,
                top_k=50,
                do_sample=False  # Deterministic generation uses less memory
            )
            
            # Clear cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return generation
