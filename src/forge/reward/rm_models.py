import logging
from typing import List, Optional, Dict, Tuple, Union
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    pipeline
)
import os
from tqdm import tqdm
from transformers import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

def fix_transformers_compatibility():
    """The QRM Model we use requires this import but our transformers version 
    has deprecated this. We manually add the missing constant."""
    try:
        import transformers.models.llama.modeling_llama as llama_modeling
        if not hasattr(llama_modeling, 'LLAMA_INPUTS_DOCSTRING'):
            llama_modeling.LLAMA_INPUTS_DOCSTRING = """
            Args:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    Indices of input sequence tokens in the vocabulary.
            """
    except Exception:
        pass

fix_transformers_compatibility()

class BaseRewardModel:
    """Base class for reward models with proper resource management"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048,
        dtype: torch.dtype = torch.float32
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        
    def load(self) -> None:
        """Load model and tokenizer - to be implemented by child classes"""
        raise NotImplementedError
        
    def unload(self) -> None:
        """Safely unload model and free GPU memory"""
        try:
            if self.model is not None:
                # TODO: CPU transfer was causing task to get killed during unloading
                # if hasattr(self.model, 'cpu'):
                #     self.model.cpu()
                del self.model
                self.model = None
                
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error unloading model: {e}")
            
    def prepare_inputs(self, instruction: str, response: str) -> str:
        """Format inputs for the model - can be overridden by child classes"""
        return f"Instruction: {instruction}\nResponse: {response}"
        
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Get reward scores for multiple instruction-response pairs"""
        if len(instructions) != len(responses):
            raise ValueError("Number of instructions and responses must match")
            
        if not instructions:
            return []
            
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            # Process in batches
            for i in tqdm(range(0, len(instructions), batch_size)):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Prepare inputs using a helper (child classes may override)
                inputs = [
                    self.prepare_inputs(inst, resp)
                    for inst, resp in zip(batch_instructions, batch_responses)
                ]
                
                # Tokenize (do not pass torch_dtype here)
                encoded = self.tokenizer(
                    inputs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                    return_tensors="pt"
                )
                # Move tokenized inputs to device (keep their native integer dtype)
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Use autocast to ensure model operations are performed in self.dtype
                with torch.no_grad():
                    # The autocast block ensures consistent dtype usage for model computation
                    with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                        outputs = self.model(**encoded)
                    # Convert logits to float32 for further standardized processing
                    scores = outputs.logits.to(torch.float32).cpu()
                
                # Process and collect scores (child classes define _process_scores)
                batch_scores = self._process_scores(scores)
                all_scores.extend(batch_scores)
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in reward scoring: {e}")
            return [None] * len(instructions)
            
    def _process_scores(self, scores: torch.Tensor) -> List[float]:
        """Process raw scores into list of floats - to be implemented by child classes"""
        raise NotImplementedError

class GRMModel(BaseRewardModel):
    """Implementation of GRM reward model following official demo"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.float16
        )
        self.model_name = "Ray2333/GRM-Llama3-8B-rewardmodel-ft"
        
    def load(self) -> None:
        try:
            logging.info(f"Loading GRM model on {self.device}...")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading GRM model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            # Process in batches
            for i in tqdm(range(0, len(instructions), batch_size), desc="GRM scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                batch_scores = []
                for prompt, response in zip(batch_instructions, batch_responses):
                    # Format as per demo
                    message = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    message_template = self.tokenizer.apply_chat_template(message, tokenize=False)
                    
                    # Tokenize following demo
                    tokens = self.tokenizer.encode_plus(
                        message_template,
                        padding='longest',
                        truncation=True,
                        return_tensors="pt"
                    )

                    # Move inputs to device
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}
                    
                    # Get reward following demo implementation
                    with torch.no_grad():
                        reward_tensor = self.model(
                            input_ids=tokens["input_ids"],
                            attention_mask=tokens["attention_mask"]
                        )[0]
                        reward = reward_tensor.cpu().detach().item()
                        batch_scores.append(reward)
                
                all_scores.extend(batch_scores)
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in GRM scoring: {e}")
            return [None] * len(instructions)

class QwenPRMModel(BaseRewardModel):
    """Implementation of QwenPRM model following official demo"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "Qwen/Qwen2.5-Math-PRM-7B"
        self.step_scores = {}  # Store step-wise scores
        
    def load(self) -> None:
        try:
            logging.info(f"Loading QwenPRM model on {self.device}...")
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading QwenPRM model: {e}")
            self.unload()
            raise
            
    def make_step_rewards(self, logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
        """Helper function from demo to calculate step rewards"""
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> Dict[str, List[float]]:
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            self.step_scores = {}  # Reset step scores
            
            # Process in batches
            for i in tqdm(range(0, len(instructions), batch_size), desc="Qwen PRM scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                for batch_idx, (prompt, response) in enumerate(zip(batch_instructions, batch_responses)):
                    overall_idx = i + batch_idx
                    
                    # Split response into steps
                    steps = [step.strip() for step in response.split("\n") if step.strip()]
                    
                    # Format messages following demo
                    messages = [
                        {"role": "system", "content": "Please reason step by step."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}
                    ]
                    
                    # Apply chat template
                    conversation = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    
                    # Tokenize
                    input_ids = self.tokenizer.encode(conversation, return_tensors="pt").to(self.device)
                    
                    # Get rewards following demo implementation
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids)
                        step_sep_id = self.tokenizer.encode("<extra_0>")[0]
                        token_masks = (input_ids == step_sep_id).to(self.device)
                        step_rewards = self.make_step_rewards(outputs[0], token_masks)
                        
                        if step_rewards and step_rewards[0]:
                            # Store step scores
                            self.step_scores[overall_idx] = step_rewards[0]
                            # Calculate min, max, and avg scores
                            min_score = min(step_rewards[0])
                            max_score = max(step_rewards[0])
                            avg_score = sum(step_rewards[0]) / len(step_rewards[0])
                            all_scores.append({
                                'min_scores': min_score,
                                'max_scores': max_score,
                                'avg_scores': avg_score
                            })
                        else:
                            self.step_scores[overall_idx] = []
                            all_scores.append({
                                'min_scores': None,
                                'max_scores': None,
                                'avg_scores': None
                            })
            
            # Reorganize scores into separate lists
            return {
                'min_scores': [score['min_scores'] for score in all_scores],
                'max_scores': [score['max_scores'] for score in all_scores],
                'avg_scores': [score['avg_scores'] for score in all_scores]
            }
            
        except Exception as e:
            logging.error(f"Error in QwenPRM scoring: {e}")
            return {
                'min_scores': [None] * len(instructions),
                'max_scores': [None] * len(instructions),
                'avg_scores': [None] * len(instructions)
            }
            
    def get_step_scores(self) -> Dict[int, List[float]]:
        """Return the stored step-wise scores"""
        return self.step_scores
        
    def unload(self) -> None:
        """Safely unload model and free GPU memory"""
        self.step_scores = {}  # Clear stored scores
        super().unload()

class SkyworksModel(BaseRewardModel):
    """Implementation of Skyworks reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
        
    def load(self) -> None:
        """Load Skyworks model and tokenizer"""
        try:
            logging.info(f"Loading Skyworks model on {self.device}...")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
                num_labels=1,
            )
            
            # Now enable flash attention
            if hasattr(self.model, 'enable_flash_attention'):
                self.model.enable_flash_attention()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                model_max_length=self.max_input_length
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading Skyworks model: {e}")
            self.unload()
            raise
            
    def _process_scores(self, scores: torch.Tensor) -> List[float]:
        """Process Skyworks scores into list of floats"""
        return [float(score[0]) for score in scores.numpy()]
    
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="Skyworks scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Format as chat messages
                batch_messages = [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    for prompt, response in zip(batch_instructions, batch_responses)
                ]
                
                # Apply chat template to all conversations
                batch_inputs = [
                    self.tokenizer.apply_chat_template(messages, tokenize=False)
                    for messages in batch_messages
                ]
                
                # Batch tokenize with padding
                tokens = self.tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                          
                with torch.no_grad():
                    outputs = self.model(**tokens)
                    scores = outputs.logits[:, 0].float().cpu().tolist()
                    
                all_scores.extend(scores)
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in Skyworks scoring: {e}")
            return [None] * len(instructions)


class QRMModel(BaseRewardModel):
    """Implementation of QRM reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "nicolinho/QRM-Llama3.1-8B-v2"
        self.attributes = [
            'helpsteer-helpfulness',
            'helpsteer-correctness',
            'helpsteer-coherence',
            'helpsteer-complexity',
            'helpsteer-verbosity'
        ]
        
    def load(self) -> None:
        try:
            logging.info(f"Loading QRM model on {self.device}...")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading QRM model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None,
        return_quantiles: bool = False
    ) -> Union[List[float], Dict[str, Union[List[float], List[List[float]]]]]:
        """Get reward scores for multiple instruction-response pairs.
        
        Args:
            instructions: List of instruction strings
            responses: List of response strings
            batch_size: Optional batch size override
            return_quantiles: If True, return quantile estimates along with scores
            
        Returns:
            If return_quantiles is False:
                List of reward scores
            If return_quantiles is True:
                Dictionary containing:
                - 'scores': List of reward scores
                - 'quantiles': List of quantile estimates for each input
                - 'attributes': List of attribute names
        """
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            all_quantiles = [] if return_quantiles else None
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="QRM scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Format as chat messages
                batch_messages = [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    for prompt, response in zip(batch_instructions, batch_responses)
                ]
                
                # Apply chat template and tokenize
                batch_inputs = [
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_tensors="pt"
                    )
                    for messages in batch_messages
                ]
                
                # Get scores and optionally quantiles
                with torch.no_grad():
                    for input_ids in batch_inputs:
                        # Move inputs to device
                        input_ids = input_ids.to(self.device)
                        outputs = self.model(input_ids)
                        score = outputs.score.float().cpu().item()
                        all_scores.append(score)
                        
                        if return_quantiles:
                            quantiles = outputs.reward_quantiles.float().cpu().tolist()
                            all_quantiles.append(quantiles)
                
            if return_quantiles:
                return {
                    'scores': all_scores,
                    'quantiles': all_quantiles,
                    'attributes': self.attributes
                }
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in QRM scoring: {e}")
            if return_quantiles:
                return {
                    'scores': [None] * len(instructions),
                    'quantiles': [None] * len(instructions),
                    'attributes': self.attributes
                }
            return [None] * len(instructions)

class URMModel(BaseRewardModel):
    """Implementation of URM reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.float16
        )
        self.model_name = "LxzGordon/URM-LLaMa-3.1-8B"
        
    def load(self) -> None:
        try:
            logging.info(f"Loading URM model on {self.device}...")
            
            # Load model with optimizations
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=self.dtype,
                use_cache=True
            )
            
            # Load tokenizer with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                model_max_length=self.max_input_length,
                padding_side="left"  # More efficient for causal models
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.eval()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"Error loading URM model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            # Pre-format all messages
            all_messages = [
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                for prompt, response in zip(instructions, responses)
            ]
            
            # Process in batches
            for i in tqdm(range(0, len(all_messages), batch_size), desc="URM scoring"):
                batch_messages = all_messages[i:i + batch_size]
                
                # Apply chat template and tokenize in one go
                batch_inputs = [
                    self.tokenizer.apply_chat_template(msgs, tokenize=False)
                    for msgs in batch_messages
                ]
                
                # Efficient batch tokenization
                encoded = self.tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                    return_tensors="pt"
                )

                # Move tokenized inputs to the correct device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Process batch with updated autocast
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        outputs = self.model(**encoded)
                        batch_scores = outputs.logits[:, 0].float().cpu().tolist()
                    
                all_scores.extend(batch_scores)
                
                # Clear CUDA cache less frequently
                if i % 5000 == 0:
                    torch.cuda.empty_cache()
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in URM scoring: {e}")
            return [None] * len(instructions)

def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer

def get_reward_model(base_causal_model, base_llm_model, is_general_preference=False, add_prompt_head=False, value_head_dim=2):
    class CustomRewardModel(base_causal_model):
        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            if not is_general_preference:
                self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
            else: 
                self.value_head = nn.Linear(config.hidden_size, value_head_dim, bias=False) 
                if add_prompt_head:
                    self.prompt_head = nn.Linear(config.hidden_size, value_head_dim // 2, bias=False) 
            
            self.is_general_preference = is_general_preference    
            self.post_init()

        def custom_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            
            if not self.is_general_preference:
                values = self.value_head(last_hidden_states).squeeze(-1)
                if self.training:
                    reward = values[:, -1]
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    reward = values.gather(dim=1, index=eos_indices).squeeze(1)
            else:
                values = self.value_head(last_hidden_states)
                if self.training:
                    reward = values[:, -1, :]
                    reward = F.normalize(reward, p=2, dim=-1)
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)
                    eos_indices = eos_indices.unsqueeze(1)
                    reward_list = []
                    for dim in range(values.size(-1)):
                        reward_list.append(values[:,:,dim].gather(dim=1, index=eos_indices))
                    reward = torch.cat(reward_list, dim=1)
                    reward = F.normalize(reward, p=2, dim=-1)
            
            if return_output:
                return reward, outputs
            return reward, None
            
    return CustomRewardModel

class GPMPipeline:
    def __init__(
        self, 
        model_name_or_path, 
        tokenizer_name_or_path,
        device=torch.device("cuda:0"), 
        is_general_preference=True,
        add_prompt_head=True,
        value_head_dim=2,
        bf16=True,
        truncation=True,
        max_length=4096,
        padding=True,
    ):
        self.device = device
        self.is_general_preference = is_general_preference
        self.add_prompt_head = add_prompt_head
        self.value_head_dim = value_head_dim
        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        base_class = AutoModel._model_mapping[type(config)]
        base_causal_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
        cls_class = get_reward_model(
            base_causal_class, 
            base_class,
            is_general_preference,
            add_prompt_head,
            value_head_dim
        )

        self.model = cls_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map="auto"
        )
        self.tokenizer = get_tokenizer(tokenizer_name_or_path, self.model)
        self.tokenizer.truncation_side = "right"
        
        self.model.eval()

    
    def __call__(self, samples: List[List[Dict[str, str]]], return_prompt=False):
        input_texts = [
            self.tokenizer.apply_chat_template(sample, tokenize=False)
            for sample in samples
        ]

        inputs = self.tokenizer(
            input_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs["input_ids"][:, -1] = self.tokenizer.eos_token_id
        inputs["attention_mask"][:, -1] = 1
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            rewards, outputs = self.model.custom_forward(**inputs, return_output=return_prompt)

        if return_prompt:
            # First, render just the prefixes
            prompt_texts = [
                self.tokenizer.apply_chat_template([sample[0]], tokenize=False)
                for sample in samples
            ]

            # Turn those length differences into plain ints
            chosen_response_len_list = []
            for idx in range(len(input_texts)):
                prompt_tokens = self.tokenizer(
                    prompt_texts[idx],
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                chosen_tokens = self.tokenizer(
                    input_texts[idx],
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_len   = prompt_tokens["attention_mask"].sum().item()
                chosen_len   = chosen_tokens["attention_mask"].sum().item()
                chosen_response_len_list.append(int(chosen_len - prompt_len))

            # Now build a proper LongTensor
            chosen_response_len = torch.tensor(
                chosen_response_len_list,
                dtype=torch.long,
                device=outputs["last_hidden_state"].device
            ).view(-1, 1)

            # Gather the hidden state right at the end of the prompt
            chosen_last_hidden = outputs["last_hidden_state"]
            # sequence_length - response_length - 1  → index of the <eos> or last prompt token
            prompt_end_index = chosen_last_hidden.size(1) - chosen_response_len - 1
            prompt_end_index_expanded = prompt_end_index.unsqueeze(-1).expand(
                -1, -1, chosen_last_hidden.size(-1)
            )
            prompt_hidden_state = torch.gather(
                chosen_last_hidden,
                dim=1,
                index=prompt_end_index_expanded
            ).squeeze(1)

            return rewards, prompt_hidden_state

        return rewards

class GPMModel(BaseRewardModel):
    """Implementation of GPM reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "general-preference/GPM-Llama-3.1-8B"
        self.value_head_dim = 6
        
    def load(self) -> None:
        try:
            logging.info(f"Loading GPM model on {self.device}...")
            
            # Initialize pipeline with correct settings
            self.pipeline = GPMPipeline(
                model_name_or_path=self.model_name,
                tokenizer_name_or_path=self.model_name,
                device=self.device,
                is_general_preference=True,
                add_prompt_head=True,
                value_head_dim=self.value_head_dim,
                bf16=True,
                max_length=self.max_input_length
            )
            
        except Exception as e:
            logging.error(f"Error loading GPM model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        try:
            if not hasattr(self, 'pipeline'):
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="GPM scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Format as chat messages
                batch_messages = [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    for prompt, response in zip(batch_instructions, batch_responses)
                ]
                
                # Get rewards and prompt hidden states
                rewards, prompt_hidden = self.pipeline(batch_messages, return_prompt=True)
                
                # Convert high-dimensional rewards to scalar scores
                scores = torch.norm(rewards, p=2, dim=1).cpu().tolist()
                all_scores.extend(scores)
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in GPM scoring: {e}")
            return [None] * len(instructions)
            
    def unload(self) -> None:
        """Unload model from GPU memory"""
        if hasattr(self, 'pipeline'):
            del self.pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class GRMLlama32Model(BaseRewardModel):
    """Implementation of GRM-Llama32 reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.float16
        )
        self.model_name = "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"
        
    def load(self) -> None:
        try:
            logging.info(f"Loading GRM-Llama32 model on {self.device}...")
            config = {
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": self.dtype,
                "low_cpu_mem_usage": True,
            }
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                torch_dtype=self.dtype,
                model_max_length=self.max_input_length
            )
            self.model.eval()
        except Exception as e:
            logging.error(f"Error loading GRM-Llama32 model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="GRM-Llama32 scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                batch_messages = [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    for prompt, response in zip(batch_instructions, batch_responses)
                ]
                
                # Apply chat template to all conversations
                batch_inputs = [
                    self.tokenizer.apply_chat_template(messages, tokenize=False)
                    for messages in batch_messages
                ]
                
                # Batch tokenize
                tokens = self.tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                
                # Move inputs onto GPU
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                with torch.no_grad():
                    outputs = self.model(**tokens)
                    scores = outputs.logits[:, 0].float().cpu().tolist()
                    
                all_scores.extend(scores)
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in GRM-Llama32 scoring: {e}")
            return [None] * len(instructions)
        
class OffsetBiasModel(BaseRewardModel):
    """Implementation of OffsetBias reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16  # OffsetBias uses bfloat16
        )
        self.model_name = "NCSOFT/Llama-3-OffsetBias-RM-8B"
        
    def load(self) -> None:
        """Load OffsetBias model and tokenizer"""
        try:
            logging.info(f"Loading OffsetBias model on {self.device}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Initialize pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device_map="auto",
                tokenizer=self.tokenizer,
                model_kwargs={"torch_dtype": self.dtype}
            )
            
        except Exception as e:
            logging.error(f"Error loading OffsetBias model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Get reward scores for multiple instruction-response pairs"""
        try:
            if not hasattr(self, 'pipeline'):
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            # Pipeline kwargs
            pipe_kwargs = {
                "return_all_scores": True,
                "function_to_apply": "none",
                "batch_size": batch_size
            }
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="OffsetBias scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Format as chat messages
                batch_messages = [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    for prompt, response in zip(batch_instructions, batch_responses)
                ]
                
                # Apply chat template and remove BOS token
                batch_inputs = [
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    ).replace(self.tokenizer.bos_token, "")
                    for messages in batch_messages
                ]
                
                # Get scores through pipeline
                outputs = self.pipeline(batch_inputs, **pipe_kwargs)
                scores = [output[0]["score"] for output in outputs]
                all_scores.extend(scores)
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in OffsetBias scoring: {e}")
            return [None] * len(instructions)
            
    def unload(self) -> None:
        """Unload model from GPU memory"""
        if hasattr(self, 'pipeline'):
            del self.pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class GRMGemmaModel(BaseRewardModel):
    """Implementation of GRM-Gemma reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.float32  # GRM-Gemma uses float32
        )
        self.model_name = "Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
        
    def load(self) -> None:
        try:
            logging.info(f"Loading GRM-Gemma model")
            config = {
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": self.dtype,
                "low_cpu_mem_usage": True,
            }
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                model_max_length=self.max_input_length
            )
            self.model.eval()
        except Exception as e:
            logging.error(f"Error loading GRM-Gemma model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="GRM-Gemma scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Format as chat messages
                batch_messages = [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    for prompt, response in zip(batch_instructions, batch_responses)
                ]
                
                # Apply chat template to all conversations
                batch_inputs = [
                    self.tokenizer.apply_chat_template(messages, tokenize=False)
                    for messages in batch_messages
                ]
                
                # Batch tokenize
                tokens = self.tokenizer(
                    batch_inputs,
                    padding='longest',
                    truncation=True,
                    return_tensors="pt"
                )
                
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=tokens["input_ids"],
                        attention_mask=tokens["attention_mask"]
                    )
                    # Extract scores from logits
                    scores = outputs.logits[:, 0].float().cpu().tolist()
                    
                all_scores.extend(scores)
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in GRM-Gemma scoring: {e}")
            return [None] * len(instructions)

class ArmorRMModel(BaseRewardModel):
    """Implementation of ArmorRM reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        
    def load(self) -> None:
        """Load ArmorRM model and tokenizer"""
        try:             
            logging.info(f"Loading ArmorRM model...")
            config = {
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": self.dtype,
                "low_cpu_mem_usage": True,
            }
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                model_max_length=self.max_input_length
            )
            self.model.eval()
        except Exception as e:
            logging.error(f"Error loading ArmorRM model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Get scores for instruction-response pairs"""
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="ArmorRM scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Process each pair individually
                for prompt, response in zip(batch_instructions, batch_responses):
                    try:
                        # Format as chat messages
                        messages = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}
                        ]
                        
                        # Apply chat template
                        input_ids = self.tokenizer.apply_chat_template(
                            messages,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        # Get model outputs
                        with torch.no_grad():
                            outputs = self.model(input_ids)
                            preference_score = outputs.score.cpu().float()
                            
                        # Add the preference score
                        all_scores.append(float(preference_score[0]))
                        
                    except Exception as e:
                        logging.warning(f"Error processing single example: {e}")
                        all_scores.append(None)
                        
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in ArmorRM scoring: {e}")
            return [None] * len(instructions)

class Qwen72BModel(BaseRewardModel):
    """Implementation of Qwen 72B reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device="auto",
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "Qwen/Qwen2.5-Math-RM-72B"
        
    def load(self) -> None:
        """Load Qwen 72B model and tokenizer"""
        try:
            logging.info("Loading Qwen 72B model with auto device mapping...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading Qwen 72B model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Get reward scores for multiple instruction-response pairs"""
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="Qwen 72B scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Format messages for each instruction-response pair
                batch_messages = [
                    [
                        {"role": "system", "content": "Please reason step by step."},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response}
                    ]
                    for instruction, response in zip(batch_instructions, batch_responses)
                ]
                
                # Apply chat template
                batch_inputs = [
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    for messages in batch_messages
                ]
                
                # Process each input separately to avoid OOM
                batch_scores = []
                for input_text in batch_inputs:
                    input_ids = self.tokenizer.encode(
                        input_text,
                        return_tensors="pt",
                        add_special_tokens=False
                    )
                    
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids)
                        # Extract score from model output
                        score = outputs[0].mean().float().cpu().item()
                        batch_scores.append(score)
                        
                all_scores.extend(batch_scores)
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in Qwen 72B scoring: {e}")
            return [None] * len(instructions)

class INFORMForSequenceClassification(LlamaPreTrainedModel):
    """INFORM model architecture for sequence classification"""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        return SequenceClassifierOutputWithPast(
            loss=None,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

class INFORMModel(BaseRewardModel):
    """Implementation of INFORM reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device="auto",  # Override device to use auto device mapping for 70B model
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "infly/INF-ORM-Llama3.1-70B"  # Not found locally
        
    def load(self) -> None:
        """Load INFORM model and tokenizer"""
        try:
            logging.info("Loading INFORM model with auto device mapping...")
            
            # Load model with flash attention and auto device mapping
            self.model = INFORMForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                num_labels=1,
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=True
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading INFORM model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Get reward scores for multiple instruction-response pairs"""
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="INFORM scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Format as chat messages
                batch_messages = [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    for prompt, response in zip(batch_instructions, batch_responses)
                ]
                
                # Apply chat template and tokenize each conversation
                batch_inputs = [
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_tensors="pt"
                    )
                    for messages in batch_messages
                ]
                
                # Process each input separately to handle variable lengths
                with torch.no_grad():
                    batch_scores = [
                        self.model(input_ids).logits[0][0].float().cpu().item()
                        for input_ids in batch_inputs
                    ]
                    
                all_scores.extend(batch_scores)
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in INFORM scoring: {e}")
            return [None] * len(instructions)
        
class SkyworksGemmaModel(BaseRewardModel):
    """Implementation of Skyworks Reward Gemma model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
        
    def load(self) -> None:
        try:
            logging.info(f"Loading Skyworks Gemma model on {self.device}...")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Load tokenizer from the same directory as model
            tokenizer_path = os.path.dirname(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=True,
                trust_remote_code=True,
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading Skyworks Gemma model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Get reward scores for multiple instruction-response pairs"""
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="Skyworks Gemma scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Format as chat messages
                batch_messages = [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    for prompt, response in zip(batch_instructions, batch_responses)
                ]
                
                # Apply chat template and tokenize
                batch_inputs = [
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_tensors="pt"
                    )
                    for messages in batch_messages
                ]
                
                # Get scores
                with torch.no_grad():
                    batch_scores = [
                        self.model(input_ids).logits[0][0].float().cpu().item()
                        for input_ids in batch_inputs
                    ]
                    
                all_scores.extend(batch_scores)
                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in Skyworks Gemma scoring: {e}")
            return [None] * len(instructions)
        
class QRMGemmaModel(BaseRewardModel):
    """Implementation of QRM Gemma 27B reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,  # Override to use auto device mapping
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "nicolinho/QRM-Gemma-2-27B"
        self.attributes = [
            'helpsteer-helpfulness',
            'helpsteer-correctness',
            'helpsteer-coherence',
            'helpsteer-complexity',
            'helpsteer-verbosity'
        ]
        
    def load(self) -> None:
        """Load Skyworks Gemma model and tokenizer"""
        try:
            logging.info(f"Loading Skyworks Gemma model...")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
                num_labels=1
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading Skyworks Gemma model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None,
        return_quantiles: bool = False
    ) -> Union[List[float], Dict[str, Union[List[float], List[List[float]]]]]:
        """Get reward scores for multiple instruction-response pairs.
        
        Args:
            instructions: List of instruction strings
            responses: List of response strings
            batch_size: Optional batch size override
            return_quantiles: If True, return quantile estimates along with scores
            
        Returns:
            If return_quantiles is False:
                List of reward scores
            If return_quantiles is True:
                Dictionary containing:
                - 'scores': List of reward scores
                - 'quantiles': List of quantile estimates for each input
                - 'attributes': List of attribute names
        """
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            all_quantiles = [] if return_quantiles else None
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="QRM Gemma scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Format as chat messages
                batch_messages = [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    for prompt, response in zip(batch_instructions, batch_responses)
                ]
                
                # Apply chat template and tokenize
                batch_inputs = [
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_tensors="pt"
                    )
                    for messages in batch_messages
                ]
                
                # Get scores and optionally quantiles
                with torch.no_grad():
                    for input_ids in batch_inputs:
                        outputs = self.model(input_ids)
                        score = outputs.score.float().cpu().item()
                        all_scores.append(score)
                        
                        if return_quantiles:
                            quantiles = outputs.reward_quantiles.float().cpu().tolist()
                            all_quantiles.append(quantiles)
                
            if return_quantiles:
                return {
                    'scores': all_scores,
                    'quantiles': all_quantiles,
                    'attributes': self.attributes
                }
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in QRM Gemma scoring: {e}")
            if return_quantiles:
                return {
                    'scores': [None] * len(instructions),
                    'quantiles': [None] * len(instructions),
                    'attributes': self.attributes
                }
            return [None] * len(instructions)

class LDLRewardGemmaModel(BaseRewardModel):
    """Implementation of LDL-Reward-Gemma model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1"
        
    def load(self) -> None:
        """Load LDL-Reward-Gemma model and tokenizer"""
        try:
            logging.info(f"Loading LDL-Reward-Gemma model on {self.device}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=True,
            )
            
            # Load model with optimizations
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading LDL-Reward-Gemma model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str]
    ) -> List[float]:
        """Get reward scores for instruction-response pairs"""
        try:
            if not self.model or not self.tokenizer:
                self.load()
                
            scores = []
            for i in range(0, len(instructions), self.batch_size):
                batch_instructions = instructions[i:i + self.batch_size]
                batch_responses = responses[i:i + self.batch_size]
                
                # Create conversation format for each pair
                conversations = [
                    [
                        {"role": "user", "content": instr},
                        {"role": "assistant", "content": resp}
                    ]
                    for instr, resp in zip(batch_instructions, batch_responses)
                ]
                
                # Tokenize conversations
                batch_inputs = [
                    self.tokenizer.apply_chat_template(
                        conv,
                        tokenize=True,
                        return_tensors="pt"
                    )
                    for conv in conversations
                ]
                
                # Get scores
                with torch.no_grad():
                    batch_scores = [
                        self.model(inputs).logits[0].item()
                        for inputs in batch_inputs
                    ]
                    
                scores.extend(batch_scores)
                
            return scores
            
        except Exception as e:
            logging.error(f"Error getting scores from LDL-Reward-Gemma: {e}")
            return [None] * len(instructions)

class InternLM2RewardModel(BaseRewardModel):
    """Implementation of InternLM2 20B reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.float16  # InternLM2 uses float16
        )
        self.model_name = "internlm/internlm2-20b-reward"
        
    def load(self) -> None:
        """Load InternLM2 reward model and tokenizer"""
        try:
            logging.info(f"Loading InternLM2 reward model on {self.device}...")
            
            # Load model with optimizations
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading InternLM2 reward model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Get reward scores for instruction-response pairs"""
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            
            # Process in batches
            for i in tqdm(range(0, len(instructions), batch_size), desc="InternLM2 scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Format as chat messages
                batch_chats = [
                    [
                        {"role": "user", "content": instr},
                        {"role": "assistant", "content": resp}
                    ]
                    for instr, resp in zip(batch_instructions, batch_responses)
                ]
                
                try:
                    # First try batch processing
                    with torch.no_grad():
                        # Process each chat in batch individually since get_scores() doesn't work as expected
                        batch_scores = [
                            float(self.model.get_score(self.tokenizer, chat))
                            for chat in batch_chats
                        ]
                        all_scores.extend(batch_scores)
                        
                except torch.cuda.OutOfMemoryError:
                    # If batch fails, process each sample individually with error handling
                    logging.warning(f"CUDA OOM for batch at index {i}, falling back to individual processing with OOM handling...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process each sample in the batch individually
                    for chat in batch_chats:
                        try:
                            with torch.no_grad():
                                score = float(self.model.get_score(self.tokenizer, chat))
                                all_scores.append(score)
                        except torch.cuda.OutOfMemoryError:
                            logging.warning(f"CUDA OOM for individual sample, marking as None")
                            all_scores.append(None)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in InternLM2 scoring: {e}")
            return [None] * len(instructions)

class EurusPRMStage2Model(BaseRewardModel):
    """Implementation of Eurus Stage 2 PRM reward model"""

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "PRIME-RL/EurusPRM-Stage2"
        self.ref_model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
        self.coef = 0.001  # Coefficient from demo
        self.step_scores: Dict[int, List[float]] = {}

    def load(self) -> None:
        logging.info(f"Loading Eurus PRM Stage 2 model on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.ref_model_name,
            device_map="auto",
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model.eval()
        self.ref_model.eval()

    def get_logps(self, model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        ).logits
        labels = inputs['labels'][:, 1:].clone().long()
        logits = logits[:, :-1, :]
        labels[labels == -100] = 0
        return torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)

    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> Dict[str, List[float]]:
        if self.model is None:
            self.load()
        batch_size = batch_size or self.batch_size
        all_scores = []
        self.step_scores = {}

        for i in range(0, len(instructions), batch_size):
            for idx_in_batch, (prompt, response) in enumerate(
                zip(instructions[i:i+batch_size], responses[i:i+batch_size])
            ):
                overall_idx = i + idx_in_batch

                # split into steps and compute step token boundaries
                steps = [s.strip() for s in response.split("\n") if s.strip()]
                step_last_tokens = []
                for step_num in range(len(steps) + 1):
                    conv = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": "\n\n".join(steps[:step_num])}
                        ],
                        tokenize=False,
                        add_generation_prompt=False
                    ).strip()
                    if 0 < step_num < len(steps):
                        conv += "\n\n"
                    token_ids = self.tokenizer.encode(conv, add_special_tokens=False)
                    step_last_tokens.append(len(token_ids) - 2)

                # full conversation tokenization
                input_ids = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "\n\n".join(steps)}
                    ],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt"
                )
                attention_mask = (input_ids != self.tokenizer.pad_token_id)

                # package inputs and move to device
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids
                }
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)

                # build mask & indices on device
                label_mask = torch.tensor(
                    [[0] * step_last_tokens[0] + [1] * (input_ids.shape[-1] - step_last_tokens[0])],
                    device=self.device
                )
                step_last_tokens_tensor = torch.tensor([step_last_tokens], device=self.device)

                # compute per‐step log‐prob differences
                with torch.no_grad():
                    per_token = self.get_logps(self.model, inputs)
                    ref_token = self.get_logps(self.ref_model, inputs)
                    raw = per_token - ref_token
                    beta = self.coef * raw * label_mask[:, 1:]
                    beta = beta.cumsum(-1)
                    step_rewards = beta.gather(
                        dim=-1,
                        index=step_last_tokens_tensor[:, 1:]
                    )

                # store & aggregate
                self.step_scores[overall_idx] = step_rewards[0].cpu().tolist()
                if step_rewards.numel() > 0:
                    all_scores.append({
                        "min_scores": float(step_rewards.min()),
                        "max_scores": float(step_rewards.max()),
                        "avg_scores": float(step_rewards.mean())
                    })
                else:
                    all_scores.append({
                        "min_scores": None,
                        "max_scores": None,
                        "avg_scores": None
                    })

        return {
            "min_scores": [s["min_scores"] for s in all_scores],
            "max_scores": [s["max_scores"] for s in all_scores],
            "avg_scores": [s["avg_scores"] for s in all_scores],
        }


class EurusPRMStage1Model(BaseRewardModel):
    """Implementation of Eurus Stage 1 PRM reward model"""

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "PRIME-RL/EurusPRM-Stage1"
        self.ref_model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
        self.coef = 0.001
        self.step_scores: Dict[int, List[float]] = {}

    def load(self) -> None:
        logging.info(f"Loading Eurus PRM Stage 1 model on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.ref_model_name,
            device_map="auto",
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model.eval()
        self.ref_model.eval()

    def get_logps(self, model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        ).logits
        labels = inputs['labels'][:, 1:].clone().long()
        logits = logits[:, :-1, :]
        labels[labels == -100] = 0
        return torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)

    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> Dict[str, List[float]]:
        if self.model is None:
            self.load()
        batch_size = batch_size or self.batch_size
        all_scores = []
        self.step_scores = {}

        for i in range(0, len(instructions), batch_size):
            for idx_in_batch, (prompt, response) in enumerate(
                zip(instructions[i:i+batch_size], responses[i:i+batch_size])
            ):
                overall_idx = i + idx_in_batch

                steps = [s.strip() for s in response.split("\n") if s.strip()]
                step_last_tokens = []
                for step_num in range(len(steps) + 1):
                    conv = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": "\n\n".join(steps[:step_num])}
                        ],
                        tokenize=False,
                        add_generation_prompt=False
                    ).strip()
                    if 0 < step_num < len(steps):
                        conv += "\n\n"
                    tokens = self.tokenizer.encode(conv, add_special_tokens=False)
                    step_last_tokens.append(len(tokens) - 2)

                input_ids = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "\n\n".join(steps)}
                    ],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt"
                )
                attention_mask = (input_ids != self.tokenizer.pad_token_id)

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids
                }
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)

                label_mask = torch.tensor(
                    [[0] * step_last_tokens[0] + [1] * (input_ids.shape[-1] - step_last_tokens[0])],
                    device=self.device
                )
                step_last_tokens_tensor = torch.tensor([step_last_tokens], device=self.device)

                with torch.no_grad():
                    per_token = self.get_logps(self.model, inputs)
                    ref_token = self.get_logps(self.ref_model, inputs)
                    raw = per_token - ref_token
                    beta = self.coef * raw * label_mask[:, 1:]
                    beta = beta.cumsum(-1)
                    step_rewards = beta.gather(
                        dim=-1,
                        index=step_last_tokens_tensor[:, 1:]
                    )

                self.step_scores[overall_idx] = step_rewards[0].cpu().tolist()
                if step_rewards.numel() > 0:
                    all_scores.append({
                        "min_scores": float(step_rewards.min()),
                        "max_scores": float(step_rewards.max()),
                        "avg_scores": float(step_rewards.mean())
                    })
                else:
                    all_scores.append({
                        "min_scores": None,
                        "max_scores": None,
                        "avg_scores": None
                    })

        return {
            "min_scores": [s["min_scores"] for s in all_scores],
            "max_scores": [s["max_scores"] for s in all_scores],
            "avg_scores": [s["avg_scores"] for s in all_scores],
        }

class InternLM2Reward7BModel(BaseRewardModel):
    """Implementation of InternLM2 7B reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.float16  # InternLM2 uses float16
        )
        self.model_name = "internlm/internlm2-7b-reward"
        
    def load(self) -> None:
        """Load InternLM2 7B reward model and tokenizer"""
        try:
            logging.info(f"Loading InternLM2 7B reward model...")
            
            # Load model with optimizations
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading InternLM2 7B reward model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Get reward scores for instruction-response pairs"""
        if not self.model or not self.tokenizer:
            self.load()
            
        batch_size = batch_size or self.batch_size
        all_scores = []
        
        # Process in batches
        for i in tqdm(range(0, len(instructions), batch_size), desc="InternLM2 7B scoring"):
            batch_instructions = instructions[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]
            
            # Format as chat messages
            batch_chats = [
                [
                    {"role": "user", "content": instr},
                    {"role": "assistant", "content": resp}
                ]
                for instr, resp in zip(batch_instructions, batch_responses)
            ]
            
            # Process each chat individually and handle errors per sample
            batch_scores = []
            for chat in batch_chats:
                try:
                    with torch.no_grad():
                        score = self.model.get_score(self.tokenizer, chat)
                        batch_scores.append(float(score))
                except Exception as e:
                    logging.warning(f"Error scoring individual sample: {e}")
                    batch_scores.append(None)
                    # Clear CUDA cache after error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            all_scores.extend(batch_scores)
            
            # Clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_scores
        
class DecisionTreeRewardModel8B(BaseRewardModel):
    """Implementation of Decision Tree Reward model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "RLHFlow/Decision-Tree-Reward-Llama-3.1-8B"
        self.attributes = [
            'helpfulness',
            'correctness',
            'coherence',
            'complexity',
            'verbosity'
        ]
        
    def load(self) -> None:
        """Load Decision Tree Reward model and tokenizer"""
        try:
            logging.info(f"Loading Decision Tree Reward model on {self.device}...")
            
            # Load model with optimizations
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=self.dtype,
                trust_remote_code=True,
                use_cache=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )
            
            # Load decision tree
            self.model.load_decision_tree(self.model_name, filename="decision_tree.pkl")
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading Decision Tree Reward model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None,
        return_attributes: bool = False
    ) -> Union[List[float], Dict[str, List[float]]]:
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            attribute_scores = {attr: [] for attr in self.attributes} if return_attributes else None
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="Decision Tree Reward scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Process each pair individually since we need to compare with itself
                for prompt, response in zip(batch_instructions, batch_responses):
                    outputs = self.model.compare(
                        prompt,
                        response,
                        response,
                        self.tokenizer,
                    )
                    
                    rewards = outputs["rewards"][0]
                    
                    if return_attributes:
                        for attr, score in zip(self.attributes, rewards):
                            attribute_scores[attr].append(float(score))
                    
                    # Get correctness score (index 1 in attributes list)
                    correctness_score = float(rewards[1])  # 'correctness' is the second attribute
                    all_scores.append(correctness_score)
            
            if return_attributes:
                return attribute_scores
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in Decision Tree Reward scoring: {e}")
            if return_attributes:
                return {attr: [None] * len(instructions) for attr in self.attributes}
            return [None] * len(instructions)
        
class DecisionTreeRewardModel27B(BaseRewardModel):
    """Implementation of Decision Tree Reward Gemma 27B model"""
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )
        self.model_name = "RLHFlow/Decision-Tree-Reward-Gemma-2-27B"
        self.attributes = [
            'helpfulness',
            'correctness',
            'coherence',
            'complexity',
            'verbosity'
        ]
        
    def load(self) -> None:
        try:
            logging.info(f"Loading Decision Tree Reward Gemma model on {self.device}...")
            
            # Load model with optimizations
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            
            # Load tokenizer from the same directory as model
            tokenizer_path = os.path.dirname(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=True,
                trust_remote_code=True,
            )
            
            # Load decision tree from the same directory
            tree_path = os.path.join(os.path.dirname(self.model_name), "decision_tree.pkl")
            self.model.load_decision_tree(tree_path)
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading Decision Tree Reward Gemma model: {e}")
            self.unload()
            raise
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None,
        return_attributes: bool = False
    ) -> Union[List[float], Dict[str, List[float]]]:
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            attribute_scores = {attr: [] for attr in self.attributes} if return_attributes else None
            
            for i in tqdm(range(0, len(instructions), batch_size), desc="Decision Tree Reward Gemma scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Process each pair individually since we need to compare with itself
                for prompt, response in zip(batch_instructions, batch_responses):
                    outputs = self.model.compare(
                        prompt,
                        response,
                        response,
                        self.tokenizer,
                    )
                    
                    rewards = outputs["rewards"][0]
                    
                    if return_attributes:
                        for attr, score in zip(self.attributes, rewards):
                            attribute_scores[attr].append(float(score))
                    
                    # Get correctness score (index 1 in attributes list)
                    correctness_score = float(rewards[1])  # 'correctness' is the second attribute
                    all_scores.append(correctness_score)
            
            if return_attributes:
                return attribute_scores
            return all_scores
            
        except Exception as e:
            logging.error(f"Error in Decision Tree Reward Gemma scoring: {e}")
            if return_attributes:
                return {attr: [None] * len(instructions) for attr in self.attributes}
            return [None] * len(instructions)
        
class Qwen72BPRMModel(BaseRewardModel):
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        max_input_length: int = 2048
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            max_input_length=max_input_length,
            dtype=torch.bfloat16
        )

        self.model_name = "Qwen/Qwen2.5-Math-PRM-72B"
        self.step_scores = {}  # Store step-wise scores
        
    def load(self) -> None:
        try:
            logging.info(f"Loading Qwen 72B PRM model on {self.device}...")
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading Qwen 72B PRM model: {e}")
            self.unload()
            raise
            
    def make_step_rewards(self, logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
        """Helper function to calculate step rewards"""
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
            
    def get_scores(
        self,
        instructions: List[str],
        responses: List[str],
        batch_size: Optional[int] = None
    ) -> Dict[str, List[float]]:
        try:
            if self.model is None:
                self.load()
                
            batch_size = batch_size or self.batch_size
            all_scores = []
            self.step_scores = {}  # Reset step scores
            
            # Process in batches
            for i in tqdm(range(0, len(instructions), batch_size), desc="Qwen 72B PRM scoring"):
                batch_instructions = instructions[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                for batch_idx, (prompt, response) in enumerate(zip(batch_instructions, batch_responses)):
                    overall_idx = i + batch_idx
                    
                    # Split response into steps
                    steps = [step.strip() for step in response.split("\n") if step.strip()]
                    
                    # Format messages following demo
                    messages = [
                        {"role": "system", "content": "Please reason step by step."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}
                    ]
                    
                    # Apply chat template
                    conversation = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    
                    # Tokenize
                    input_ids = self.tokenizer.encode(
                        conversation,
                        return_tensors="pt"
                    )
                    
                    # Get rewards following demo implementation
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids)
                        step_sep_id = self.tokenizer.encode("<extra_0>")[0]
                        token_masks = (input_ids == step_sep_id)
                        step_rewards = self.make_step_rewards(outputs[0], token_masks)
                        
                        if step_rewards and step_rewards[0]:
                            # Store step scores
                            self.step_scores[overall_idx] = step_rewards[0]
                            # Calculate min, max, and avg scores
                            min_score = min(step_rewards[0])
                            max_score = max(step_rewards[0])
                            avg_score = sum(step_rewards[0]) / len(step_rewards[0])
                            all_scores.append({
                                'min_scores': min_score,
                                'max_scores': max_score,
                                'avg_scores': avg_score
                            })
                        else:
                            self.step_scores[overall_idx] = []
                            all_scores.append({
                                'min_scores': None,
                                'max_scores': None,
                                'avg_scores': None
                            })
            
            # Reorganize scores into separate lists
            return {
                'min_scores': [score['min_scores'] for score in all_scores],
                'max_scores': [score['max_scores'] for score in all_scores],
                'avg_scores': [score['avg_scores'] for score in all_scores]
            }
            
        except Exception as e:
            logging.error(f"Error in Qwen 72B PRM scoring: {e}")
            return {
                'min_scores': [None] * len(instructions),
                'max_scores': [None] * len(instructions),
                'avg_scores': [None] * len(instructions)
            }
            
    def get_step_scores(self) -> Dict[int, List[float]]:
        """Return the stored step-wise scores"""
        return self.step_scores
        
    def unload(self) -> None:
        """Safely unload model and free GPU memory"""
        self.step_scores = {}  # Clear stored scores
        super().unload()