import gc
from enum import Enum
import itertools
from typing import Optional
from pyparsing import Dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
from tqdm import tqdm      

from utils import InputPart, OutputsExtractor, _get_ig_error, _get_scaled_inputs, generate_baseline_with_padded_query_but_special_tokens, get_interesting_modules, get_token_types_spans, INPUT_PART_TO_POSITION

class NeuronType(Enum):
    ATTENTION = "attention"
    FFN = "ffn"

    def check_idx(self, model, idx: int) -> bool:
        nb_attention_heads = model.config.num_attention_heads
        ffn_dim = model.config.intermediate_size

        if self == NeuronType.ATTENTION:
            return nb_attention_heads > idx >= 0  # Attention heads can be indexed from 0 to num_heads - 1
        else:
            return ffn_dim > idx >= 0  # FFN neurons can be indexed from 0 to ffn_dim - 1
    
    def get_module_name(self, layer_idx: int) -> str:
        if self == NeuronType.ATTENTION:
            return f"bert.encoder.layer.{layer_idx}.attention.self.dropout"
        else:
            return f"bert.encoder.layer.{layer_idx}.intermediate.dense"
        
def get_layer_idx_from_module_name(module_name: str) -> int:
    """
    Extract the layer index from a module name string.
    Assumes the module name follows the format 'bert.encoder.layer.{layer_idx}...'.
    """
    parts = module_name.split('.')
    try:
        layer_idx_position = parts.index('layer') + 1
        layer_idx = int(parts[layer_idx_position])
        return layer_idx
    except (ValueError, IndexError):
        raise ValueError(f"Invalid module name format: {module_name}")

def aggregate_conditional_nig(nig, spans, layer_idx: int, neuron_idx: int, aggregate_per_token_type=False, source_input_part: Optional[InputPart] = None, target_input_part: Optional[InputPart] = None):
    """
    Aggregate the neuron integrated gradients (NIG) across token types if specified.
    For conditional NIG, this function also acts as a selector of the active slices and if filters out all the other
    by setting them to NaN.
    """
    storage = dict()
    
    for key, value in nig.items():
        if aggregate_per_token_type:
            storage[key] = dict()
            if "attention_probs" in key:
                layer_idx_in_key = get_layer_idx_from_module_name(key)
                if layer_idx_in_key == layer_idx:
                    # If target layer of the conditonal NIG, we apply the filtering
                    if source_input_part is not None and target_input_part is not None:
                        # If both inpupt parts are specified, we only keep the corresponding slice
                        target_input_part_str = str(target_input_part)
                        source_input_part_str = str(source_input_part)
                        storage[key][f"{target_input_part_str}_{source_input_part_str}"] = torch.sum(value[:,spans[INPUT_PART_TO_POSITION[target_input_part_str]],spans[INPUT_PART_TO_POSITION[source_input_part_str]]], axis=(1,2), keepdim=True)

                    elif source_input_part is not None and target_input_part is None:
                        source_input_part_str = str(source_input_part)
                        # If only one input part is specified, we keep all slices involving this input part
                        for targets in INPUT_PART_TO_POSITION.keys():
                            storage[key][f"{targets}_{source_input_part_str}"] = torch.sum(value[:,spans[INPUT_PART_TO_POSITION[targets]],spans[INPUT_PART_TO_POSITION[source_input_part_str]]], axis=(1,2), keepdim=True)

                    elif source_input_part is None and target_input_part is not None:
                        target_input_part_str = str(target_input_part)
                        for sources in INPUT_PART_TO_POSITION.keys():
                            storage[key][f"{target_input_part_str}_{sources}"] = torch.sum(value[:,spans[INPUT_PART_TO_POSITION[target_input_part_str]],spans[INPUT_PART_TO_POSITION[sources]]], axis=(1,2), keepdim=True)
                    else:
                        # If no input part is specified, we keep everything
                        for couple in itertools.product(INPUT_PART_TO_POSITION.keys(), repeat=2):
                            # itertools.product is equivalent to a nest for loop and creates every possible combinations of the input_parts (total nb is 25).
                            # For each couple of input parts, we select the corresponding slices in the last two dimensions and sum over these dimensions.
                            # To mitigate the impact of the input length, we then average it by the product of the lengths of the two slices.
                            storage[key][f"{couple[0]}_{couple[1]}"] = torch.sum(value[:,spans[INPUT_PART_TO_POSITION[couple[0]]],spans[INPUT_PART_TO_POSITION[couple[1]]]], axis=(1,2), keepdim=True)

                    if neuron_idx is not None:
                        # Set to NaN all other positions except the one corresponding to the neuron_idx
                        for k in storage[key].keys():
                            temp = storage[key][k]
                            mask = torch.ones_like(temp, dtype=bool)
                            mask[neuron_idx,:,:] = False
                            temp = temp.masked_fill(mask, float('nan'))
                            storage[key][k] = temp
                else:
                    # If not the target layer of the conditonal NIG, we proceed as usual 
                    for couple in itertools.product(INPUT_PART_TO_POSITION.keys(), repeat=2):
                        # itertools.product is equivalent to a nest for loop and creates every possible combinations of the input_parts (total nb is 25).
                        # For each couple of input parts, we select the corresponding slices in the last two dimensions and sum over these dimensions.
                        # To mitigate the impact of the input length, we then average it by the product of the lengths of the two slices.
                        storage[key][f"{couple[0]}_{couple[1]}"] = torch.sum(value[:,spans[INPUT_PART_TO_POSITION[couple[0]]],spans[INPUT_PART_TO_POSITION[couple[1]]]], axis=(1,2), keepdim=True)
                        
            else:
                for input_part, idx in INPUT_PART_TO_POSITION.items():
                    storage[key][input_part] = torch.sum(value[spans[idx],:], axis=0)
        else:
            if "attention_probs" in key:
                storage[key] = {'all': torch.sum(value, axis=(1,2))}
            else:
                storage[key] = {'all': torch.sum(value, axis=0)}

    print(f"Results have been aggregated.")

    return storage

def conditional_nig(
    model, 
    input_embeddings, 
    token_type_ids,
    attention_mask,
    baseline_embeddings,
    num_reps: int, 
    batch_size: int, 
    layer_idx: int, 
    neuron_idx: int, 
    neuron_type: NeuronType, 
    source_input_part: Optional[InputPart] = None,
    target_input_part: Optional[InputPart] = None,
    spans: Optional[torch.Tensor] = None,
    compute_error: bool = False,
) -> Dict:
    """
    Compute the attribution (Neuron Integrated Gradients) of each unit for all the interesting modules in the model.

    :param torch.nn.Module model: Model for which to compute the conductance.
    :param torch.Tensor input_embeddings: Embeddings of the input.
    :param torch.Tensor token_type_ids: Token type ids of the input.
    :param torch.Tensor attention_mask: ATtention mask on the input.
    :param torch.Tensor baseline_embeddings: Embedding of the baseline for the given input.
    :param int num_label: Number of output labels for the model.
    :param int num_reps: Number of iteration to approximate the integrated gradients.
    :param int batch_size: Batch size used for each iteration (true number of steps is batch_size x num_reps).
    :param int layer_idx: The layer index of the target neuron.
    :param int neuron_idx: The index of the target neuron within the layer (between 0 and num_heads-1 for attention / between 0 and ffn_dim-1 for FFN). 
    :param NeuronType neuron_type: The nature of the target neuron (attention or feedforward).
    :param InputPart target_input_part: If provided, condition the value of the neruon we are interested in to a specific input part.
    :param Tensor spans: If provided, delimitate the spans corresponding to the different input parts and is used when target_input_part is specified.
    :param bool compute_error: Whether to compute the IG approximation error or not.
    :return Dict: Attribution for each activation unit for each layer in the model.
    """

    layer_names, _ = get_interesting_modules(
        model=model,
        list_regex=None # at this point we don't want to filter the modules for now
    )

    # Access the activation of the target_neuron
    target_layer_name = neuron_type.get_module_name(layer_idx)
    if target_layer_name not in layer_names:
        raise ValueError(f"Target layer {target_layer_name} not found among interesting modules ({layer_names}).")
    
    # Keep only modules up to, and including, the target layer
    idx = layer_names.index(target_layer_name)
    layer_names = layer_names[: idx + 1]

    if "attention" in target_layer_name:
        target_layer_name = target_layer_name.replace("dropout", "attention_probs")

    extractor = OutputsExtractor(
        model=model,
        layer_names=layer_names,
    )

    list_scaled_embeddings = _get_scaled_inputs(
        input_embeddings[0].detach().cpu().numpy(), 
        baseline_embeddings[0].detach().cpu().numpy(), 
        batch_size=batch_size, 
        num_reps=num_reps, 
        device=model.device
    ) 
    all_outputs = list()
    path_gradients = dict() # Stores the gradient corresponding to each input wrt the output 

    for i in tqdm(range(len(list_scaled_embeddings))):
        batch_pos_inputs = torch.Tensor(list_scaled_embeddings[i]).to(torch.float)
        batch_pos_inputs.requires_grad = True
        _ = extractor.forward(batch_pos_inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)
    
        target_activation = extractor.outputs_store[target_layer_name]  
        if neuron_type == NeuronType.ATTENTION:
            # For attention, we have a 4D tensor: (batch_size, num_heads, seq_len, seq_len)
            target_neuron_activation = target_activation[:, neuron_idx, :, :] # Shape (batch_size, seq_len, seq_len)
            if not target_input_part and not source_input_part:
                # Aggregate over all attention positions
                target_spans = (slice(None), slice(None))
            elif not target_input_part:
                target_spans = (slice(None), spans[INPUT_PART_TO_POSITION[str(source_input_part)]])
            elif not source_input_part:
                target_spans = (spans[INPUT_PART_TO_POSITION[str(target_input_part)]], slice(None))
            else:
                target_spans = (spans[INPUT_PART_TO_POSITION[str(target_input_part)]], spans[INPUT_PART_TO_POSITION[str(source_input_part)]])

            target_neuron_activation = torch.sum(target_neuron_activation[:,target_spans[0], target_spans[1]], axis=(1,2), keepdim=True)                
        else:
            # For FFN, we have a 3D tensor: (batch_size, seq_len, hidden_size)
            target_neuron_activation = target_activation[:, :, neuron_idx] 
            if not target_input_part:
                target_neuron_activation = target_neuron_activation.sum(dim=(1))
            else:
                target_neuron_activation = torch.sum(target_neuron_activation[:,spans[INPUT_PART_TO_POSITION[str(target_input_part)]]], axis=(1), keepdim=True) 

        # Now do a backward pass per input in the batch
        for j in range(batch_pos_inputs.shape[0]):
            extractor.model.zero_grad()

            # Backward from the target neuron activation
            target_neuron_activation[j].backward(retain_graph=True)

            for key, activation in extractor.outputs_store.items():
                grad = activation.grad.detach().cpu()
                value = activation.detach().cpu()

                if i == 0 and j == 0:
                    # Skip baseline point, or init accumulator
                    previous_activations = {}
                    for k in extractor.outputs_store.keys():
                        previous_activations[k] = extractor.outputs_store[k][0].detach().cpu()
                    continue

                # Compute contribution for this step
                diff = value[j] - previous_activations[key]  # shape: [num_neurons]
                prod = diff * grad[j]  # element-wise: shape [num_neurons]

                if key not in path_gradients:
                    path_gradients[key] = prod
                else:
                    path_gradients[key] += prod

            # Save current activations as previous for next step
            for key in previous_activations:
                previous_activations[key] = extractor.outputs_store[key][j].detach().cpu()

    extractor.clear_items()
    extractor.remove_hooks()

    if compute_error:
        errors = dict()
        for key in path_gradients.keys():
           errors[key] = _get_ig_error(path_gradients[key], all_outputs[0][0], all_outputs[-1][-1], debug=False)
            
    gc.collect()
    torch.cuda.empty_cache() 
    return path_gradients, errors if compute_error else None      

def compute_conditional_nig(
        query, 
        passage, 
        num_reps, 
        batch_size: int, 
        layer_idx: int, 
        neuron_idx: int, 
        neuron_type: NeuronType, 
        source_input_part: Optional[InputPart] = None,
        target_input_part: Optional[InputPart] = None,
        aggregate_per_token_type=False, 
        max_input_length=128
    ):
    """
    Compute the NIG wrt a specific hidden state instead of the final output.

    :param str query: Query text.
    :param str passage: Passage text.
    :param int num_reps: Number of iteration to approximate the integrated gradients.
    :param int batch_size: Batch size used for each iteration (true number of steps is batch_size x num_reps).
    :param int layer_idx: Layer index of the target neuron.
    :param int neuron_idx: Neuron index within the target layer.
    :param NeuronType neuron_type: Type of the neuron (e.g., attention, feedforward).
    :param InputPart target_input_part: If provided, condition the value of the neuron we are interested in to a specific input part.
    :param bool aggregate_per_token_type: Whether to aggregate the NIG per input part or not.
    :param int max_input_length: Maximum input length for the model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L12-v2").to(device)
    model.eval()
    # First, validate the target neuron index
    if not neuron_type.check_idx(model, neuron_idx):
        raise ValueError(f"Invalid target neuron index {neuron_idx} for neuron type {neuron_type}.")
    
    if layer_idx >= model.config.num_hidden_layers:
        raise ValueError(f"Invalid target layer index {layer_idx}. Model has {model.config.num_hidden_layers} layers.")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
        query,
        passage,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    ).to(model.device)

    spans = get_token_types_spans(inputs["input_ids"], tokenizer)

    embeddings = model.bert.get_input_embeddings()
    input_embeds = embeddings(inputs["input_ids"])

    # Baseline gradient
    baseline_inputs = inputs.copy()
    baseline_embeds = generate_baseline_with_padded_query_but_special_tokens(
        tokenizer,
        baseline_inputs["input_ids"],
        embeddings,
        device
    )

    # Compute conditional NIG
    nig, error = conditional_nig(
        model=model,
        input_embeddings=input_embeds,
        token_type_ids=inputs["token_type_ids"],
        attention_mask=inputs["attention_mask"],
        baseline_embeddings=baseline_embeds,
        num_reps=num_reps,
        batch_size=batch_size,
        layer_idx=layer_idx,
        neuron_idx=neuron_idx,
        neuron_type=neuron_type,
        source_input_part=source_input_part,
        target_input_part=target_input_part,
        spans=spans
    )

    nig = aggregate_conditional_nig(
        nig,
        spans=spans,
        layer_idx=layer_idx,
        neuron_idx=neuron_idx,
        aggregate_per_token_type=aggregate_per_token_type,
        source_input_part=source_input_part,
        target_input_part=target_input_part,
    )

    return nig, error


if __name__ == "__main__":
    query = "what was the immediate impact of the success of the manhattan project?"  # "what do the xylem and the phloem do"
    passage = "The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science."
    # passage = "Phloem and xylem are complex tissues that perform transportation of food and water in a plant."
    max_input_length = 128
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L12-v2").to(device)
    model.eval()

    print("Running original forward pass...")
    inputs = tokenizer(
        query,
        passage,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model(**inputs)
    
    print(f"** Default model output: {outputs} **")
    if model.config.num_labels == 1:
        pos_to_watch = 0
        activation_fct = F.sigmoid
    else: # Always watch for the positive class
        pos_to_watch = 1
        activation_fct = F.softmax
    score = activation_fct(outputs.logits)[0][pos_to_watch]
    print(f"** From logits to proba: {score} **")

    print(f"** From proba to label: {1 if score >= 0.5 else 0} **")

    print("Running neuron integrated gradients...")
    nig, error = compute_conditional_nig(
        query, 
        passage, 
        10, 
        10, 
        aggregate_per_token_type=True, 
        max_input_length=max_input_length, 
        layer_idx=2, 
        neuron_idx=5, 
        neuron_type=NeuronType.ATTENTION,
        source_input_part=InputPart.QUERY,
        target_input_part=None
    )
