import gc
from enum import Enum
from pyparsing import Dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
from tqdm import tqdm      

from neuron_integrated_gradients import aggregate_nig
from utils import OutputsExtractor, _get_ig_error, _get_scaled_inputs, generate_baseline_with_padded_query_but_special_tokens, get_interesting_modules, get_token_types_spans, INPUT_PART_TO_POSITION

class NeuronType(Enum):
    ATTENTION = "attention"
    FFN = "ffn"

    def check_idx(self, model, idx: int) -> bool:
        nb_attention_heads = model.config.num_attention_heads
        ffn_dim = model.config.hidden_size

        if self == NeuronType.ATTENTION:
            return nb_attention_heads > idx >= 0  # Attention heads can be indexed from 0 to num_heads - 1
        else:
            return ffn_dim > idx >= 0  # FFN neurons can be indexed from 0 to ffn_dim - 1
    
    def get_module_name(self, layer_idx: int) -> str:
        if self == NeuronType.ATTENTION:
            return f"bert.encoder.layer.{layer_idx}.attention.self.dropout"
        else:
            return f"bert.encoder.layer.{layer_idx}.intermediate.dense"


def conditional_nig(
    model, 
    input_embeddings, 
    token_type_ids,
    attention_mask,
    baseline_embeddings,
    num_reps: int, 
    batch_size: int, 
    num_labels: int,
    layer_idx: int, 
    neuron_idx: int, 
    neuron_type: NeuronType, 
    aggregate_per_token_type=False,
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
            if not aggregate_per_token_type:
                # Aggregate over all attention positions
                target_neuron_activation = target_neuron_activation.sum(dim=(1,2))  # Shape (batch_size)
            else:
                raise NotImplementedError("Aggregation per token type not implemented yet.")
        else:
            # For FFN, we have a 3D tensor: (batch_size, seq_len, hidden_size)
            target_neuron_activation = target_activation[:, :, neuron_idx] # Shape (batch_size, seq_len)
            if not aggregate_per_token_type:
                target_neuron_activation = target_neuron_activation.sum(dim=(1))  # Shape (batch_size)
            else:
                raise NotImplementedError("Aggregation per token type not implemented yet.")

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

    num_labels = model.config.num_labels

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
        num_labels=num_labels,
        layer_idx=layer_idx,
        neuron_idx=neuron_idx,
        neuron_type=neuron_type,
        aggregate_per_token_type=aggregate_per_token_type,
    )

    nig = aggregate_nig(
        nig,
        spans=spans,
        aggregate_per_token_type=aggregate_per_token_type,
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
        aggregate_per_token_type=False, 
        max_input_length=max_input_length, 
        layer_idx=5, 
        neuron_idx=3, 
        neuron_type=NeuronType.FFN
    )
