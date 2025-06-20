import itertools
from typing import Callable, Dict, Iterable
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer       

from neuron_integrated_gradients import predict
from prune import get_masks
from utils import get_token_types_spans, INPUT_PART_TO_POSITION, untuple

class PrunedModel(torch.nn.Module):
    # Hook used to mask the output of some layer given a pruning scheme
    def __init__(self, model: torch.nn.Module, positions_per_module: Dict, device):
        super().__init__()
        self.model = model
        self.device = device
        self.hooks_handles = list()

        for layer_name, layer in positions_per_module.items():
            if "attention_probs" in layer_name:
                num_head = int(layer_name.split(".")[-1])
                dropout_module_name = layer_name.replace(f"attention_probs.{num_head}", "dropout")
                layer = dict([*self.model.named_modules()])[dropout_module_name]
                self.hooks_handles.append(layer.register_forward_hook(self.prune_dropout(self.device, positions_per_module[layer_name], num_head)))
            else:
                layer = dict([*self.model.named_modules()])[layer_name]

                self.hooks_handles.append(layer.register_forward_hook(self.prune_outputs_hook(self.device, positions_per_module[layer_name])))

    def prune_outputs_hook(self, device, positions: Iterable) -> Callable:
        def hook(model, input, output):
            output = untuple(output)
            mask = torch.ones(output[0].numel())
            for idx in positions:
                mask[int(idx)] = 0
            mask = mask.to(device)
            masked_output = list()
            for elmt in output:
                flattened_elmt = elmt.flatten()
                masked_elmt = mask * flattened_elmt
                masked_elmt = masked_elmt.view(elmt.shape)
                masked_output.append(masked_elmt)
            masked_output = torch.stack(masked_output, dim=0)
            return masked_output
        return hook
    
    def prune_dropout(self, device, head_mask: Iterable, num_head: int) -> Callable:
        def hook(model, input, output):
            output = untuple(output)
            
            mask = head_mask.to(device)
            masked_output = list()
            for all_heads in output: # Iterates over every sample in the batch
                masked_head = all_heads[num_head] * mask # We keep targetting this specific head
                # So far we have only considered the head we want to prune, we need to put back the other heads
                all_heads[num_head] = masked_head 
                # Then we add everything back to the list
                masked_output.append(all_heads)
            masked_output = torch.stack(masked_output, dim=0)
            return masked_output
        return hook

    def forward(self, input_ids, token_type_ids, attention_mask):
        model_outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return model_outputs
    
    def remove_hooks(self):
        for handle in self.hooks_handles:
            handle.remove()


def pruned_forward(model, tokenizer, inputs, neurons_to_prune, input_length: int = 128):
    """
    Forward pass by cutting the top neurons out of the model.
    """

    ### This block should probably be factorized if multiple forwards need to be called ###
    neurons_to_dimensions = dict()
    for key, value_per_input_part in neurons_to_prune.items():
        if "attention_probs" in key:
            neurons_to_dimensions[key] = torch.ones((input_length, input_length), device=model.device)
            value = value_per_input_part.get("all", None)
            if value is not None:
                neurons_to_dimensions[key] *= (1-value)
            else:
                for couple in itertools.product(INPUT_PART_TO_POSITION.keys(), repeat=2): 
                    value = value_per_input_part.get(f"{couple[0]}_{couple[1]}", None)
                    spans = get_token_types_spans(inputs["input_ids"], tokenizer)
                    positions_to_cut_attention_from = spans[INPUT_PART_TO_POSITION[couple[0]]]
                    positions_to_cut_attention_to = spans[INPUT_PART_TO_POSITION[couple[1]]]
                    neurons_to_dimensions[key][positions_to_cut_attention_from, positions_to_cut_attention_to] *= (1-value)

        else:
            for input_part, value in value_per_input_part.items():
                if len(value) > 0:
                    if input_part == "all":
                        new_list = list()
                        out_features = model.get_submodule(key).out_features
                        for neuron_position in value:
                            new_list += [neuron_position+i*out_features for i in range(input_length)] # Need to scale one neuron to the whole column
                        neurons_to_dimensions[key] = new_list
                    else:
                        # In that case we need to prune the neurons only on the range corresponding to the input part
                        spans = get_token_types_spans(inputs["input_ids"], tokenizer)
                        positions_to_cut = spans[INPUT_PART_TO_POSITION[input_part]]
                        new_list = list()
                        out_features = model.get_submodule(key).out_features
                        for neuron_position in value:
                            new_list += [neuron_position+i*out_features for i in range(positions_to_cut.start, positions_to_cut.stop)] # Need to scale one neuron to the whole column
                        neurons_to_dimensions[key] = new_list if neurons_to_dimensions.get(key) is None else neurons_to_dimensions[key] + new_list

    ## End of the block that should be factorized ###

    pruned_model = PrunedModel(
        model=model,
        positions_per_module=neurons_to_dimensions,
        device=model.device
    )

    if model.config.num_labels == 1:
        pos_to_watch = 0
        activation_fct = lambda x, dim: F.sigmoid(x)
    else: # Always watch for the positive class
        pos_to_watch = 1
        activation_fct = lambda x, dim: F.softmax(x, dim=dim)

    with torch.no_grad():
        pruned_pred = activation_fct(
            pruned_model(
                **inputs
            ).logits,
            dim=-1
        ).cpu()

    pruned_model.remove_hooks()

    return pruned_pred[0][pos_to_watch].item()


if __name__ == '__main__':
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
    nig, error = predict(query, passage, 10, 10, aggregate_per_token_type=True)

    print("Generating the masks...")
    top_neurons = get_masks(nig, 0.0, 1.0)

    print("Using masks to prune the model...")
    pruned_score = pruned_forward(
        model,
        tokenizer=tokenizer,
        inputs=inputs,
        neurons_to_prune=top_neurons,
        input_length=max_input_length
    )

    print(f"** Pruned model output: {pruned_score} **")
    print(f"** Pruned label: {1 if pruned_score >= 0.5 else 0} **")