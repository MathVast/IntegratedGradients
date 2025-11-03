import itertools
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer       

from neuron_integrated_gradients import aggregate_nig, predict, neuron_integrated_gradients
from prune import get_masks
from pruned_forward import PrunedModel, pruned_forward
from utils import generate_baseline_with_padded_query_but_special_tokens, get_token_types_spans, INPUT_PART_TO_POSITION

def post_pruning_nig(query, passage, num_reps, batch_size, neurons_to_prune, aggregate_per_token_type=False, max_input_length=128):
    """
    Compute NIG while taking into account a pruning scheme.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L12-v2").to(device)
    model.eval()

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

    ### This block should probably be factorized if multiple forwards need to be called ###
    neurons_to_dimensions = dict()
    for key, value_per_input_part in neurons_to_prune.items():
        if "attention_probs" in key:
            neurons_to_dimensions[key] = torch.ones((max_input_length, max_input_length), device=model.device)
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
                            new_list += [neuron_position+i*out_features for i in range(max_input_length)] # Need to scale one neuron to the whole column
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

    pruned_model = PrunedModel(
        model=model,
        positions_per_module=neurons_to_dimensions,
        device=model.device
    )

    nig, error = neuron_integrated_gradients(
        model=pruned_model,
        input_embeddings=input_embeds,
        token_type_ids=inputs["token_type_ids"],
        attention_mask=inputs["attention_mask"],
        baseline_embeddings=baseline_embeds,
        num_reps=num_reps,
        batch_size=batch_size,
        num_labels=num_labels,
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
    nig, error = predict(query, passage, 10, 10, aggregate_per_token_type=True, max_input_length=max_input_length)

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

    print("Running neuron integrated gradients on the pruned model...")
    new_nig, error = post_pruning_nig(
        query, 
        passage, 
        num_reps=10, 
        batch_size=10, 
        neurons_to_prune=top_neurons, 
        aggregate_per_token_type=True, 
        max_input_length=max_input_length
    )
     
    new_top_neurons = get_masks(new_nig, 0.0, 1.0)