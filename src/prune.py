import torch
from neuron_integrated_gradients import predict

def get_masks(nig_model, pruning_percentage_attention: float, pruning_percentage_ffn: float):
    attention_probs_keys = [key for key in nig_model.keys() if "attention_probs" in key]

    for key in attention_probs_keys:
        if "attention_probs" in key:
            attention_probs = nig_model.pop(key)
            for direction, values in attention_probs.items():
                for n, attention_head in enumerate(values):
                    if key + f'.{n}' in nig_model.keys():
                        nig_model[key + f'.{n}'][direction] = attention_head # Little trick to tranform every tensor to the same shape
                    else:
                        nig_model[key + f'.{n}'] = {direction: attention_head}

    attention_all_sums_along_input = list()
    ffn_all_sums_along_input = list()
    for key, values_per_input_type in nig_model.items():
        for value in values_per_input_type.values():
            if "attention_probs" in key:
                attention_all_sums_along_input.append(value.unsqueeze(dim=0))
            else:
                ffn_all_sums_along_input.append(value)

    # Deduce the threshold valeus from the aggregation and the pruning percentages
    attention_all_sums_along_input = torch.cat(attention_all_sums_along_input, dim=0)
    attention_all_sums_along_input = torch.sort(attention_all_sums_along_input).values
    ffn_all_sums_along_input = torch.cat(ffn_all_sums_along_input, dim=0)
    ffn_all_sums_along_input = torch.sort(ffn_all_sums_along_input).values
    if pruning_percentage_attention > 0:
        attention_threshold_value = attention_all_sums_along_input[int((1 - pruning_percentage_attention) * len(attention_all_sums_along_input))]
    else:
        attention_threshold_value = attention_all_sums_along_input[-1] # If no pruning, set threshold to 0
    if pruning_percentage_ffn > 0:
        ffn_threshold_value = ffn_all_sums_along_input[int((1 - pruning_percentage_ffn) * len(ffn_all_sums_along_input))]
    else: 
        ffn_threshold_value = ffn_all_sums_along_input[-1] # If no pruning, set threshold to 0
    
    # Apply them to get the masks
    top_neurons_per_layer_model = dict()
    for key, values_per_input_type in nig_model.items():
        top_neurons_per_layer_model[key] = dict()
        for input_part, value in values_per_input_type.items():
            if "attention_probs" in key:
                if value.unsqueeze(dim=0) > attention_threshold_value:
                    top_neurons_per_layer_model[key][input_part] = 1
                else:
                    top_neurons_per_layer_model[key][input_part] = 0

            else:
                indices = (value > ffn_threshold_value).nonzero().squeeze()
                if len(indices.size()) == 0:
                    top_neurons_per_layer_model[key][input_part] = indices.unsqueeze(dim=0)
                else:
                    top_neurons_per_layer_model[key][input_part] = indices

    return top_neurons_per_layer_model

    

if __name__ == '__main__':
    query = "what was the immediate impact of the success of the manhattan project?"  # "what do the xylem and the phloem do"
    passage = "The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science."

    nig, error = predict(query, passage, 20, 10, aggregate_per_token_type=True)
    print(nig.keys())

    top_neurons = get_masks(nig, pruning_percentage_attention=0.01, pruning_percentage_ffn=0.01)
    print(top_neurons)