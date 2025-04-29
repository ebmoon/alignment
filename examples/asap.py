import os
import torch
import json
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from alignment.monitor.grammar import LLGuidanceMonitor
from alignment.models import TransformersModel
from alignment.monitor.adaptive_utils import AdaptiveMaskTrie
from datasets import load_dataset

# Constants
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 64
NUM_SAMPLES = 200
NUM_STEPS = 51
DATASET = "ebmoon/GAD-dataset"
SPLIT = "SLIA"
ID = 'qm_max3'
VERSION = 'bias_2x'
OUTPUT_FILE = f"{ID}_outputs_{VERSION}.jsonl"

def load_model_and_tokenizer():
    """Load the model and tokenizer"""
    cache_dir = "/trunk/model-hub"
    device = torch.device(DEVICE)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        cache_dir=cache_dir, 
        device_map="auto", 
        torch_dtype=DTYPE
    )
    model.resize_token_embeddings(len(tokenizer))
    
    model = TransformersModel(model, tokenizer)
    print(f"Model loaded on device: {model.device}")
    
    return model, tokenizer

def load_dataset_and_grammar():
    """Load the dataset and grammar"""
    dataset = load_dataset(DATASET, split=SPLIT)
    
    prompt = None
    grammar_str = None
    
    for data in dataset:
        if ID == data['id']:
            prompt = data['prompt']
            grammar_str = data['grammar']
            break
    
    grammar_str = grammar_str.replace("ntInt", "ntint")
    grammar_str = grammar_str.replace("ntBool", "ntbool")
    grammar_str = grammar_str.replace("ntString", "ntstring")
    grammar_str = grammar_str.replace("root", "start")
    grammar_str = grammar_str.replace("Start", "ntstart")
    grammar_str = grammar_str.replace("::=", ":")
    
    print("Grammar loaded:")
    print(grammar_str)

    # Save grammar to temporary file
    with open("tmp.lark", "w") as f:
        f.write(grammar_str)
    
    return prompt, grammar_str

def prepare_input(tokenizer, prompt):
    """Prepare the input for generation"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    decode_output = tokenizer(
        [formatted_prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )
    input_ids = decode_output["input_ids"]
    attention_mask = decode_output["attention_mask"]
    
    return input_ids, attention_mask

def logprob_from_logits(scores: torch.Tensor, query_ids: torch.Tensor) -> float:
    """
    Get the log probability of a sequence given the model scores.
    `scores` has shape (1, seq_len, vocab_size).
    `query_ids` has shape (1, seq_len).
    Result has shape (batch_size,).
    """
    scores = torch.stack(scores, dim=1)
    logprobs = torch.log_softmax(scores, dim=-1)
    query_logprobs = logprobs[0, torch.arange(logprobs.shape[1]), query_ids[0]]
    logprob = query_logprobs.sum().item()
    return logprob

def write_quadruples_to_jsonl(data_2d, output_file):
    """
    data_2d is a 2D list where each entry is a quadruple:
       (string_value, torch_tensor_of_integers, raw_probability, adjusted_probability).
    This function writes each "row" (i.e., sub-list) of data_2d as one line in a JSONL file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in data_2d:
            row_as_dicts = []
            for (text, tokens_tensor, raw_prob, adj_prob) in row:
                row_as_dicts.append({
                    "generated_sentence": text,
                    # Convert the tensor to list so json.dumps can serialize it
                    "tokens": tokens_tensor.tolist(),
                    "raw_probability": raw_prob,
                    "adjusted_probability": adj_prob
                })
            f.write(json.dumps(row_as_dicts) + "\n")

def tokens_to_str(tokens):
    return "_".join(str(t) for t in tokens)

def calculate_kl_divergence(outputs, num_samples, num_steps):
    """Calculate KL divergence and return data for plotting"""
    global_prob_dict = {}

    # Count the number of times each output is generated
    for output in outputs:
        for out, tokens, raw_log_prob, gcd_prog_prob in output:
            key = tokens_to_str(tokens)

            raw_prob = np.exp(raw_log_prob)
            gcd_prob = np.exp(gcd_prog_prob)

            if key not in global_prob_dict:
                global_prob_dict[key] = (out, raw_prob, gcd_prob, 1)
            else:
                out, raw_prob, gcd_prob, count = global_prob_dict[key]
                global_prob_dict[key] = (out, raw_prob, gcd_prob, count + 1)

    prob_dicts = []
    for i in range(num_steps):
        prob_dict = {k:(v[0], v[1], v[2], 0) for k, v in global_prob_dict.items()}
        for j in range(len(outputs)):
            out, tokens, raw_log_prob, gcd_prog_prob = outputs[j][i]
            key = tokens_to_str(tokens)

            raw_prob = np.exp(raw_log_prob)
            gcd_prob = np.exp(gcd_prog_prob)

            out, raw_prob, gcd_prob, count = prob_dict[key]
            prob_dict[key] = (out, raw_prob, gcd_prob, count + 1)
            
        prob_dicts.append(prob_dict)

    orig_prob_sum = sum(v[1] for v in global_prob_dict.values())

    kls = []
    for d in prob_dicts:
        counts = [v[3] / num_samples for v in d.values()]
        orig_probs = [(v[1] / orig_prob_sum) for v in d.values()]

        kl = scipy.stats.entropy(counts, orig_probs)
        kls.append(kl)
    
    return kls

def analyze_output_file(output_file, plot_prefix=None, num_samples=None, num_steps=None):
    """
    Analyze JSONL output file and generate plots
    
    Args:
        output_file (str): Path to the output JSONL file
        plot_prefix (str, optional): Prefix for saved plot files. Defaults to file name without extension.
        num_samples (int, optional): Number of samples in the data. Calculated from file if None.
        num_steps (int, optional): Number of MCMC steps per sample. Calculated from file if None.
    """
    print(f"Analyzing output file: {output_file}")
    
    # Set plot_prefix based on output_file if not provided
    if plot_prefix is None:
        plot_prefix = os.path.splitext(os.path.basename(output_file))[0]
    
    # Read the JSONL file
    outputs = []
    with open(output_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            sample_output = []
            for item in sample:
                generated_sentence = item["generated_sentence"]
                tokens = torch.tensor(item["tokens"])
                raw_prob = item["raw_probability"]
                adj_prob = item["adjusted_probability"]
                sample_output.append((generated_sentence, tokens, raw_prob, adj_prob))
            outputs.append(sample_output)
    
    # Calculate num_samples and num_steps if not provided
    if num_samples is None:
        num_samples = len(outputs)
    if num_steps is None:
        num_steps = len(outputs[0]) if outputs else 0
    
    print(f"Found {num_samples} samples, each with {num_steps} ASAp steps")
    
    # Calculate KL divergence
    kls = calculate_kl_divergence(outputs, num_samples, num_steps)
    
    # Generate KL divergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(kls)), kls, '--b', marker='o')
    plt.title('KL Divergence over ASAp Steps')
    plt.xlabel('ASAp Step')
    plt.ylabel('KL Divergence')
    plt.grid(True, linestyle='--', alpha=0.7)
    kl_plot_path = f'{plot_prefix}_kl_divergence_plot.png'
    plt.savefig(kl_plot_path)
    plt.close()
    print(f"KL divergence plot saved as {kl_plot_path}")
    
    # Generate probability distribution plot
    unique_outputs = {}
    for sample in outputs:
        for out, tokens, raw_prob, adj_prob in sample:
            key = tokens_to_str(tokens)
            if key not in unique_outputs:
                unique_outputs[key] = (out, 1)
            else:
                out, count = unique_outputs[key]
                unique_outputs[key] = (out, count + 1)
    
    # Sort by frequency
    sorted_outputs = sorted(unique_outputs.items(), key=lambda x: x[1][1], reverse=True)
    top_n = min(10, len(sorted_outputs))
    
    if top_n > 0:
        plt.figure(figsize=(12, 8))
        labels = [f"{item[1][0][:20]}..." for item in sorted_outputs[:top_n]]
        counts = [item[1][1] for item in sorted_outputs[:top_n]]
        
        plt.bar(range(top_n), counts)
        plt.xticks(range(top_n), labels, rotation=45, ha='right')
        plt.title(f'Top {top_n} Most Frequent Outputs')
        plt.xlabel('Generated Output')
        plt.ylabel('Frequency')
        plt.tight_layout()
        freq_plot_path = f'{plot_prefix}_output_frequency.png'
        plt.savefig(freq_plot_path)
        plt.close()
        print(f"Output frequency plot saved as {freq_plot_path}")
    
    # Calculate and print summary statistics
    total_generations = num_samples * num_steps
    unique_count = len(unique_outputs)
    most_common = sorted_outputs[0] if sorted_outputs else None
    
    print(f"Summary Statistics:")
    print(f"- Total generations: {total_generations}")
    print(f"- Unique outputs: {unique_count} ({unique_count/total_generations:.2%} of total)")
    if most_common:
        print(f"- Most common output: '{most_common[1][0][:50]}...' (frequency: {most_common[1][1]})")
    
    return kls

def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset and grammar
    # prompt, grammar_str = load_dataset_and_grammar()
    
    with open("data/sygus/prompts/qm_max3.txt", "r") as f:
        prompt = f.read().strip()

    # grammar_path = "tmp.lark"
    grammar_path = "data/sygus/grammar/qm_max3.lark"

    # Prepare input for generation
    input_ids, attention_mask = prepare_input(tokenizer, prompt)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    
    # Initialize monitor and processors
    monitor = LLGuidanceMonitor.from_tokenizer(grammar_path, tokenizer)
    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList([inf_nan_remove_processor])
    
    # Run MCMC inference
    outputs = []
    for _ in tqdm(range(NUM_SAMPLES), desc="Running Inference"):
        adaptive_mask = AdaptiveMaskTrie()
        outputs_single_iter = []
        for _ in range(NUM_STEPS):
            # Generate sequences
            output = model.generate(
                input_ids,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                logits_processor=logits_processors,
                num_return_sequences=1,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                output_logits=True,
                output_scores=True,
                jump_forward=False,
                monitor=monitor,
                adaptive_mask=adaptive_mask,
            )

            # Detokenize generate output
            input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]

            generated_tokens = output.sequences[:, input_length:]
            generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            raw_prob = logprob_from_logits(output.logits, generated_tokens)
            gcd_prob = logprob_from_logits(output.scores, generated_tokens)

            outputs_single_iter.append([generations[0], generated_tokens, raw_prob, gcd_prob])
        outputs.append(outputs_single_iter)
    
    # Save outputs to JSONL file
    write_quadruples_to_jsonl(outputs, OUTPUT_FILE)
    print(f"Outputs saved to {OUTPUT_FILE}")
    
    # Calculate KL divergence and plot
    kls = calculate_kl_divergence(outputs, NUM_SAMPLES, NUM_STEPS)
    
    # Generate and save plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(kls)), kls, '--b', marker='o')
    plt.title('KL Divergence over MCMC Steps')
    plt.xlabel('MCMC Step')
    plt.ylabel('KL Divergence')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{ID}_kl_divergence_plot_{VERSION}.png')
    print("KL divergence plot saved as kl_divergence_plot.png")
    plt.close()

if __name__ == "__main__":
    main()
