import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor

from alignment.monitor.grammar import CFGMonitor
from alignment.monitor.adaptive_utils import AdaptiveMaskTrie
from alignment.models import TransformersModel

NUM_ITER = 3
# MODEL_ID = "TinyLlama/TinyLlama_v1.1"
# MODEL_ID = "Salesforce/codegen-350M-multi"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
GRAMMAR_PATH = "examples/test/binary_len_5_0.ebnf"
TRIE_PATH = "tries/binary_len_5_0_trie.json"
DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0

CACHE_DIR = "/trunk/model-hub"

device = torch.device(DEVICE)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model.to(device)
model.to(dtype=DTYPE)
model.resize_token_embeddings(len(tokenizer))
# model = torch.compile(model, mode='reduce-overhead', fullgraph=True)

model = TransformersModel(model, tokenizer)

# Load EBNF grammar
# with open("/home/kangheepark/xgrammar/grammars/go_grammar.lark", "r") as f:
#     grammar_str = f.read()
with open("/home/kangheepark/syncode/syncode/parsers/grammars/java_grammar.lark", "r") as f:
    grammar_str = f.read()
# with open("examples/test.lark", "r") as f:
#     grammar_str = f.read()

# Initialize logits processor for the grammar
inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
logits_processors = LogitsProcessorList([inf_nan_remove_processor])

# Tokenize prompt into ids
# prompt = '''from typing import List

# def has_close_elements(numbers: List[float], threshold: float) -> bool:
#     """ 
#     Check if in given list of numbers, are any two numbers closer to each other than given threshold. 
#     >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False 
#     >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True 
#     """
# '''

prompt = "Give me java code to find the factorial of a number."

if 'instruct' in MODEL_ID.lower():
    messages = [
        {"role": "system", "content": "You are a skilled java programming assistant. Give me java code without any syntax errors, without any explanation."},
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

decode_output = tokenizer(
    [prompt], add_special_tokens=False, return_tensors="pt", padding=True
)
input_ids = decode_output["input_ids"]
input_ids = input_ids.to(model.device)

attention_mask = decode_output["attention_mask"]
attention_mask.to(model.device)

start = time.time()

monitor = CFGMonitor.from_tokenizer(grammar_str, tokenizer)
adaptive_mask = AdaptiveMaskTrie(batch_size = input_ids.shape[0])

end = time.time()
elapsed = end - start

print("Precomputing: ", elapsed)

# Inference Loop
outputs = []
for _ in tqdm(range(NUM_ITER), desc="Running Inference"):
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
        output_scores=True,
        attention_mask=attention_mask,
        repetition_penalty=REPETITION_PENALTY,
        # jump_forward=False,
        monitor=monitor
        # adaptive_mask=adaptive_mask
    )

    # Detokenize generate output
    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    outputs.append(generations[0])

    print(generations[0])

# print(outputs)
