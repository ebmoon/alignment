import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from alignment.monitor.grammar import CFGMonitor
from alignment.models import TransformersModel

NUM_ITER = 1
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
GRAMMAR_PATH = "examples/test/binary_len_5_0.ebnf"
TRIE_PATH = "tries/binary_len_5_0_trie.json"
DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 8
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0

cache_dir = "/trunk/model-hub"

device = torch.device(DEVICE)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=cache_dir)
model.to(device)
model.to(dtype=DTYPE)
model.resize_token_embeddings(len(tokenizer))
# model = torch.compile(model, mode='reduce-overhead', fullgraph=True)

model = TransformersModel(model, tokenizer)


grammar_str = """
start: "[" INT (", " INT)* "]"
%import common.INT
"""

# Initialize logits processor for the grammar
inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
logits_processors = LogitsProcessorList([inf_nan_remove_processor])

# Tokenize prompt into ids
prompt = """Generate an arbitrary list of integers that sum to 50."""

decode_output = tokenizer(
    [prompt], add_special_tokens=False, return_tensors="pt", padding=True
)
input_ids = decode_output["input_ids"]
input_ids = input_ids.to(model.device)

attention_mask = decode_output["attention_mask"]
attention_mask.to(model.device)

monitor = CFGMonitor.from_tokenizer(grammar_str, tokenizer)

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
        top_p=TOP_P,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        logits_processor=logits_processors,
        repetition_penalty=REPETITION_PENALTY,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        attention_mask=attention_mask,
        # jump_forward=False,
        monitor=monitor
    )

    # Detokenize generate output
    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    outputs.append(generations[0])

print(outputs)