import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from alignment.monitor.grammar import CFGMonitor
from alignment.models import TransformersModel

NUM_ITER = 10
# MODEL_ID = "TinyLlama/TinyLlama_v1.1"
MODEL_ID = "Salesforce/codegen-350M-multi"
GRAMMAR_PATH = "examples/test/binary_len_5_0.ebnf"
TRIE_PATH = "tries/binary_len_5_0_trie.json"
DEVICE = "cpu"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0

device = torch.device(DEVICE)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.to(device)
model.to(dtype=DTYPE)
model.resize_token_embeddings(len(tokenizer))
model = torch.compile(model, mode='reduce-overhead', fullgraph=True)

model = TransformersModel(model, tokenizer)


# Load EBNF grammar
# grammar_str = """
#     ?start: sum
#         | NAME "=" sum    

#     ?sum: product
#         | sum "+" product   
#         | sum "-" product   

#     ?product: atom
#         | product "*" atom  
#         | product "/" atom 

#     ?atom: NUMBER           
#         | "-" atom        
#         | NAME            
#         | "(" sum ")"

#     %import common.CNAME -> NAME
#     %import common.NUMBER
#     %import common.WS_INLINE

#     %ignore WS_INLINE
# """

grammar_str = """
    ?start : "00000"
        | "11111"
"""

# Initialize logits processor for the grammar
inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
logits_processors = LogitsProcessorList([inf_nan_remove_processor])

# Tokenize prompt into ids
prompt = "Generate any equation"
decode_output = tokenizer(
    [prompt], add_special_tokens=False, return_tensors="pt", padding=True
)
print(decode_output)
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
        cache_implementation="static",
        attention_mask=attention_mask,
        jump_forward=False,
        monitor=monitor
    )

    monitor.reset()

    # Detokenize generate output
    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    outputs.append(generations[0])

print(outputs)