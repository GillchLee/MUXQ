import torch
from tqdm import tqdm
#from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

from gpt2 import GPT2LMHeadModel, GPT2Config, OutlierTracer

## import custom model
## in custom model, Conv1D is replaced with Custom Conv1D on MLP, attn projection

from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import argparse 
import pdb



def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--quant_act_config", type=str, default=None)
	parser.add_argument("--quant_weight_config", type=str, default=None)
	return parser.parse_args()



device = "cuda"
#model_path = "/home/ssl/.cache/huggingface/hub/models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519"
model_path = './models/gpt2'
#model_path = 'openai-community/gpt2-medium'
#model_path = 'openai-community/gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_path,).to(device)
#model = model.half() ### to make models to fp16
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_length = model.config.n_positions

tracer = OutlierTracer.get_instance()
tracer.initialize(model)

#pdb.set_trace()
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")



 
stride = 512
seq_len = encodings.input_ids.size(1)



### evaluation ###

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())

print(ppl)