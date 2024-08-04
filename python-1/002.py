import torch

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("TheBloke/llama2_7b_chat_uncensored-GPTQ")
# model = AutoModelForCausalLM.from_pretrained("TheBloke/llama2_7b_chat_uncensored-GPTQ")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  # Handle the case where GPU is not available (use float32 on CPU)
  pass

model_name_or_path = "TheBloke/llama2_7b_chat_uncensored-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="cpu",
                                             trust_remote_code=True,
                                             torch_dtype=torch.float32,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''### HUMAN:
{prompt}

### RESPONSE:

'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids  # .cuda()
print(input_ids.type())
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])
