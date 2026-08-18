import torch
import os, numpy as np
from model import ModelArgs, Transformer

checkpoint_path = 'out/stories15M.pt'   # your actual path
device = 'cpu'                           # oracle stays on CPU, deterministic

checkpoint_dict = torch.load(checkpoint_path, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = Transformer(gptconf)

state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict, strict=True)   # fail loudly

model.eval()
torch.manual_seed(1337)

print("--- config ---")
print(gptconf)

print("\n--- architecture ---")
print(model)

print("\n--- module names ---")
for name, _ in model.named_modules():
    print(name)

# print(model.tok_embeddings.weight.data_ptr() == model.output.weight.data_ptr())

os.makedirs("dumps", exist_ok=True)

def make_hook(name):
    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            np.save(f"dumps/{name}.npy", out.detach().cpu().numpy())
    return hook

for name, module in model.named_modules():
    if name and "dropout" not in name:
        module.register_forward_hook(make_hook(name))

tokens = torch.tensor([[1]], dtype=torch.long)
with torch.no_grad():
    logits = model(tokens)

np.save("dumps/logits.npy", logits.detach().cpu().numpy())
print("dumped", len(os.listdir("dumps")), "tensors")

# print(model.tok_embeddings.weight[0][:5])

# print(model.layers[0].attention.wq.weight.flatten()[:5])