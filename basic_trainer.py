import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os

# ========= LOAD DATA =========
with open("/content/mtgencode/data/funny_training.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(chars)

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# ========= HYPERPARAMS (mtg-rnn style) =========
batch_size = 32
block_size = 128
n_embed = 256
n_hidden = 512
n_layers = 3
dropout = 0.5
lr = 2e-3

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= MODEL =========
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, num_layers=n_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

model = CharModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ========= BATCHING =========
def get_batch():
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# train
def save_checkpoint(step, loss):
    path = f"/content/checkpoint_{step}.pt"
    torch.save({
        "model": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "step": step,
        "loss": loss,
        "lr": optimizer.param_groups[0]['lr']
    }, path)
    print(f"Saved checkpoint: {path}")


#  load latest checkpoint automatically
def load_checkpoint():
    checkpoints = [f for f in os.listdir("/content") if f.startswith("checkpoint_")]

    if not checkpoints:
        return 0

    # sort by step number
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest = checkpoints[-1]

    ckpt = torch.load(f"/content/{latest}", map_location=device)
    model.load_state_dict(ckpt["model"])

    # restore learning rate too
    for g in optimizer.param_groups:
        g['lr'] = ckpt.get("lr", lr)

    print(f"Resumed from {latest} (step {ckpt['step']}, lr {optimizer.param_groups[0]['lr']})")
    return ckpt["step"]


#  learning rate scheduler (mtg-rnn didn't have this, but it fixes instability)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10000,   # every 10k steps
    gamma=0.5          # cut LR in half
)

start_step = load_checkpoint()

for step in range(start_step, start_step + 40000):

    x, y = get_batch()

    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()

    #  critical for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

    optimizer.step()
    scheduler.step()

    if step % 100 == 0:
        print(f"step {step} | loss {loss.item():.4f} | lr {optimizer.param_groups[0]['lr']:.6f}")

    #  save every 1000 steps
    if step % 5000 == 0:
        save_checkpoint(step, loss.item())
