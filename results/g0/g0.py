"""DP-PCD G0: divergence profile + teacher headroom. Inference only.
Choices: model Qwen/Qwen3-1.7B, thinking disabled; data GSM8K train (500 records for profile),
GSM8K test (500) for headroom; privileged r = the gold worked solution (reference answer text).
Student = q0 (no r) samples 4 rollouts T=1.0, max 384 new tokens. Teacher q = same model with r in context.
Per sampled token: lr = log q(a)-log q0(a); two-sided Renyi D_alpha at alphas from RDP orders; d_mu2 = 2D/alpha.
"""
import os, re, json, time, sys, math, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "4,7,10").split(",")]
NREC, K, MAXNEW = 500, 4, 384
M = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).cuda().eval()
ds = load_dataset("openai/gsm8k", "main")
SYS = "Solve the math problem. Reason step by step, then give the final answer as '#### <number>'."
def prompt(q, r=None):
    u = q if r is None else (f"{q}\n\nHere is a reference solution you may use (privileged):\n{r}")
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
def ans(t):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", t) or re.findall(r"(-?[\d,]*\.?\d+)", t)
    return m[-1].replace(",", "").rstrip(".") if m else None
def gold(s): return s.split("####")[-1].strip().replace(",", "")
def eq(a, b):
    try: return a is not None and abs(float(a) - float(b)) < 1e-6
    except: return False
@torch.no_grad()
def gen(prompts, n=1, greedy=False, bs=32):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i+bs], return_tensors="pt", padding=True).to("cuda")
        g = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=not greedy, temperature=1.0 if not greedy else None,
                           top_p=1.0 if not greedy else None, top_k=0 if not greedy else None, num_return_sequences=n,
                           pad_token_id=tok.pad_token_id)
        outs += [x[enc.input_ids.shape[1]:] for x in g]
    return outs
def strip(ids):
    ids = ids.tolist(); out = []
    for t in ids:
        if t == tok.pad_token_id: break
        out.append(t)
        if t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")): break
    return out
@torch.no_grad()
def logprobs(ptxt, resp):
    p = tok(ptxt, add_special_tokens=False).input_ids
    ids = torch.tensor([p + resp], device="cuda")
    lg = model(ids).logits[0, len(p)-1:len(p)-1+len(resp)].float()
    return torch.log_softmax(lg, -1)
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
# (b) headroom on GSM8K test 500, greedy
te = ds["test"].select(range(500))
res = {}
for name, ps in [("q0", [prompt(x["question"]) for x in te]), ("teacher_r", [prompt(x["question"], x["answer"]) for x in te])]:
    o = gen(ps, greedy=True)
    acc = np.mean([eq(ans(tok.decode(strip(y), skip_special_tokens=True)), gold(x["answer"])) for y, x in zip(o, te)])
    res[name] = float(acc); log("headroom", name, acc)
json.dump(res, open(f"{OUT}/headroom.json", "w"), indent=1)
# (a) profile on GSM8K train 500 x 4
tr = ds["train"].shuffle(seed=0).select(range(NREC))
P0 = [prompt(x["question"]) for x in tr]; PR = [prompt(x["question"], x["answer"]) for x in tr]
rolls = gen(P0, n=K, bs=16); log("rollouts done", len(rolls))
LR, D, REC, POS, CORR = [], {a: [] for a in ALPHAS}, [], [], []
for j, y in enumerate(rolls):
    i = j // K; resp = strip(y)
    if not resp: continue
    CORR.append(eq(ans(tok.decode(resp, skip_special_tokens=True)), gold(tr[i]["answer"])))
    l0 = logprobs(P0[i], resp); l1 = logprobs(PR[i], resp)
    a = torch.tensor(resp, device="cuda")
    LR.append((l1.gather(1, a[:, None]) - l0.gather(1, a[:, None]))[:, 0].cpu().numpy())
    for al in ALPHAS:
        f = torch.logsumexp(al*l1 + (1-al)*l0, -1)/(al-1); b = torch.logsumexp(al*l0 + (1-al)*l1, -1)/(al-1)
        D[al].append((2*torch.maximum(f, b).clamp_min(0)/al).cpu().numpy())
    REC.append(np.full(len(resp), i)); POS.append(np.arange(len(resp)))
    if j % 200 == 0: log("scored", j)
np.savez_compressed(f"{OUT}/profile.npz", lr=np.concatenate(LR), rec=np.concatenate(REC), pos=np.concatenate(POS),
                    **{f"d_mu2_a{a:g}": np.concatenate(D[a]) for a in ALPHAS}, rollout_correct=np.array(CORR))
json.dump({"rollout_acc_q0_T1": float(np.mean(CORR)), "n_tokens": int(sum(map(len, LR))), "secs": time.time()-t0},
          open(f"{OUT}/meta.json", "w"), indent=1)
log("DONE")
