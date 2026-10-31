"""DP-PCD G0b (inference only): public-nucleus variant + mixture-teacher accuracy at eps=4.
Units: mu2_t = 2*D_alpha/alpha (two-sided), same as dpcd.efficiency. eps=4 RDP: alpha=7, mu_rdp=0.788 -> record budget mu2=0.621
(= dpcd.accounting.rdp_budget_at_order(4,1e-6,7)*2/7). Support: q0's top-p nucleus / top-k (public), both q and q0 restricted+renormalized.
"""
import os, re, json, time, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
M = os.path.expanduser("~/dppcd_g0/Qwen3-1.7B")
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
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
STOP = torch.tensor([tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")], device="cuda")
MAXNEW = 384; ALPHAS = (22.0, 7.0, 4.0); MU2_BUDGET = 0.788**2
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

def support_mask(l0, sup):
    if sup == "full": return None
    kind, v = sup[0], float(sup[1:])
    if kind == "k":
        th = l0.topk(int(v), -1).values[:, -1:]; return l0 >= th
    s, idx = l0.sort(-1, descending=True); c = s.exp().cumsum(-1)
    keep = (c - s.exp()) < v
    return torch.zeros_like(keep).scatter(-1, idx, keep)
def restrict(l, mask):
    if mask is None: return l
    l = l.masked_fill(~mask, float("-inf")); return l - torch.logsumexp(l, -1, keepdim=True)
def mu2(la, lb, al):  # two-sided, la/lb logprobs on common support; -inf entries excluded
    def D(x, y):
        z = torch.where(torch.isfinite(x), al*x + (1-al)*y, torch.full_like(x, float("-inf")))
        return torch.logsumexp(z, -1)/(al-1)
    return (2*torch.maximum(D(la, lb), D(lb, la)).clamp_min(0)/al)
def mix(l1, l0, lam):
    lam = lam[:, None]
    return torch.logaddexp(torch.log(lam.clamp_min(1e-30)) + l1, torch.log1p(-lam.clamp_max(1-1e-7)) + l0)
def lam_for(l1, l0, bound, al, iters=14):
    full = mu2(l1, l0, al); ok = full <= bound
    lo = torch.zeros_like(bound); hi = torch.ones_like(bound)
    for _ in range(iters):
        mid = (lo+hi)/2; good = mu2(mix(l1, l0, mid), l0, al) <= bound
        lo = torch.where(good, mid, lo); hi = torch.where(good, hi, mid)
    return torch.where(ok, torch.ones_like(lo), lo)

@torch.no_grad()
def run(P0, PR, sup, mode, greedy, bs, b2=None, profile=False):
    """mode: 'q0' sample q0 only (profile) | 'fixed' mixture with per-token bound b2 | 'filter' adaptive budget."""
    res = []
    for i in range(0, len(P0), bs):
        st = [tok(p[i:i+bs], return_tensors="pt", padding=True).to("cuda") for p in (P0, PR)]
        B = st[0].input_ids.shape[0]; caches = [None, None]; am = [s.attention_mask for s in st]; inp = [s.input_ids for s in st]
        done = torch.zeros(B, dtype=torch.bool, device="cuda"); spent = torch.zeros(B, device="cuda")
        out = [[] for _ in range(B)]; stats = {k: [[] for _ in range(B)] for k in ("lr", "out", "a22", "a7", "a4")}
        for t in range(MAXNEW):
            ls = []
            for j in range(2):
                pos = (am[j].cumsum(-1) - 1).clamp_min(0)[:, -inp[j].shape[1]:]
                o = model(input_ids=inp[j], attention_mask=am[j], position_ids=pos, past_key_values=caches[j], use_cache=True)
                caches[j] = o.past_key_values; ls.append(torch.log_softmax(o.logits[:, -1].float(), -1))
            l1f, l0f = ls[1], ls[0]; mask = support_mask(l0f, sup)
            l1, l0 = restrict(l1f, mask), restrict(l0f, mask)
            if mode == "q0": lp = l0
            else:
                if mode == "fixed": bound = torch.full((B,), b2, device="cuda")
                else: bound = (MU2_BUDGET - spent).clamp_min(0)
                lam = lam_for(l1, l0, bound, 7.0); lp = mix(l1, l0, lam)
                cost = torch.where(lam > 0, mu2(lp, l0, 7.0), torch.zeros_like(lam))
                spent += torch.where(done, torch.zeros_like(cost), cost if mode == "filter" else bound)
            a = lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1)[:, 0]
            if profile:
                lr = (l1.gather(1, a[:, None]) - l0.gather(1, a[:, None]))[:, 0]
                outm = (1 - l1f.exp().masked_fill(~mask, 0).sum(-1)) if mask is not None else torch.zeros(B, device="cuda")
                d = {al: mu2(l1, l0, al) for al in ALPHAS}
                for b in (~done).nonzero()[:, 0].tolist():
                    stats["lr"][b].append(lr[b].item()); stats["out"][b].append(outm[b].item())
                    for al, k in zip(ALPHAS, ("a22", "a7", "a4")): stats[k][b].append(d[al][b].item())
            for b in (~done).nonzero()[:, 0].tolist(): out[b].append(a[b].item())
            done |= torch.isin(a, STOP)
            if done.all(): break
            inp = [a[:, None], a[:, None]]; am = [torch.cat([m, torch.ones(B, 1, dtype=m.dtype, device="cuda")], 1) for m in am]
        for b in range(B):
            res.append({"text": tok.decode(out[b], skip_special_tokens=True), "len": len(out[b]), "spent": spent[b].item(),
                        **({k: stats[k][b] for k in stats} if profile else {})})
        del caches; torch.cuda.empty_cache()
    return res

SUPS = os.environ.get("SUPS", "full,p0.9,p0.95,p0.99").split(",")
# (1) profile: 500 train records x 4 rollouts from (truncated) q0
tr = ds["train"].shuffle(seed=0).select(range(500))
P0 = [prompt(x["question"]) for x in tr for _ in range(4)]; PR = [prompt(x["question"], x["answer"]) for x in tr for _ in range(4)]
G = [gold(x["answer"]) for x in tr for _ in range(4)]
meanlen = {}
for sup in SUPS:
    if os.path.exists(f"{OUT}/profile_{sup}.npz"): meanlen[sup] = float(np.load(f"{OUT}/profile_{sup}.npz")["meanlen"]); continue
    r = run(P0, PR, sup, "q0", False, 64, profile=True)
    cat = lambda k: np.array([v for x in r for v in x[k]], dtype=np.float32)
    meanlen[sup] = float(np.mean([x["len"] for x in r]))
    np.savez_compressed(f"{OUT}/profile_{sup}.npz", lr=cat("lr"), outside=cat("out"), d_mu2_a22=cat("a22"), d_mu2_a7=cat("a7"),
                        d_mu2_a4=cat("a4"), meanlen=meanlen[sup], acc=np.mean([eq(ans(x["text"]), g) for x, g in zip(r, G)]))
    log("profile", sup, "meanlen", meanlen[sup])
# (2) mixture accuracy at eps=4 on GSM8K test 500
te = ds["test"].select(range(500)); T0 = [prompt(x["question"]) for x in te]; TR = [prompt(x["question"], x["answer"]) for x in te]
GT = [gold(x["answer"]) for x in te]
accf = f"{OUT}/mixture_acc.json"; acc = json.load(open(accf)) if os.path.exists(accf) else {}
b2fit = MU2_BUDGET / meanlen["full"]
for sup in os.environ.get("MIXSUPS", "full,p0.95").split(","):
    for name, mode, b2 in [("q0", "q0", None), ("fixed_b2_budget", "fixed", b2fit), ("fixed_b2_0.01", "fixed", 0.01), ("filter", "filter", None)]:
        for dec in ("greedy", "sample"):
            key = f"{sup}|{name}|{dec}"
            if key in acc: continue
            r = run(T0, TR, sup, mode, dec == "greedy", 100, b2=b2)
            acc[key] = {"acc": float(np.mean([eq(ans(x["text"]), g) for x, g in zip(r, GT)])), "b2": b2,
                        "mean_len": float(np.mean([x["len"] for x in r])), "mean_spent_mu2": float(np.mean([x["spent"] for x in r])),
                        "share_over_budget": float(np.mean([x["spent"] > MU2_BUDGET + 1e-6 for x in r]))}
            json.dump(acc, open(accf, "w"), indent=1); log(key, acc[key])
log("DONE")
