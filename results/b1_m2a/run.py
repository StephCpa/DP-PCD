"""B1 / M2a runner. usage: run.py b1 LR SEED OUT | gen SPLIT SEED OUT | m2a LR SEED SYN LAMHAT OUT | base OUT
Eval on dev (500) or test (1319) via env EVAL=dev|test|both."""
import os, re, sys, json, math, time, random, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/dppcd_g0"))
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from dpcd import accounting as A
M = os.path.expanduser("~/dppcd_g0/Qwen3-1.7B")
EPS, DELTA, N, MAXNEW, NSYN = 4.0, 1e-6, 4096, 320, 2
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
SYS = "Solve the math problem. Reason step by step, then give the final answer as '#### <number>'."
def prompt(q, r=None):
    u = q if r is None else (f"{q}\n\nHere is a reference solution you may use (privileged):\n{r}")
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
clean = lambda s: re.sub(r"<<.*?>>", "", s)
def ans(t):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", t) or re.findall(r"(-?[\d,]*\.?\d+)", t)
    return m[-1].replace(",", "").rstrip(".") if m else None
def gold(s): return s.split("####")[-1].strip().replace(",", "")
def eq(a, b):
    try: return a is not None and abs(float(a) - float(b)) < 1e-6
    except: return False
ds = load_dataset("openai/gsm8k", "main"); tr = ds["train"].shuffle(seed=0)
PRIV, DEV, PROXY = tr.select(range(N)), tr.select(range(N, N+500)), tr.select(range(N+500, N+1000))
TEST = ds["test"]
IMEND = tok.convert_tokens_to_ids("<|im_end|>")
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def load(lora=True):
    m = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).cuda()
    if lora:
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
        for n_, p in m.named_parameters():
            if p.requires_grad: p.data = p.data.float()
    return m
def encode(q, resp_ids):
    p = tok(prompt(q), add_special_tokens=False).input_ids
    ids = p + resp_ids; lab = [-100]*len(p) + resp_ids
    return torch.tensor([ids], device="cuda"), torch.tensor([lab], device="cuda")
def resp_of(r): return tok(clean(r), add_special_tokens=False).input_ids + [IMEND]

@torch.no_grad()
def evaluate(m, split, bs=96):
    m.eval(); data = DEV if split == "dev" else TEST; ps = [prompt(x["question"]) for x in data]; corr = []
    for i in range(0, len(ps), bs):
        enc = tok(ps[i:i+bs], return_tensors="pt", padding=True).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = m.generate(**enc, max_new_tokens=384, do_sample=False, pad_token_id=tok.pad_token_id)
        for y, x in zip(g[:, enc.input_ids.shape[1]:], data.select(range(i, min(i+bs, len(ps))))):
            corr.append(int(eq(ans(tok.decode(y, skip_special_tokens=True)), gold(x["answer"]))))
    a = float(np.mean(corr)); return {"acc": a, "se": math.sqrt(a*(1-a)/len(corr)), "correct": corr}

def do_eval(m, out, extra):
    res = dict(extra)
    for sp in {"dev": ["dev"], "test": ["test"], "both": ["dev", "test"]}[os.environ.get("EVAL", "dev")]:
        res[sp] = evaluate(m, sp); log("eval", sp, res[sp]["acc"], "+-", res[sp]["se"])
    json.dump(res, open(out, "w")); log("DONE", out)

def b1(lr, seed, out):
    seed_all(seed); m = load(); params = [p for p in m.parameters() if p.requires_grad]
    bsz, epochs, C = 256, 4, 1.0; q = bsz / N; steps = epochs * math.ceil(N / bsz)
    z = A.dpsgd_noise_multiplier(EPS, DELTA, q, steps); e = A.dpsgd_epsilon(z, q, steps, DELTA)
    assert e <= EPS, (z, e); log("B1 z", z, "eps", e, "steps", steps)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0); rng = np.random.default_rng(seed)
    data = [encode(x["question"], resp_of(x["answer"])) for x in PRIV]; m.train()
    for st in range(steps):
        idx = np.where(rng.random(N) < q)[0]; acc = [torch.zeros_like(p) for p in params]; norms = []
        for i in idx:
            ids, lab = data[i]
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = m(input_ids=ids, labels=lab).loss
            g = torch.autograd.grad(loss, params)
            nrm = torch.sqrt(sum((x.double()**2).sum() for x in g)).item(); norms.append(nrm); s = min(1.0, C/(nrm+1e-12))
            for a_, x in zip(acc, g): a_.add_(x, alpha=s)
        for p, a_ in zip(params, acc):
            p.grad = (a_ + torch.randn_like(a_) * z * C) / bsz  # fixed expected batch size denominator
        opt.step(); opt.zero_grad(set_to_none=True)
        log(f"B1 step {st+1}/{steps} B={len(idx)} med_norm={np.median(norms):.3f} last_loss={loss.item():.3f}")
    do_eval(m, out, {"arm": "B1", "lr": lr, "seed": seed, "z": z, "eps": e, "steps": steps})

# ---- M2a mechanism ----
def mu2_64(la, lb, al=7.0):
    la, lb = la.double(), lb.double()
    D = lambda x, y: torch.logsumexp(al*x + (1-al)*y, -1)/(al-1)
    return 2*torch.maximum(D(la, lb), D(lb, la)).clamp_min(0)/al
def mu2_32(la, lb, al=7.0):
    D = lambda x, y: torch.logsumexp(al*x + (1-al)*y, -1)/(al-1)
    return 2*torch.maximum(D(la, lb), D(lb, la)).clamp_min(0)/al
def mix(l1, l0, lam):
    lam = lam[:, None]; return torch.logaddexp(torch.log(lam.clamp_min(1e-30)) + l1, torch.log1p(-lam.clamp_max(1-1e-7)) + l0)
def lam_for(l1, l0, b, iters=16):
    lo = torch.zeros(l1.shape[0], device="cuda"); hi = torch.ones_like(lo)
    for _ in range(iters):
        mid = (lo+hi)/2; good = mu2_32(mix(l1, l0, mid), l0) <= 0.98*b
        lo = torch.where(good, mid, lo); hi = torch.where(good, hi, mid)
    lam = lo; cost = mu2_64(mix(l1, l0, lam), l0)
    for _ in range(30):  # strict float64 check; shrink until satisfied
        bad = cost > b
        if not bad.any(): break
        lam = torch.where(bad, lam*0.5, lam); cost = mu2_64(mix(l1, l0, lam), l0)
    lam = torch.where(cost > b, torch.zeros_like(lam), lam); cost = torch.where(cost > b, torch.zeros_like(cost), cost)
    return lam, cost

@torch.no_grad()
def gen(split, seed, out, bs=64):
    seed_all(seed); m = load(lora=False).eval(); data = {"priv": PRIV, "proxy": PROXY}[split]
    MU2 = A.mu_rdp(EPS, DELTA)[0]**2; b = MU2 / (NSYN*MAXNEW)
    log("gen", split, "mu2", MU2, "b", b)
    P0 = [prompt(x["question"]) for x in data for _ in range(NSYN)]; PR = [prompt(x["question"], clean(x["answer"])) for x in data for _ in range(NSYN)]
    rec = [i for i in range(len(data)) for _ in range(NSYN)]
    ledger = np.zeros(len(data), dtype=np.float64); outs, lams = [], []
    for i in range(0, len(P0), bs):
        st = [tok(p[i:i+bs], return_tensors="pt", padding=True).to("cuda") for p in (P0, PR)]
        B = st[0].input_ids.shape[0]; caches = [None, None]; am = [s.attention_mask for s in st]; inp = [s.input_ids for s in st]
        done = torch.zeros(B, dtype=torch.bool, device="cuda"); o_ = [[] for _ in range(B)]; lsum = torch.zeros(B, dtype=torch.float64, device="cuda")
        for t in range(MAXNEW):
            ls = []
            for j in range(2):
                pos = (am[j].cumsum(-1) - 1).clamp_min(0)[:, -inp[j].shape[1]:]
                o = m(input_ids=inp[j], attention_mask=am[j], position_ids=pos, past_key_values=caches[j], use_cache=True)
                caches[j] = o.past_key_values; ls.append(torch.log_softmax(o.logits[:, -1].float(), -1))
            l0, l1 = ls
            lam, cost = lam_for(l1, l0, b)
            a = torch.multinomial(mix(l1, l0, lam).exp(), 1)[:, 0]
            cost = torch.where(done, torch.zeros_like(cost), cost).cpu().numpy()
            for k in range(B):
                if not done[k]:
                    ledger[rec[i+k]] += cost[k]; o_[k].append(a[k].item()); lams.append(lam[k].item())
            done |= (a == IMEND) | (a == tok.eos_token_id)
            if done.all(): break
            inp = [a[:, None], a[:, None]]; am = [torch.cat([x, torch.ones(B, 1, dtype=x.dtype, device="cuda")], 1) for x in am]
        outs += o_; del caches; torch.cuda.empty_cache()
        assert (ledger <= MU2).all(), ("OVERSPEND", ledger.max(), MU2)  # strict, zero tolerance
        log(f"gen {i+B}/{len(P0)} mean_lam={np.mean(lams):.4f} ledger_max={ledger.max():.6f}/{MU2:.6f}")
    acc = float(np.mean([eq(ans(tok.decode(y, skip_special_tokens=True)), gold(data[r]["answer"])) for y, r in zip(outs, rec)]))
    json.dump({"split": split, "seed": seed, "b": b, "mu2_budget": MU2, "lam_mean": float(np.mean(lams)), "ledger_max": float(ledger.max()),
               "ledger_mean": float(ledger.mean()), "syn_acc": acc, "rec": rec, "toks": outs}, open(out, "w"))
    log("syn acc", acc, "lam_mean", np.mean(lams), "DONE", out)

MB = int(os.environ.get("M2A_MB", "4"))
def m2a(lr, seed, syn, lam_hat, out, bsz=16, epochs=2):
    seed_all(seed); m = load(); params = [p for p in m.parameters() if p.requires_grad]
    S = json.load(open(syn)); opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    ex = [(PRIV[r]["question"], y if (y and y[-1] in (IMEND, tok.eos_token_id)) else y) for r, y in zip(S["rec"], S["toks"]) if len(y) > 0]
    rng = np.random.default_rng(seed); tok.padding_side = "right"; m.train(); step = 0
    for ep in range(epochs):
        order = rng.permutation(len(ex))
        for i in range(0, len(order), bsz):
            batch = [encode(ex[j][0], ex[j][1]) for j in order[i:i+bsz]]
            L = max(x[0].shape[1] for x in batch)
            ids = torch.cat([torch.nn.functional.pad(x[0], (0, L-x[0].shape[1]), value=tok.pad_token_id) for x in batch])
            lab = torch.cat([torch.nn.functional.pad(x[1], (0, L-x[1].shape[1]), value=-100) for x in batch])
            att = (torch.arange(L, device="cuda")[None] < torch.tensor([x[0].shape[1] for x in batch], device="cuda")[:, None]).long()
            tgt = lab[:, 1:]; msk = tgt != -100; tg = tgt.clamp_min(0); ntok = msk.sum(); loss = 0.0
            # OOM fix (2026-11-01): micro-batches of MB with gradient accumulation; loss normalised by the
            # full-batch token count, so the gradient is identical to the original single 16-example step.
            for s0 in range(0, ids.shape[0], MB):
                sl = slice(s0, s0+MB)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad(), m.disable_adapter():
                        lq0 = torch.log_softmax(m(input_ids=ids[sl], attention_mask=att[sl]).logits[:, :-1].float(), -1).gather(-1, tg[sl][..., None])[..., 0]
                    lpi = torch.log_softmax(m(input_ids=ids[sl], attention_mask=att[sl]).logits[:, :-1].float(), -1).gather(-1, tg[sl][..., None])[..., 0]
                nll = -torch.logaddexp(math.log(lam_hat) + lpi, math.log1p(-lam_hat) + lq0)
                part = (nll*msk[sl]).sum()/ntok; part.backward(); loss += part.item(); del lq0, lpi, nll, part
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 20 == 0: log(f"M2a ep{ep} step {step} loss={loss:.4f}")
    tok.padding_side = "left"
    do_eval(m, out, {"arm": "M2a", "lr": lr, "seed": seed, "lam_hat": lam_hat, "syn": syn, "syn_lam_mean": S["lam_mean"], "syn_acc": S["syn_acc"]})

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "b1": b1(float(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif c == "gen": gen(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif c == "m2a": m2a(float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], float(sys.argv[5]), sys.argv[6])
    elif c == "base": do_eval(load(lora=False), sys.argv[2], {"arm": "base"})
