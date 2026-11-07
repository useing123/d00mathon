#!/usr/bin/env python3
"""Jev-compatible decision server (POST /v1/decisions {model, state, questions} -> {answers}) on a GPU pod.

The client is always cqa/jev.py (CQA_JEV_BASE_URL=http://127.0.0.1:8001/v1/decisions), so every decision method
sees the SAME questions/state as Jev. LitJev serves this contract natively (/v1/systemone) and is not wrapped here.

    # (3) our vLLM guided_choice decision mode: letters A.. per option, one constrained token, top-20 logprobs
    python decision_server.py vllmchoice --vllm http://127.0.0.1:8000/v1 --model qwen3.8-27b-fp8
    # (2) qwen-jev-like (github.com/sting8k/qwen-jev-like, MIT): in-process vLLM, single-pass trie logprob readout
    PYTHONPATH=/root/qwen-jev-like python decision_server.py qjl --model Qwen/Qwen3.8-27B-FP8
    # (4) openjev (huggingface.co/AlexWortega/openjev, MIT): NLI cross-encoder, P(entailment) per option
    PYTHONPATH=<snapshot>/code python decision_server.py openjev --model <snapshot> --subfolder qwen3.5-4b-nli-v5
"""
import argparse
import concurrent.futures as cf
import json
import math
import string
import time
import urllib.request

LETTERS = string.ascii_uppercase


def options_of(q):
    """[(key, description)] in the order the question lists them; score keys are level indices '0'..'n-1'."""
    c = q.get("criteria")
    if q["type"] == "score":
        return [(str(i), d) for i, d in enumerate(c)]
    if q["type"] == "noul":
        c = c or {}
        return [("true", c.get("true", "yes")), ("false", c.get("false", "no"))]
    return list(c.items())


def to_answer(q, probs):
    """{key: p} -> Jev answer dict (the shape cqa.jev.parse_answers reads)."""
    s = sum(probs.values()) or 1.0
    probs = {k: v / s for k, v in probs.items()}
    best = max(probs, key=probs.get)
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs.get("true", 0.0)}
    if q["type"] == "score":
        return {"type": "score", "score": sum(int(k) * v for k, v in probs.items()), "confidence": probs[best],
                "probabilities": probs}
    return {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}


class VllmChoice:
    """One chat call per question: state first (prefix cache), options as letters, structured_outputs choice."""

    def __init__(self, a):
        self.url, self.model, self.key = a.vllm.rstrip("/"), a.model, a.key
        self.pool = cf.ThreadPoolExecutor(32)

    def _one(self, state, q):
        opts = options_of(q)
        L = LETTERS[: len(opts)]
        user = (f"{q.get('instructions', '')}\n\nВарианты ответа:\n"
                + "\n".join(f"{x}) {d}" for x, (_, d) in zip(L, opts, strict=False))
                + "\n\nОтветь одной буквой варианта.")
        body = {"model": self.model, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
                "messages": [{"role": "system", "content": state}, {"role": "user", "content": user}],
                "structured_outputs": {"choice": list(L)},
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            ch = json.load(r)["choices"][0]
        top = ch["logprobs"]["content"][0]["top_logprobs"]
        lp = {t["token"].strip(): math.exp(t["logprob"]) for t in top}
        probs = {k: lp.get(x, 0.0) for x, (k, _) in zip(L, opts, strict=False)}
        mass = sum(probs.values())
        if mass == 0:   # emitted letter only (raw logprobs had no option in the top-20)
            em = ch["message"]["content"].strip()
            probs = {k: float(x == em) for x, (k, _) in zip(L, opts, strict=False)}
        a = to_answer(q, probs)
        a["in_set_mass"] = round(mass, 4)
        return a

    def run(self, state, questions):
        futs = {k: self.pool.submit(self._one, state, q) for k, q in questions.items()}
        return {k: f.result() for k, f in futs.items()}


class QwenJevLike:
    def __init__(self, a):
        import threading

        from core.jev_engine import JevEngine
        from transformers import AutoTokenizer
        from vllm import LLM
        llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=0.9, enable_prefix_caching=True,
                  language_model_only=True, trust_remote_code=True, max_num_seqs=64)
        self.eng = JevEngine(llm, AutoTokenizer.from_pretrained(a.model))
        self.lock = threading.Lock()

    def run(self, state, questions):
        with self.lock:
            return self.eng.run(state, questions)["answers"]


class OpenJevBackend:
    def __init__(self, a):
        import threading

        import torch
        from openjev_decide import RUBRIC_MARK, OpenJev
        self.mark = RUBRIC_MARK
        self.m = OpenJev.from_pretrained(a.model, subfolder=a.subfolder, device="cuda", dtype=torch.bfloat16)
        self.lock = threading.Lock()

    def run(self, state, questions):
        qs, keys = [], list(questions)
        for k in keys:
            q = questions[k]
            opts = options_of(q)
            labels = ["yes" if o == "true" else "no" if o == "false" else o for o, _ in opts]
            qs.append({"type": q["type"], "options": labels,
                       "instructions": q.get("instructions", "") + self.mark
                       + json.dumps({lab: d for lab, (_, d) in zip(labels, opts, strict=False)}, ensure_ascii=False)})
        with self.lock:
            out = self.m.decide(state, qs)
        ans = {}
        for k, q, o in zip(keys, qs, out, strict=False):
            if q["type"] == "noul":
                ans[k] = to_answer(questions[k], {"true": o["noul"], "false": 1 - o["noul"]})
            else:
                ans[k] = to_answer(questions[k], o["probabilities"])
        return ans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["vllmchoice", "qjl", "openjev"])
    p.add_argument("--model", required=True)
    p.add_argument("--vllm", default="http://127.0.0.1:8000/v1")
    p.add_argument("--key", default="none")
    p.add_argument("--subfolder")
    p.add_argument("--max-len", type=int, default=16384)
    p.add_argument("--port", type=int, default=8001)
    a = p.parse_args()
    eng = {"vllmchoice": VllmChoice, "qjl": QwenJevLike, "openjev": OpenJevBackend}[a.backend](a)

    import uvicorn
    from fastapi import FastAPI, Request
    app = FastAPI()

    @app.post("/v1/decisions")
    async def decisions(req: Request):
        import asyncio
        b = await req.json()
        t = time.time()
        ans = await asyncio.get_running_loop().run_in_executor(None, eng.run, b["state"], b["questions"])
        return {"model": b.get("model"), "answers": ans, "usage": {"latency_s": round(time.time() - t, 3)}}

    @app.get("/health")
    def health():
        return {"ok": True, "backend": a.backend, "model": a.model}

    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
