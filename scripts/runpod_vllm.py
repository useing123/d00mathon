#!/usr/bin/env python3
"""Deploy / stop / inspect a vLLM OpenAI-compatible server on RunPod (GraphQL API).

    python scripts/runpod_vllm.py up      [--model Qwen/Qwen3-32B-AWQ] [--gpu "NVIDIA RTX A6000"] [--max-hours 1.5]
    python scripts/runpod_vllm.py status  [--pod ID]
    python scripts/runpod_vllm.py wait    [--pod ID]     # poll until /v1/models answers
    python scripts/runpod_vllm.py down    [--pod ID]     # terminate (deletes the pod)
    python scripts/runpod_vllm.py balance

Secrets: RUNPOD_API_KEY and CQA_LOCAL_LLM_TOKEN are read from env or ~/.hermes/.env (token is
generated + appended on first `up`). The pod id of the last `up` is stored in .cache/runpod_pod.json.
`up` also spawns a detached watchdog that terminates the pod after --max-hours no matter what.
Never prints secrets.
"""
import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ENV_FILE = Path.home() / ".hermes" / ".env"
STATE = Path(__file__).resolve().parent.parent / ".cache" / "runpod_pod.json"
API = "https://api.runpod.io/graphql"
POD_NAME = "cqa-vllm"
GPU_PREF = ["NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"]


def env(key):
    if os.environ.get(key):
        return os.environ[key]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def token():
    t = env("CQA_LOCAL_LLM_TOKEN")
    if not t:
        t = secrets.token_urlsafe(32)
        with ENV_FILE.open("a") as f:
            f.write(f"\nCQA_LOCAL_LLM_TOKEN={t}\n")
    return t


def gql(query, variables=None):
    req = urllib.request.Request(API, data=json.dumps({"query": query, "variables": variables or {}}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "cqa-runpod/1.0",
                                          "Authorization": f"Bearer {env('RUNPOD_API_KEY')}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        j = json.loads(r.read())
    if j.get("errors"):
        raise RuntimeError(json.dumps(j["errors"])[:800])
    return j["data"]


def balance():
    return gql("{ myself { clientBalance currentSpendPerHr } }")["myself"]


def pod_id(a):
    if getattr(a, "pod", None):
        return a.pod
    if STATE.exists():
        return json.loads(STATE.read_text())["id"]
    sys.exit("no pod id (pass --pod)")


def status(pid):
    return gql("query($i:String!){ pod(input:{podId:$i}){ id name desiredStatus costPerHr lastStatusChange "
               "runtime{ uptimeInSeconds } machine{ gpuDisplayName } } }", {"i": pid})["pod"]


def vllm_args(a):
    args = [f"--model {a.model}", f"--served-model-name {a.served_name}", f"--max-model-len {a.max_len}",
            "--gpu-memory-utilization 0.92", "--port 8000", f"--api-key {token()}",
            "--enable-auto-tool-choice --tool-call-parser hermes", "--max-num-seqs 32", "--enable-prefix-caching"]
    if a.no_think:  # Qwen3: disable <think> by default so outputs are comparable with the cloud path
        args.append("--default-chat-template-kwargs '{\"enable_thinking\": false}'")
    return " ".join(args + a.extra)


def up(a):
    gpus = [a.gpu] if a.gpu else GPU_PREF
    last = None
    for g in gpus:
        for cloud in ("COMMUNITY", "SECURE"):
            inp = {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": g, "name": POD_NAME,
                   "imageName": a.image, "containerDiskInGb": 60, "volumeInGb": 0, "ports": "8000/http",
                   "dockerArgs": vllm_args(a), "minVcpuCount": 4, "minMemoryInGb": 30,
                   "env": [{"key": "HF_HUB_ENABLE_HF_TRANSFER", "value": "1"}]}
            try:
                m = gql("mutation($in: PodFindAndDeployOnDemandInput){ podFindAndDeployOnDemand(input:$in){ id "
                        "costPerHr machine{ gpuDisplayName } } }", {"in": inp})["podFindAndDeployOnDemand"]
            except Exception as e:
                last = e
                print(f"  {g}/{cloud}: {str(e)[:160]}")
                continue
            m.update(created=time.time(), model=a.served_name, cloud=cloud,
                     url=f"https://{m['id']}-8000.proxy.runpod.net/v1")
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps(m))
            # watchdog: terminate after max-hours regardless of what happens to this process
            subprocess.Popen([sys.executable, __file__, "watchdog", "--pod", m["id"], "--max-hours", str(a.max_hours)],
                             start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(json.dumps(m, indent=1))
            return
    sys.exit(f"no GPU available: {last}")


def wait(a):
    pid, t0 = pod_id(a), time.time()
    url = f"https://{pid}-8000.proxy.runpod.net/v1/models"
    while time.time() - t0 < a.timeout:
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token()}", "User-Agent": "cqa"})
            with urllib.request.urlopen(req, timeout=20) as r:
                print(f"ready after {time.time() - t0:.0f}s:", r.read().decode()[:300])
                return
        except Exception as e:
            print(f"  {time.time() - t0:5.0f}s not ready ({str(e)[:60]})", flush=True)
        time.sleep(20)
    sys.exit("timeout")


def down(pid):
    gql("mutation($i:String!){ podTerminate(input:{podId:$i}) }", {"i": pid})
    time.sleep(3)
    try:
        s = status(pid)
    except Exception as e:
        s = f"gone ({str(e)[:80]})"
    print("after terminate:", s, "| account:", balance())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["up", "down", "status", "wait", "balance", "watchdog"])
    p.add_argument("--pod")
    p.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    p.add_argument("--served-name", default="qwen3-32b-awq")
    p.add_argument("--gpu")
    p.add_argument("--image", default="vllm/vllm-openai:latest")
    p.add_argument("--max-len", type=int, default=32768)
    p.add_argument("--no-think", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-hours", type=float, default=1.5)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--extra", nargs="*", default=[])
    a = p.parse_args()
    if a.cmd == "up":
        up(a)
    elif a.cmd == "wait":
        wait(a)
    elif a.cmd == "status":
        print(status(pod_id(a)), balance())
    elif a.cmd == "balance":
        print(balance())
    elif a.cmd == "down":
        down(pod_id(a))
    elif a.cmd == "watchdog":
        time.sleep(a.max_hours * 3600)
        try:
            down(a.pod)
        except Exception:
            pass


if __name__ == "__main__":
    main()
