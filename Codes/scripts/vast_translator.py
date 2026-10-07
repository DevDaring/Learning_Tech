#!/usr/bin/env python3
"""Rent one Vast.ai GPU for IndicTrans2, connect it through an SSH tunnel, destroy it after.

    python scripts/vast_translator.py up        # rent, verify, install, start service, open tunnel
    python scripts/vast_translator.py status
    python scripts/vast_translator.py down      # destroy the instance and close the tunnel

State (contract id, tunnel pid, access token) is written to runs/vast_translator.json
the moment the contract exists, so the instance can always be destroyed. The pipeline
uses the service when TLT_TRANSLATE_URL and TLT_TRANSLATE_TOKEN are set (scripts print
the export line), otherwise it translates on the local CPU.
Credentials are read from Codes/.env by name and never printed. The Hugging Face token is
sent to the box over SSH only to download the gated models, then deleted there.
"""
from __future__ import annotations

import json
import secrets as pysecrets
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tlt import secrets  # noqa: E402
from tlt.config import load_yaml  # noqa: E402

STATE = ROOT / "runs" / "vast_translator.json"
API = "https://console.vast.ai/api/v0"
KEYFILE = str(Path.home() / ".ssh" / "id_ed25519")
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"  # small image; all deps are pre-built wheels
GPU_OK = ("RTX 3090", "RTX 4090", "RTX A5000", "RTX A6000", "A10", "L4", "A40", "RTX 4080", "RTX 6000Ada", "L40")
LOCAL_PORT = 18000
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30"]


def key() -> str:
    k = secrets.get("Vast_AI_PHD_API_KEY")
    if not k:
        raise SystemExit("Vast_AI_PHD_API_KEY missing in .env")
    return k


def api(path: str, data=None, method="GET", base=API):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"},
                                 method=method)
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))
    STATE.chmod(0o600)


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def ssh(st: dict, cmd: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-i", KEYFILE, "-p", str(st["port"]), f"root@{st['ip']}", *SSH_OPTS, cmd],
                          capture_output=True, text=True, timeout=timeout)


def offers(max_dph: float = 0.6) -> list[dict]:
    q = {"verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False}, "num_gpus": {"eq": 1},
         "gpu_ram": {"gte": 16000}, "reliability2": {"gte": 0.98}, "inet_down": {"gte": 1000},
         "disk_space": {"gte": 50}, "direct_port_count": {"gte": 1}, "dph_total": {"lte": max_dph},
         "cuda_max_good": {"gte": 12.4}, "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 200}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{API}/bundles/?q=" + urllib.parse.quote(json.dumps(q)),
        headers={"Authorization": f"Bearer {key()}"}), timeout=90).read())["offers"]
    badf = ROOT / "runs" / "vast_bad_machines.json"
    bad = set(json.loads(badf.read_text())) if badf.exists() else set()   # hosts that failed before
    return [o for o in r if any(o["gpu_name"].startswith(g) for g in GPU_OK) and o.get("machine_id") not in bad]


def up() -> None:
    st = load()
    if st.get("contract"):
        raise SystemExit(f"an instance is already recorded ({st['contract']}); run `down` first")
    for o in offers()[:6]:
        print(f"trying offer {o['id']}: {o['gpu_name']} ${o['dph_total']:.3f}/h", flush=True)
        r = api(f"/asks/{o['id']}/", {"client_id": "me", "image": IMAGE, "disk": 50, "label": "tlt-indictrans",
                                       "runtype": "ssh_direc ssh_proxy", "onstart": "touch /root/.ready"}, "PUT")
        cid = r.get("new_contract")
        if not cid:
            continue
        st = {"contract": cid, "gpu": o["gpu_name"], "dph": o["dph_total"], "rented_at": time.time(),
              "token": pysecrets.token_hex(24)}
        save(st)  # billing has started: record the contract before anything else
        print(json.dumps({"event": "rented", "contract": cid, "gpu": o["gpu_name"], "dph": o["dph_total"]}), flush=True)
        if wait_ready(st):
            break
        print(f"destroying {cid} and trying the next offer", flush=True)
        down(quiet=True)
        st = {}
    if not st.get("ip"):
        raise SystemExit("no offer passed verification")
    safe_setup(st)


def safe_setup(st: dict) -> None:
    """Run setup; on ANY failure destroy the instance so nothing is left billing."""
    try:
        setup(st)
    except BaseException as e:
        print(f"setup failed ({type(e).__name__}: {e}); destroying the instance", flush=True)
        down()
        raise


def wait_ready(st: dict) -> bool:
    """Wait for the instance to run, then retry SSH until CUDA answers (sshd starts late)."""
    cid = st["contract"]
    for _ in range(60):
        time.sleep(15)
        try:
            m = api(f"/instances/{cid}/")["instances"]
        except Exception:
            continue
        p22 = ((m.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort")
        if m.get("actual_status") == "running" and p22:
            st.update(ip=m["public_ipaddr"], port=int(p22))
            save(st)
            break
    if "ip" not in st:
        return False
    for attempt in range(12):
        probe = ssh(st, "python -c \"import torch;print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))\"", 180)
        if "CUDA True" in probe.stdout:
            print(probe.stdout.strip(), flush=True)
            return True
        time.sleep(20)
    print("CUDA check failed:", (probe.stdout + probe.stderr)[-300:], flush=True)
    return False


def resume() -> None:
    st = load()
    if not st.get("contract"):
        raise SystemExit("no instance recorded")
    if not st.get("ip") and not wait_ready(st):
        down()
        raise SystemExit("instance never became reachable; destroyed")
    safe_setup(st)


def setup(st: dict) -> None:
    tcfg = load_yaml(ROOT / "config" / "models.yaml")["translation"]
    models = [tcfg["en_indic"], tcfg["indic_en"]]
    revs = tcfg.get("revisions", {})
    # install and start the service
    subprocess.run(["scp", "-i", KEYFILE, "-P", str(st["port"]), *SSH_OPTS, str(ROOT / "scripts" / "indictrans_server.py"),
                    f"root@{st['ip']}:/root/indictrans_server.py"], check=True, capture_output=True, timeout=120)
    for attempt in range(3):
        inst = ssh(st, "pip install -q --retries 10 --timeout 60 'transformers==4.46.0' sentencepiece "
                       "'indictranstoolkit==1.1.1' >/dev/null 2>&1; python -c 'import transformers, IndicTransToolkit; "
                       "print(\"PKG_OK\")'", 1800)
        if "PKG_OK" in inst.stdout:
            break
    print("install:", "ok" if "PKG_OK" in inst.stdout else "FAILED " + inst.stderr[-200:], flush=True)
    if "PKG_OK" not in inst.stdout:
        raise SystemExit("package install failed (bad host network?)")
    # Pre-built FlashAttention-2 wheel matching the box's python / torch / CUDA / C++ ABI
    # (no compilation). Falls back to PyTorch SDPA attention if no wheel fits.
    fa = ssh(st, "python - <<'PY'\n"
                 "import sys, torch, subprocess, importlib.util\n"
                 "py=f'cp{sys.version_info.major}{sys.version_info.minor}'\n"
                 "tv='.'.join(torch.__version__.split('+')[0].split('.')[:2])\n"
                 "abi='TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'\n"
                 "ver='2.7.4.post1'\n"
                 "url=f'https://github.com/Dao-AILab/flash-attention/releases/download/v{ver}/flash_attn-{ver}+cu12torch{tv}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl'\n"
                 "print('WHEEL', url)\n"
                 "r=subprocess.run([sys.executable,'-m','pip','install','-q',url],capture_output=True,text=True)\n"
                 "print('PIP', r.returncode, r.stderr[-200:])\n"
                 "print('FA_OK', importlib.util.find_spec('flash_attn') is not None)\n"
                 "PY", 900)
    print("flash-attn:", fa.stdout[-400:], flush=True)
    attn = "flash_attention_2" if "FA_OK True" in fa.stdout else "sdpa"
    st["attn"] = attn
    save(st)
    # download the gated models with a script file (no nested shell quoting); the token is
    # only in this command's environment and the cached token file is removed afterwards
    script = "from huggingface_hub import snapshot_download as s\n" + "".join(
        f"print(s('{m}', revision='{revs.get(m) or 'main'}'))\n" for m in models)
    local = STATE.parent / "dl_models.py"
    local.write_text(script)
    subprocess.run(["scp", "-i", KEYFILE, "-P", str(st["port"]), *SSH_OPTS, str(local), f"root@{st['ip']}:/root/dl.py"],
                   check=True, capture_output=True, timeout=120)
    hf = secrets.get("HUGGINGFACE_TOKEN")
    r = ssh(st, f"cd /root && HF_TOKEN={hf} python dl.py 2>&1 | tail -2; rm -f /root/.cache/huggingface/token", 3600)
    print("download:", (r.stdout + r.stderr)[-200:].replace(hf, "***"), flush=True)
    if "snapshots" not in r.stdout:
        raise SystemExit("model download failed")
    serve(st, attn)


def serve(st: dict | None = None, attn: str | None = None) -> None:
    """Start the translation service and the SSH tunnel on an instance that has the models."""
    st = st or load()
    attn = attn or st.get("attn", "flash_attention_2")
    tcfg = load_yaml(ROOT / "config" / "models.yaml")["translation"]
    models = [tcfg["en_indic"], tcfg["indic_en"]]
    revs = tcfg.get("revisions", {})
    preload = " ".join(f"{m}@{revs.get(m, '')}" for m in models)
    try:  # ssh may keep the channel open on the detached job; the health check below decides
        ssh(st, f"cd /root && TLT_TOKEN={st['token']} TLT_DTYPE=float16 TLT_ATTN={attn} HF_HUB_OFFLINE=1 "
                f"setsid nohup python indictrans_server.py "
                f"--port 8000 --preload {preload} > /root/server.log 2>&1 < /dev/null &", 30)
    except subprocess.TimeoutExpired:
        pass
    tun = subprocess.Popen(["ssh", "-i", KEYFILE, "-p", str(st["port"]), *SSH_OPTS, "-N", "-L",
                            f"{LOCAL_PORT}:127.0.0.1:8000", f"root@{st['ip']}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    st["tunnel_pid"] = tun.pid
    save(st)
    for _ in range(80):
        time.sleep(15)
        try:
            h = json.loads(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{LOCAL_PORT}/health", headers={"X-Token": st["token"]}), timeout=10).read())
            if len(h.get("models", [])) == len(models):
                print(json.dumps({"event": "ready", "device": h["device"], "models": h["models"]}), flush=True)
                print(f"export TLT_TRANSLATE_URL=http://127.0.0.1:{LOCAL_PORT}  # token is read from {STATE.name}")
                return
        except Exception:
            pass
    print("service did not become ready; server log tail:", ssh(st, "tail -20 /root/server.log", 60).stdout)
    raise SystemExit("service did not become ready")


def status() -> None:
    st = load()
    if not st.get("contract"):
        print("no instance recorded")
        return
    try:
        m = api(f"/instances/{st['contract']}/")["instances"]
        hours = (time.time() - st["rented_at"]) / 3600
        print(json.dumps({"contract": st["contract"], "gpu": st.get("gpu"), "status": m.get("actual_status"),
                          "hours": round(hours, 2), "approx_cost_usd": round(hours * st.get("dph", 0), 2)}))
    except Exception as e:
        print("status error:", type(e).__name__)


def down(quiet: bool = False) -> None:
    st = load()
    if st.get("tunnel_pid"):
        subprocess.run(["kill", str(st["tunnel_pid"])], capture_output=True)
    if st.get("contract"):
        try:
            api(f"/instances/{st['contract']}/", {}, "DELETE")
            if not quiet:
                hours = (time.time() - st["rented_at"]) / 3600
                print(f"destroyed {st['contract']} after {hours:.2f} h (about ${hours * st.get('dph', 0):.2f})")
        except Exception as e:
            print("DESTROY FAILED - destroy it in the Vast console:", st["contract"], type(e).__name__)
            return
    STATE.unlink(missing_ok=True)


if __name__ == "__main__":
    {"up": up, "resume": resume, "serve": serve, "status": status, "down": down}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
