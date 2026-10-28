"""
Builds the search catalog from the task ZIP (products.csv + images/).

Folder layout (unzipped task data):
    products.csv
    images/001_....jpg ...
    build_catalog.py   <- this file

Setup:
    pip install requests pillow numpy
    export NVIDIA_API_KEY="nvapi-..."

Commands:
    python build_catalog.py check
        -> tests the embedding model on one image
    python build_catalog.py embed https://raw.githubusercontent.com/<user>/<repo>/main/
        -> catalog_embeddings.json  (one 1024-dim vector per product)
        -> catalog.json             (products + public image links)
       The URL is the raw address of the GitHub repo where you uploaded images/.
    python build_catalog.py evaluate
        -> leave-one-out quality check (same category / subcategory)

If embed stops halfway, run it again: finished images are skipped.
"""
import base64, csv, io, json, os, sys, time
import requests
from PIL import Image

API = "https://integrate.api.nvidia.com/v1/embeddings"
MODEL = os.environ.get("EMB_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")
MAX_RPM = 35
IMG_SIDE = 336
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "products.csv")
OUT_EMB = os.path.join(HERE, "catalog_embeddings.json")
OUT_CAT = os.path.join(HERE, "catalog.json")
CACHE = os.path.join(HERE, ".cache_embeddings.json")

_last = [0.0]
def throttle():
    wait = 60.0 / MAX_RPM - (time.time() - _last[0])
    if wait > 0: time.sleep(wait)
    _last[0] = time.time()

def key():
    k = os.environ.get("NVIDIA_API_KEY")
    if not k: sys.exit("NVIDIA_API_KEY is not set: export NVIDIA_API_KEY='nvapi-...'")
    return k

def data_url(path):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3])
    bg.thumbnail((IMG_SIDE, IMG_SIDE))
    buf = io.BytesIO(); bg.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def embed(path, tries=3):
    body = {"model": MODEL, "input": [data_url(path)], "input_type": "passage",
            "modality": "image", "dimensions": 1024, "encoding_format": "float"}
    for i in range(tries):
        throttle()
        r = requests.post(API, json=body, timeout=60,
                          headers={"Authorization": f"Bearer {key()}", "Accept": "application/json"})
        if r.status_code == 429 or r.status_code >= 500:
            print(f"   ... {r.status_code}, waiting"); time.sleep(5 * (i + 1)); continue
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        return [round(x, 5) for x in r.json()["data"][0]["embedding"]]
    raise RuntimeError("too many retries")

def products():
    return list(csv.DictReader(open(CSV, encoding="utf-8")))

def cmd_check():
    p = products()[0]
    t = time.time(); v = embed(os.path.join(HERE, p["image_file"]))
    print(f"OK {MODEL}: {len(v)} dimensions, {time.time() - t:.1f}s for '{p['name']}'")

def cmd_embed(base_url):
    if not base_url.startswith("http"):
        sys.exit("Give the raw GitHub base URL, e.g.\n  python build_catalog.py embed https://raw.githubusercontent.com/<user>/<repo>/main/")
    base_url = base_url.rstrip("/") + "/"
    rows = products()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for i, p in enumerate(rows, 1):
        if p["id"] in cache: continue
        try:
            cache[p["id"]] = embed(os.path.join(HERE, p["image_file"]))
            json.dump(cache, open(CACHE, "w"))
            print(f"[{i}/{len(rows)}] {p['name']}")
        except Exception as e:
            print(f"[{i}/{len(rows)}] SKIPPED {p['name']}: {e}")
    missing = [p["id"] for p in rows if p["id"] not in cache]
    if missing:
        sys.exit(f"\n{len(missing)} product(s) missing: {missing}. Run the same command again.")
    catalog = [{"id": int(p["id"]), "title": p["name"], "brand": p["brand"], "category": p["category"],
                "subcategory": p["subcategory"], "price_usd": float(p["price_usd"]),
                "image": base_url + p["image_file"], "description": ""} for p in rows]
    json.dump(catalog, open(OUT_CAT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"model": MODEL, "dim": 1024, "image_side": IMG_SIDE,
               "items": [{"id": int(p["id"]), "embedding": cache[p["id"]]} for p in rows]},
              open(OUT_EMB, "w"))
    print(f"\nDone: catalog.json + catalog_embeddings.json ({len(rows)} products)")
    print(f"Check one image link in a browser: {catalog[0]['image']}")

def cmd_evaluate():
    import numpy as np
    cat = {p["id"]: p for p in json.load(open(OUT_CAT, encoding="utf-8"))}
    emb = json.load(open(OUT_EMB))["items"]
    ids = [e["id"] for e in emb]
    X = np.array([e["embedding"] for e in emb]); X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T; np.fill_diagonal(S, -1)
    nn = S.argmax(1)
    for level in ("category", "subcategory"):
        acc = np.mean([cat[ids[i]][level] == cat[ids[j]][level] for i, j in enumerate(nn)])
        print(f"Nearest neighbor in the same {level}: {acc:.1%}")
    top5 = np.argsort(-S, axis=1)[:, :5]
    hit = np.mean([any(cat[ids[i]]["subcategory"] == cat[ids[j]]["subcategory"] for j in row) for i, row in enumerate(top5)])
    print(f"Same subcategory within the top 5: {hit:.1%}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check": cmd_check()
    elif cmd == "embed": cmd_embed(sys.argv[2] if len(sys.argv) > 2 else "")
    elif cmd == "evaluate": cmd_evaluate()
    else: print(__doc__)
