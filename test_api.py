"""
Vizual axtarış API-sini lokal şəkillə sınayır.

  export SEARCH_KEY="vs_..."
  python test_api.py sekil.jpg          # ən oxşar 5 məhsul
  python test_api.py sekil.jpg 10       # ən oxşar 10 məhsul

Şəkil göndərilməzdən əvvəl kataloqla eyni qaydada hazırlanır:
ağ fon, uzun tərəf maksimum 336 px, JPEG keyfiyyəti 85.
"""
import base64, io, os, sys, time
import requests
from PIL import Image

API = os.environ.get("API_BASE", "https://visiual-search.lovable.app") + "/api/search"

def prepare(path, side=336):
    im = Image.open(path)
    im = im.convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3])
    bg.thumbnail((side, side))
    buf = io.BytesIO(); bg.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    key = os.environ.get("SEARCH_KEY") or sys.exit("SEARCH_KEY təyin edilməyib: export SEARCH_KEY='vs_...'")
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    image = prepare(sys.argv[1])
    t = time.time()
    r = requests.post(API, json={"image": image, "top_k": top_k},
                      headers={"x-api-key": key}, timeout=60)
    total = (time.time() - t) * 1000
    if r.status_code != 200:
        sys.exit(f"Xəta {r.status_code}: {r.text}")
    d = r.json()
    print(f"\nModel: {d.get('model')}   server: {d.get('query_time_ms')} ms   ümumi: {total:.0f} ms\n")
    print(f"{'#':<3}{'Oxşarlıq':<10}{'Məhsul':<38}{'Kateqoriya':<26}{'Qiymət':>10}")
    for i, p in enumerate(d["results"], 1):
        cat = f"{p['category']} / {p['subcategory']}"
        print(f"{i:<3}{p['similarity']:<10.3f}{p['title'][:36]:<38}{cat[:24]:<26}{p['price_usd']:>9.2f}$")
