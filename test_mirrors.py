import requests

q = '[out:json][timeout:60];node[amenity=restaurant](32.5,-117.3,33.0,-116.8);out center tags;'

mirrors = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass.osm.rambler.ru/cgi/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

print("Testing Overpass mirrors...\n")
for m in mirrors:
    try:
        r = requests.post(m, data={"data": q}, timeout=45)
        count = 0
        if r.status_code == 200:
            try:
                count = len(r.json().get("elements", []))
            except Exception:
                count = "?"
        print(f"{m}  →  Status: {r.status_code}  |  Results: {count}")
    except Exception as e:
        print(f"{m}  →  ERROR: {str(e)[:60]}")