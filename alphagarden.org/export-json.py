import csv
import json

def export_to_json(csv_name, out_name):
  with open(csv_name) as r:
    data = []
    next(r)
    for line in r:
      _, name, x, y, _, _, radius = line.split(',')[:7]
      radius = int(radius)
      x, y = int(x), int(y)
      data.append({"type": name, "center_x": x, "center_y": y, "radius": radius})
    with open(out_name, 'w') as w:
      json.dump(data, w)
    print(f"Export successful! Wrote {len(data)} plants to {out_name}")
  