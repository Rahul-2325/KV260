import json, collections
plan = json.load(open('/home/root/weights/layer_plan.json'))
print("total layers:", len(plan))
print("\nop_type counts:")
for k, v in collections.Counter(l.get('op_type','?') for l in plan).most_common():
    print("  %-22s %d" % (k, v))
print("\ndevice counts:")
for k, v in collections.Counter(str(l.get('device','?')) for l in plan).most_common():
    print("  %-10s %d" % (k, v))
print("\nfirst layer keys:", sorted(plan[0].keys()))
print("\nLAST layer:")
print(json.dumps(plan[-1], indent=1)[:700])
wm = json.load(open('/home/root/weights/weight_map.json'))
print("\nweight_map type:", type(wm).__name__, "entries:", len(wm))
k0 = list(wm)[0] if isinstance(wm, dict) else None
if k0: print("sample weight_map entry:", k0, "->", json.dumps(wm[k0])[:300])
