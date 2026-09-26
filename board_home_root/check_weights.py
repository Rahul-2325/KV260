import json, os, collections
wm = json.load(open('/home/root/weights/weight_map.json'))
tot = ok = null = 0
missing = []
notes = collections.Counter()
for lid, entries in wm.items():
    for e in entries:
        tot += 1
        f = e.get('file')
        if f:
            ok += 1
            p = os.path.join('/home/root/weights/data', os.path.basename(f))
            if not os.path.exists(p): missing.append(p)
        else:
            null += 1
            notes[str(e.get('note'))[:60]] += 1
print("total tensor entries : %d" % tot)
print("  with file          : %d" % ok)
print("  file=null          : %d" % null)
print("  referenced-but-missing on disk: %d" % len(missing))
print("\nnull-note reasons:")
for k,v in notes.most_common(5): print("  %-62s %d" % (k, v))
print("\nactual .npy files on disk: %d" % len(os.listdir('/home/root/weights/data')))
# which layers are fully satisfied?
full = sum(1 for lid, es in wm.items() if all(e.get('file') for e in es))
print("layers with ALL tensors present: %d / %d" % (full, len(wm)))
# sample a real file entry
for lid, es in wm.items():
    for e in es:
        if e.get('file'):
            print("\nsample OK entry: layer", lid, json.dumps(e)[:200]); break
    else: continue
    break
