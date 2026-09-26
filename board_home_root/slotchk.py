import json
plan=json.load(open('/home/root/weights/layer_plan.json'))
for lid in (23, 25, 138):
    L=[x for x in plan if x['id']==lid][0]
    print("=== L%d  %s ===" % (lid, L['op_type']))
    for k in ('input_0','input_1','input_2'):
        if k+'_name' in L:
            print("   %s  shape=%-20s fp=%-4s name=%s" % (k, L.get(k+'_shape'), L.get(k+'_fp'), L.get(k+'_name')))
    print("   out_shape=%s kernel=%s group=%s" % (L.get('out_shape'), L.get('kernel'), L.get('group')))
