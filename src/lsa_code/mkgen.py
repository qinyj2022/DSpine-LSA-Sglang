import json
p=json.load(open("/mnt/sda/qyj1044/lsa_code/n16k.json"))
p["max_tokens"]=128
json.dump(p, open("/mnt/sda/qyj1044/lsa_code/n16k_gen.json","w"))
print("ok")