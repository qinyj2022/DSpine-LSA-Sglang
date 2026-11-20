import re,io
p="/mnt/sda/qyj1044/lsa_code/acctest.py"
s=open(p,encoding="utf-8").read()
old = """            if isinstance(res,dict):
                txt=res["text"][0]; meta=res["meta_info"][0]
            else:
                txt=res[0]["text"]; meta=res[0].get("meta_info",{})"""
new = """            if isinstance(res,dict):
                t=res.get("text"); txt=t[0] if isinstance(t,list) else (t or "")
                m=res.get("meta_info"); meta=m[0] if isinstance(m,list) else (m or {})
            else:
                txt=res[0]["text"]; meta=res[0].get("meta_info",{})"""
assert old in s, "pattern not found"
open(p,"w",encoding="utf-8").write(s.replace(old,new))
print("patched")