import os
F = "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/qwen3_5/backend.py"
s = open(F, encoding="utf-8").read()
old = '''        _fm = forward_batch.forward_mode
        if _fm.is_extend() or _fm.is_target_verify() or _fm.is_draft_extend_v2():
            return index.extend('''
new = '''        _fm = forward_batch.forward_mode
        if _fm.is_extend() or _fm.is_target_verify() or _fm.is_draft_extend_v2():
            # [PATCH spec] a verify batch reaches here without the metadata hook
            # having run, so put the request rows on it and advance the index now.
            if getattr(forward_batch, "lsa_rows", None) is None:
                from lsa.sglang.index import registry as _registry
                _registry.maintain(forward_batch)
            return index.extend('''
assert old in s, "patch4 anchor not found"
s = s.replace(old, new, 1)
open(F, "w", encoding="utf-8").write(s)
import py_compile; py_compile.compile(F, doraise=True)
print("patch4 OK + compile OK")
for ln, line in enumerate(s.splitlines(), 1):
    if "[PATCH spec]" in line:
        print(f"  {ln}: {line.strip()[:100]}")