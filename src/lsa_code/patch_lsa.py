import shutil, sys, os
F = "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/qwen3_5/backend.py"
BAK = F + ".orig"
if not os.path.exists(BAK):
    shutil.copy2(F, BAK)
    print("backup ->", BAK)
else:
    print("backup exists ->", BAK)
s = open(F, encoding="utf-8").read()

# --- 1) 解除启动期 guard ---
old1 = '''    if get_spec().speculative_algorithm is not None:
        raise NotImplementedError("Qwen under LSA serves without speculative decoding")
'''
new1 = '''    # [PATCH spec] speculative verify is served through the extend-shaped selection
'''
assert old1 in s, "patch1 pattern not found"
s = s.replace(old1, new1, 1)

# --- 2) select(): verify 路由到 extend ---
old2 = '''        if forward_batch.forward_mode.is_extend():
            return index.extend('''
new2 = '''        _fm = forward_batch.forward_mode
        if _fm.is_extend() or _fm.is_target_verify() or _fm.is_draft_extend_v2():
            return index.extend('''
assert old2 in s, "patch2 pattern not found"
s = s.replace(old2, new2, 1)

# --- 3) 解除 forward 期 guard ---
old3 = '''    if mode.is_target_verify() or mode.is_draft_extend_v2():
        raise NotImplementedError(
            "Qwen under LSA reads a plain prefill or decode, without speculative decoding"
        )
'''
new3 = '''    # [PATCH spec] verify/draft-extend read the extend-shaped selection
'''
assert old3 in s, "patch3 pattern not found"
s = s.replace(old3, new3, 1)

open(F, "w", encoding="utf-8").write(s)
print("patched OK")
print("--- 结果确认 ---")
for ln, line in enumerate(s.splitlines(), 1):
    if "[PATCH spec]" in line or "is_target_verify()" in line:
        print(f"  {ln}: {line.rstrip()[:110]}")
import py_compile
py_compile.compile(F, doraise=True)
print("compile OK")