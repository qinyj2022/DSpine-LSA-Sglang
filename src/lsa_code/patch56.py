import py_compile, shutil, os
B = "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/qwen3_5/backend.py"
R = "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/index/registry.py"
if not os.path.exists(R + ".orig"): shutil.copy2(R, R + ".orig")

s = open(R, encoding="utf-8").read()
if "[PATCH spec] a verify batch is extend-shaped" in s:
    print("patch5 already applied")
else:
    old = """    seq_lens = forward_batch.seq_lens_cpu.tolist()[:request_count]
    if forward_batch.forward_mode.is_extend():
        lengths = list(forward_batch.extend_seq_lens_cpu)[:request_count]
    else:
        lengths = [1] * request_count
"""
    new = """    seq_lens = forward_batch.seq_lens_cpu.tolist()[:request_count]
    _fm = forward_batch.forward_mode
    if _fm.is_target_verify() or _fm.is_draft_extend_v2():
        # [PATCH spec] a verify batch is extend-shaped but carries no extend lengths
        _k = int(forward_batch.positions.numel()) // max(1, request_count)
        lengths = [_k] * request_count
    elif _fm.is_extend():
        lengths = list(forward_batch.extend_seq_lens_cpu)[:request_count]
    else:
        lengths = [1] * request_count
"""
    assert old in s, "patch5 anchor missing"
    open(R, "w", encoding="utf-8").write(s.replace(old, new, 1))
    py_compile.compile(R, doraise=True)
    print("patch5 OK")

s = open(B, encoding="utf-8").read()
if "[PATCH spec] synthesize the extend lengths" in s:
    print("patch6 already applied")
else:
    old = """            return index.extend(
                inputs,
                positions,
                slots,
                forward_batch.lsa_rows,
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_prefix_lens_cpu,
            )
"""
    new = """            if _fm.is_target_verify() or _fm.is_draft_extend_v2():
                # [PATCH spec] synthesize the extend lengths a verify batch omits
                _bs = len(forward_batch.lsa_rows)
                _k = int(forward_batch.positions.numel()) // max(1, _bs)
                _lens = [_k] * _bs
                _prefix = [int(s) - _k for s in forward_batch.seq_lens_cpu.tolist()[:_bs]]
            else:
                _lens = forward_batch.extend_seq_lens_cpu
                _prefix = forward_batch.extend_prefix_lens_cpu
            return index.extend(
                inputs,
                positions,
                slots,
                forward_batch.lsa_rows,
                _lens,
                _prefix,
            )
"""
    assert old in s, "patch6 anchor missing"
    open(B, "w", encoding="utf-8").write(s.replace(old, new, 1))
    py_compile.compile(B, doraise=True)
    print("patch6 OK")