import py_compile
B = "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/qwen3_5/backend.py"
s = open(B, encoding="utf-8").read()
if "[LSA-DBG]" in s:
    print("debug already present")
else:
    old = "        inputs = index_space.key_and_query(q, k, self.geometry, self.heads)\n"
    new = """        try:
            inputs = index_space.key_and_query(q, k, self.geometry, self.heads)
        except Exception:
            import os as _os
            if _os.environ.get("LSA_SPEC_DEBUG"):
                print(
                    "[LSA-DBG] mode=%s q=%s k=%s pos=%s seqlens=%s rows=%s "
                    "geom kv=%s q=%s dh=%s rot=%s pf=%s"
                    % (
                        forward_batch.forward_mode,
                        tuple(q.shape),
                        tuple(k.shape),
                        tuple(forward_batch.positions.shape),
                        forward_batch.seq_lens_cpu.tolist(),
                        getattr(forward_batch, "lsa_rows", None),
                        self.geometry.kv_head_count,
                        self.geometry.query_head_count,
                        self.geometry.head_dim,
                        self.geometry.rotary_dim,
                        self.geometry.head_dim - self.geometry.rotary_dim,
                    ),
                    flush=True,
                )
            raise
"""
    assert old in s, "anchor missing"
    open(B, "w", encoding="utf-8").write(s.replace(old, new, 1))
    py_compile.compile(B, doraise=True)
    print("debug patch OK")