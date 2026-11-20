import importlib, traceback
TARGETS = [
 "sglang.srt.configs.model_config.ModelConfig.__init__",
 "sglang.srt.layers.attention.dsa.dsa_indexer.Indexer",
 "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.init_forward_metadata",
 "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.init_forward_metadata_out_graph",
 "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_fa3",
 "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_flashmla_sparse",
 "sglang.srt.model_executor.forward_batch_info.build_inner_fb_view",
 "sglang.srt.model_executor.runner.decode_cuda_graph_runner.build_replay_fb_view",
 "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator._compute_cell_size",
 "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._profile_available_bytes",
 "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._resolve_memory_pool_config",
]
ELIDED = [
 "sglang.srt.mem_cache.kv_cache_configurator.dsa_layer_skips_topk",
 "sglang.srt.model_executor.pool_configurator.dsa_layer_skips_topk",
]
REQUIRED = [
 "sglang.srt.environ.envs","sglang.srt.environ.envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH",
 "sglang.srt.layers.dp_attention.is_dp_attention_enabled","sglang.srt.layers.dp_attention.attn_tp_all_reduce",
 "sglang.srt.configs.model_config.dsa_layer_skips_topk",
 "sglang.srt.layers.attention.dsa.utils.is_graph_dsa_split_op_surface",
 "sglang.srt.model_executor.forward_context.get_attn_backend",
 "sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
 "sglang.srt.runtime_context.get_schedule","sglang.srt.runtime_context.get_memory",
 "sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute.compute_cu_seqlens",
]
GQB = ("sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend",
       "sglang.srt.layers.attention.triton_backend.TritonAttnBackend")
GQM = ("__init__","init_forward_metadata","init_forward_metadata_out_graph","forward_extend","forward_decode")
GQT = tuple(f"{b}.{m}" for b in GQB for m in GQM)
EXTRA = ["sglang.srt.plugins.hook_registry.HookRegistry",
         "sglang.srt.plugins.hook_registry.HookType",
         "sglang.srt.plugins.hook_registry"]

def resolve(t):
    mp, _, attr = t.rpartition(".")
    try:
        holder = importlib.import_module(mp)
    except ModuleNotFoundError:
        outer, _, cls = mp.rpartition(".")
        holder = getattr(importlib.import_module(outer), cls)
    return getattr(holder, attr)

for name, group in (("TARGETS",TARGETS),("ELIDED",ELIDED),("REQUIRED",REQUIRED),
                    ("GROUPED_QUERY_TARGETS",GQT),("HOOK_REGISTRY",EXTRA)):
    bad = []
    for t in group:
        try: resolve(t)
        except Exception as e: bad.append((t, type(e).__name__, str(e)[:90]))
    print(f"--- {name}: {len(group)-len(bad)}/{len(group)} OK ---")
    for t,k,m in bad: print(f"   MISSING {t}  [{k}] {m}")