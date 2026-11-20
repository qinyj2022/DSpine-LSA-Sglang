import os, sys, warnings

def main():
    import uvicorn
    uds = os.environ.get("SGLANG_UDS_PATH")
    if uds:
        _orig = uvicorn.Config.__init__
        def _patched(self, app, **kw):
            kw.pop("host", None)
            kw.pop("port", None)
            kw["uds"] = uds
            return _orig(self, app, **kw)
        uvicorn.Config.__init__ = _patched
        print(f"[uds] uvicorn will bind unix socket {uds}", flush=True)

    from sglang.srt.arg_groups.overrides import resolving_view  # noqa: F401
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    from sglang.launch_server import run_server
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

if __name__ == "__main__":
    main()