import asyncio, os, sys, signal

LISTEN_HOST = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
LISTEN_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 30001
UDS = sys.argv[3] if len(sys.argv) > 3 else "/mnt/sda/qyj1044/run/sglang.sock"
BUF = 1 << 16

async def pump(r, w, tag):
    try:
        while True:
            data = await r.read(BUF)
            if not data:
                break
            w.write(data)
            await w.drain()
    except Exception:
        pass
    finally:
        try:
            if w.can_write_eof():
                w.write_eof()
        except Exception:
            pass

async def handle(cr, cw):
    try:
        ur, uw = await asyncio.open_unix_connection(UDS)
    except Exception as e:
        print(f"[bridge] uds connect failed: {e}", flush=True)
        try: cw.close()
        except Exception: pass
        return
    await asyncio.gather(pump(cr, uw, "c2s"), pump(ur, cw, "s2c"))
    for s in (cw, uw):
        try: s.close()
        except Exception: pass

async def main():
    srv = await asyncio.start_server(handle, LISTEN_HOST, LISTEN_PORT, limit=1 << 25)
    print(f"[bridge] tcp {LISTEN_HOST}:{LISTEN_PORT} -> uds {UDS}  READY", flush=True)
    async with srv:
        await srv.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())