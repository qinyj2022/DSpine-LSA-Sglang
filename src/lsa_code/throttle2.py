import socket, time, threading
def run(nbytes, chunk, delay, label):
    srv=socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    srv.bind(("127.0.0.1",0)); srv.listen(1)
    port=srv.getsockname()[1]
    got={"n":0,"err":None}
    def acc():
        try:
            c,_=srv.accept()
            while True:
                b=c.recv(262144)
                if not b: break
                got["n"]+=len(b)
            c.close()
        except OSError as e:
            got["err"]=f"{type(e).__name__}:{e}"
    t=threading.Thread(target=acc); t.start(); time.sleep(0.15)
    sent=0
    try:
        s=socket.create_connection(("127.0.0.1",port),timeout=60)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        buf=bytearray(chunk)
        for _ in range(0, nbytes, chunk):
            n=min(chunk, nbytes-sent)
            s.sendall(bytes(buf[:n])); sent+=n
            if delay: time.sleep(delay)
        s.close()
        print(f"  [{label}] client sent {sent}B OK; server got {got['n']}B err={got['err']}", flush=True)
    except Exception as e:
        print(f"  [{label}] client FAILED after {sent}B: {type(e).__name__} {e}; server got {got['n']}B err={got['err']}", flush=True)
    t.join(timeout=3); srv.close()

print("=== 1MB, 不同分块/限速 ===")
run(1_000_000, 32768, 0,     "1MB 32K-chunk 无延迟")
run(1_000_000,  4096, 0,     "1MB 4K-chunk  无延迟")
run(1_000_000,  4096, 0.002, "1MB 4K-chunk  2ms延迟")
run(1_000_000,  1024, 0.001, "1MB 1K-chunk  1ms延迟")