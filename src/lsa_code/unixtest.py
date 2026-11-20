import socket, threading, time, os
P="/tmp/lsa_sock_test.sock"
try: os.unlink(P)
except OSError: pass
def run(nbytes, label):
    srv=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(P); srv.listen(1)
    got={"n":0}
    def acc():
        c,_=srv.accept()
        while True:
            b=c.recv(262144)
            if not b: break
            got["n"]+=len(b)
        c.close()
    t=threading.Thread(target=acc); t.start(); time.sleep(0.15)
    sent=0
    try:
        s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(P)
        chunk=65536
        while sent<nbytes:
            n=min(chunk,nbytes-sent); s.sendall(b"y"*n); sent+=n
        s.close()
        print(f"  [AF_UNIX {label}] sent {sent}B OK, server got {got['n']}B", flush=True)
    except Exception as e:
        print(f"  [AF_UNIX {label}] FAILED after {sent}B: {type(e).__name__} {e}; server got {got['n']}B", flush=True)
    t.join(timeout=3); srv.close()
    try: os.unlink(P)
    except OSError: pass
for n,lab in ((200_000,"200KB"),(1_000_000,"1MB"),(4_000_000,"4MB")):
    run(n,lab)