import socket, time, json, sys
BODY = open("/mnt/sda/qyj1044/lsa_code/longtest.py","rb")  # placeholder
body = b"x"*200000
def try_send(delay, chunk, host="127.0.0.1", port=39992, label=""):
    srv=socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    srv.bind((host,port)); srv.listen(1)
    got={"n":0}
    import threading
    def acc():
        c,_=srv.accept(); data=0
        try:
            while True:
                b=c.recv(65536)
                if not b: break
                data+=len(b)
                if data>=200: break
        except OSError: pass
        got["n"]=data; c.close()
    t=threading.Thread(target=acc); t.start()
    time.sleep(0.2)
    try:
        c=socket.create_connection((host,port),timeout=30)
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sent=0
        for i in range(0,len(body),chunk):
            c.sendall(body[i:i+chunk]); sent+=len(body[i:i+chunk])
            if delay: time.sleep(delay)
        print(f"  [{label}] sent {sent} bytes OK (delay={delay}, chunk={chunk})", flush=True)
        c.close()
    except Exception as e:
        print(f"  [{label}] FAILED at {sent if 'sent' in dir() else 0}B: {type(e).__name__} {e}", flush=True)
    t.join(timeout=5); srv.close()
print("=== 200KB, 各种发送策略 ===")
try_send(0.000,  65536, label="no-delay 64K")
try_send(0.005,   4096, label="5ms/4K")
try_send(0.020,   4096, label="20ms/4K")
try_send(0.050,   8192, label="50ms/8K")