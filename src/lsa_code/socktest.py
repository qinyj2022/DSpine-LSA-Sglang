import socket, threading, sys, http.client
def server(port, ready):
    s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    s.bind(("127.0.0.1",port)); s.listen(4); ready.set()
    for _ in range(2):
        try:
            c,_=s.accept()
        except OSError:
            break
        data=b""
        try:
            while True:
                b=c.recv(65536)
                if not b: break
                data+=b
                if b"\r\n\r\n" in data:
                    hdr,_,rest=data.partition(b"\r\n\r\n")
                    cl=0
                    for line in hdr.split(b"\r\n"):
                        if line.lower().startswith(b"content-length:"):
                            cl=int(line.split(b":")[1])
                    if len(rest)>=cl: break
        except OSError as e:
            print("  SERVER recv error:", type(e).__name__, e, flush=True)
        print("  SERVER received", len(data), "bytes", flush=True)
        try: c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        except OSError: pass
        c.close()
    s.close()

PORT=39991
ready=threading.Event()
t=threading.Thread(target=server,args=(PORT,ready),daemon=True)
t.start(); ready.wait(5)
for size in (50_000, 200_000, 1_000_000):
    body=b"x"*size
    try:
        c=http.client.HTTPConnection("127.0.0.1",PORT,timeout=20)
        c.putrequest("POST","/"); c.putheader("Content-Length",str(len(body))); c.endheaders()
        for i in range(0,len(body),8192): c.send(body[i:i+8192])
        r=c.getresponse(); r.read()
        print(f"  CLIENT send {size} bytes -> OK ({r.status})", flush=True)
    except Exception as e:
        print(f"  CLIENT send {size} bytes -> FAILED {type(e).__name__} {e}", flush=True)