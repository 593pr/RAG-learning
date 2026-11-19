import json
import subprocess
import sys
import threading

p = subprocess.Popen(
    [sys.executable,"mcp_server.py"],
    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,
    text=True,encoding="utf-8",bufsize=1,
)

#用同一个python解释器，拉起一个子进程，然后管理三个参数

def send(obj):
    p.stdin.write(json.dumps(obj,ensure_ascii=False) + "\n")
    p.stdin.flush()
    print(">>",json.dumps(obj,ensure_ascii=False))

def read_reply(label,timeout=5):
    box = {}
    t = threading.Thread(target = lambda:box.setdefault("line",p.stdout.readline()),daemon=True)
    t.start()
    t.join(timeout)
    if "line" not in box:
        p.kill()
        raise SystemExit(f"[{label}] {timeout}s 没有回包，往上翻 server 的 stderr 日志。")
    print("<<",box["line".strip()])
    
send({"jsonrpc":"2.0","id":"1","method":"initialize","params":{
    "protocolVersion":"2025-11-25",
    "capabilities":{},
    "clientInfo":{"name":"raw","version":"0.1"},
}})

read_reply("initialize")
#server 会回一个 initialize 响应，里面包含 server 的 serverInfo、capabilities、协商后的 protocolVersion。
#read_reply("initialize") 把这条响应读出来打印。没有这一步，后续任何 tools/* 调用都会被 server 拒绝。

send({"jsonrpc":"2.0","method":"notifications/initialized"})
#通知：无 id、无回包

send({"jsonrpc":"2.0","id":"2","method":"tools/list","params":{}})
read_reply("tools/list")

send({"jsonrpc":"2.0","id":"3","method":"tools/call",
      "params":{"name":"calculator","arguments":{"expression":"37*48"}}})
read_reply("tools/call")

p.kill()
