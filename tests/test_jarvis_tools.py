import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from ollama import chat

logging.basicConfig(
    level=getattr(logging, os.getenv("JARVIS_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("jarvis")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
MAX_TOOL_ITERATIONS = 5
MAX_FILE_BYTES = 200_000
MAX_READ_LINES = 200

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "system_diagnostics",
            "description": "Return CPU, memory, disk, load, and uptime information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_battery": {
                        "type": "boolean",
                        "description": "Include battery stats when available.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_processes",
            "description": "List the highest CPU or memory usage processes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of processes to return.", "default": 10},
                    "sort_by": {"type": "string", "enum": ["cpu", "memory"], "default": "cpu"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to inspect."},
                    "max_items": {"type": "integer", "description": "Maximum number of items to return.", "default": 50},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file with a safe line limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read."},
                    "max_lines": {"type": "integer", "description": "Maximum number of lines to return.", "default": 200},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "network_summary",
            "description": "Return network interface and socket connection summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_connections": {"type": "boolean", "description": "Include active socket connections."},
                },
                "required": [],
            },
        },
    },
]


def _normalize_tool_args(raw_args: Any) -> Dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {"value": raw_args}
    return dict(raw_args)


def _safe_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path_str}")
    return path


def system_diagnostics(include_battery: bool = False) -> Dict[str, Any]:
    try:
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        load = os.getloadavg() if hasattr(os, "getloadavg") else None
        boot_time = psutil.boot_time()
        uptime_seconds = max(0, int(__import__("time").time() - boot_time))

        result = {
            "status": "ok",
            "cpu": {
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "cpu_percent": psutil.cpu_percent(interval=None),
            },
            "memory": {
                "total_mb": vm.total // (1024 * 1024),
                "available_mb": vm.available // (1024 * 1024),
                "used_mb": vm.used // (1024 * 1024),
                "percent_used": vm.percent,
            },
            "disk": {
                "total_gb": disk.total // (1024 * 1024 * 1024),
                "used_gb": disk.used // (1024 * 1024 * 1024),
                "free_gb": disk.free // (1024 * 1024 * 1024),
                "percent_used": disk.percent,
            },
            "system": {
                "uptime_seconds": uptime_seconds,
                "load_average": load,
                "platform": sys.platform,
            },
        }

        if include_battery:
            battery = psutil.sensors_battery()
            if battery is None:
                result["battery"] = {"status": "unavailable"}
            else:
                result["battery"] = {
                    "percent": battery.percent,
                    "plugged_in": battery.power_plugged,
                    "secsleft": battery.secsleft,
                }

        return result
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("system diagnostics failed")
        return {"status": "error", "message": str(exc)}


def list_processes(limit: int = 10, sort_by: str = "cpu") -> Dict[str, Any]:
    try:
        limit = max(1, int(limit))
        sort_by = sort_by.lower()
        if sort_by not in {"cpu", "memory"}:
            sort_by = "cpu"

        proc_rows = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                proc_rows.append({
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "cpu_percent": round(info.get("cpu_percent") or 0.0, 2),
                    "memory_percent": round(info.get("memory_percent") or 0.0, 2),
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        proc_rows.sort(key=lambda x: x[f"{sort_by}_percent"], reverse=True)
        return {"status": "ok", "processes": proc_rows[:limit]}
    except Exception as exc:
        logger.exception("process listing failed")
        return {"status": "error", "message": str(exc)}


def list_directory(path: str, max_items: int = 50) -> Dict[str, Any]:
    try:
        target = _safe_path(path)
        entries = []
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            try:
                stat = child.stat()
                entries.append({
                    "name": child.name,
                    "type": "directory" if child.is_dir() else "file",
                    "size": stat.st_size,
                    "modified": __import__("datetime").datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            except OSError:
                entries.append({"name": child.name, "type": "unknown"})
            if len(entries) >= max(1, int(max_items)):
                break

        return {"status": "ok", "path": str(target), "entries": entries}
    except Exception as exc:
        logger.exception("directory listing failed")
        return {"status": "error", "message": str(exc)}


def read_file(path: str, max_lines: int = MAX_READ_LINES) -> Dict[str, Any]:
    try:
        target = _safe_path(path)
        if not target.is_file():
            raise ValueError(f"Not a file: {path}")

        with target.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()

        if len(lines) > int(max_lines):
            preview = lines[:int(max_lines)]
            truncated = True
        else:
            preview = lines
            truncated = False

        content = "".join(preview)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            content = content[:MAX_FILE_BYTES]
            truncated = True

        return {
            "status": "ok",
            "path": str(target),
            "truncated": truncated,
            "total_lines": len(lines),
            "content": content,
        }
    except Exception as exc:
        logger.exception("file read failed")
        return {"status": "error", "message": str(exc)}


def network_summary(include_connections: bool = False) -> Dict[str, Any]:
    try:
        interfaces = []
        for name, addrs in psutil.net_if_addrs().items():
            addresses = []
            for addr in addrs:
                addresses.append({
                    "family": str(addr.family),
                    "address": addr.address,
                    "netmask": addr.netmask,
                })
            interfaces.append({"name": name, "addresses": addresses})

        net_io = psutil.net_io_counters()
        result = {
            "status": "ok",
            "interfaces": interfaces,
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        }

        if include_connections:
            connections = []
            for conn in psutil.net_connections(kind="inet"):
                connections.append({
                    "fd": conn.fd,
                    "family": str(conn.family),
                    "type": str(conn.type),
                    "laddr": (conn.laddr.ip, conn.laddr.port) if conn.laddr else None,
                    "raddr": (conn.raddr.ip, conn.raddr.port) if conn.raddr else None,
                    "status": conn.status,
                })
            result["connections"] = connections

        return result
    except Exception as exc:
        logger.exception("network summary failed")
        return {"status": "error", "message": str(exc)}


def execute_tool(name: str, raw_args: Any) -> Dict[str, Any]:
    args = _normalize_tool_args(raw_args)
    logger.info("Executing tool %s with args=%s", name, args)

    try:
        if name == "system_diagnostics":
            return system_diagnostics(include_battery=args.get("include_battery", False))
        if name == "list_processes":
            return list_processes(limit=args.get("limit", 10), sort_by=args.get("sort_by", "cpu"))
        if name == "list_directory":
            return list_directory(path=args["path"], max_items=args.get("max_items", 50))
        if name == "read_file":
            return read_file(path=args["path"], max_lines=args.get("max_lines", MAX_READ_LINES))
        if name == "network_summary":
            return network_summary(include_connections=args.get("include_connections", False))
        return {"status": "error", "message": f"Tool not recognized: {name}"}
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected tool execution failure for %s", name)
        return {"status": "error", "message": str(exc)}


def run_jarvis(prompt_text: str, max_iterations: int = MAX_TOOL_ITERATIONS) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are JARVIS. Be precise, concise, and tool-aware. Use tools when useful."},
        {"role": "user", "content": prompt_text},
    ]

    for _ in range(max_iterations):
        try:
            response = chat(model=OLLAMA_MODEL, messages=messages, tools=TOOLS)
        except Exception as exc:
            logger.exception("Ollama chat request failed")
            return f"Error contacting Ollama: {exc}"

        tool_calls = getattr(response.message, "tool_calls", None) or []
        if not tool_calls:
            return response.message.content or "No response from model."

        messages.append({
            "role": "assistant",
            "content": response.message.content or "",
            "tool_calls": [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            result = execute_tool(tc.function.name, tc.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": getattr(tc, "id", None),
                "content": json.dumps(result),
            })

    return "Max tool iterations reached without final model response."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python jarvis_app.py \"Your question here\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    print(run_jarvis(prompt))

