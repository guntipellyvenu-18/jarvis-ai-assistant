import json

from jarvis_app import execute_tool, list_directory, read_file, system_diagnostics


def test_system_diagnostics():
    result = execute_tool("system_diagnostics", {"include_battery": False})
    assert result["status"] == "ok"
    assert "memory" in result
    assert "cpu" in result


def test_list_directory(tmp_path):
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("hello world\n", encoding="utf-8")
    result = execute_tool("list_directory", {"path": str(tmp_path), "max_items": 10})
    assert result["status"] == "ok"
    assert any(item["name"] == "alpha.txt" for item in result["entries"])


def test_read_file(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
    result = execute_tool("read_file", {"path": str(sample_file), "max_lines": 10})
    assert result["status"] == "ok"
    assert "line 1" in result["content"]


def test_list_directory_function():
    result = list_directory(".", max_items=5)
    assert result["status"] == "ok"
    assert "entries" in result


def test_read_file_function(tmp_path):
    f = tmp_path / "demo.txt"
    f.write_text("abc\n123\n", encoding="utf-8")
    result = read_file(str(f), max_lines=20)
    assert result["status"] == "ok"
    assert result["content"].startswith("abc")


def test_system_diagnostics_function():
    result = system_diagnostics(include_battery=True)
    assert result["status"] == "ok"
    assert "memory" in result
