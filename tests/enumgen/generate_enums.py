#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys


HEADER_RE = re.compile(r"^(enum|flags)\s+([A-Za-z_]\w*)(?:\s*:\s*(.+))?$")


@dataclass(frozen=True)
class EntrySpec:
    kind: str
    name: str
    expr: str | None
    doc_raw: str


@dataclass
class EnumSpec:
    kind: str
    name: str
    underlying: str | None
    docs_raw: list[str] = field(default_factory=list)
    entries: list[EntrySpec] = field(default_factory=list)


def cpp_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n")
    return f'"{escaped}"'


def strip_comment_markers(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if text.startswith("/*!<"):
        text = text.removeprefix("/*!<").strip()
    elif text.startswith("/*!"):
        text = text.removeprefix("/*!").strip()
    elif text.startswith("/**"):
        text = text.removeprefix("/**").strip()
    elif text.startswith("///"):
        text = text.removeprefix("///").strip()
    if text.endswith("*/"):
        text = text[:-2].strip()
    text = text.lstrip("*").strip()
    text = re.sub(r"^(?:\\brief|@brief)\s+", "", text)
    return text


def collapse_doc_lines(lines: list[str]) -> str:
    parts = [strip_comment_markers(line) for line in lines]
    parts = [part for part in parts if part]
    return " ".join(parts)


def parse_entry(path: Path, lineno: int, line: str) -> EntrySpec:
    kind, payload = line.split(None, 1)
    doc_start = payload.find("/*!")
    doc_raw = ""
    if doc_start != -1:
        doc_raw = payload[doc_start:].strip()
        payload = payload[:doc_start].rstrip()

    if "=" in payload:
        name, expr = payload.split("=", 1)
        name = name.strip()
        expr = expr.strip()
    else:
        name = payload.strip()
        expr = None

    if not name:
        raise RuntimeError(f"{path}:{lineno}: missing entry name")
    if kind == "alias" and expr is None:
        raise RuntimeError(f"{path}:{lineno}: alias entries require an expression")
    return EntrySpec(kind=kind, name=name, expr=expr, doc_raw=doc_raw)


def parse_def(path: Path) -> list[EnumSpec]:
    specs: list[EnumSpec] = []
    pending_docs: list[str] = []
    current: EnumSpec | None = None

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("doc "):
            pending_docs.append(line[4:])
            continue
        if line == "end":
            if current is None:
                raise RuntimeError(f"{path}:{lineno}: stray 'end'")
            specs.append(current)
            current = None
            continue
        if line.startswith("enum ") or line.startswith("flags "):
            if current is not None:
                raise RuntimeError(f"{path}:{lineno}: nested enum block")
            match = HEADER_RE.match(line)
            if not match:
                raise RuntimeError(f"{path}:{lineno}: invalid enum header")
            current = EnumSpec(
                kind=match.group(1),
                name=match.group(2),
                underlying=match.group(3).strip() if match.group(3) else None,
                docs_raw=list(pending_docs),
            )
            pending_docs.clear()
            continue
        if line.startswith("item ") or line.startswith("alias "):
            if current is None:
                raise RuntimeError(f"{path}:{lineno}: entry outside enum block")
            current.entries.append(parse_entry(path, lineno, line))
            continue
        raise RuntimeError(f"{path}:{lineno}: unsupported directive: {line}")

    if current is not None:
        raise RuntimeError(f"{path}: missing 'end' for enum {current.name}")
    if pending_docs:
        raise RuntimeError(f"{path}: dangling doc lines at end of file")
    return specs


def emit_enum(spec: EnumSpec) -> list[str]:
    lines: list[str] = []
    lines.extend(spec.docs_raw)
    header = f"enum class {spec.name}"
    if spec.underlying:
        header += f" : {spec.underlying}"
    header += " {"
    lines.append(header)
    for entry in spec.entries:
        line = f"    {entry.name}"
        if entry.expr is not None:
            line += f" = {entry.expr}"
        line += ","
        if entry.doc_raw:
            line += f" {entry.doc_raw}"
        lines.append(line)
    lines.append("};")
    return lines


def emit_traits(spec: EnumSpec) -> list[str]:
    fq_name = f"test::enumgen_demo::{spec.name}"
    lines = [
        "template<>",
        f"struct enum_traits<{fq_name}> {{",
        f"    static constexpr bool             is_bitflag = {'true' if spec.kind == 'flags' else 'false'};",
        f"    static constexpr std::string_view doc        = {cpp_string(collapse_doc_lines(spec.docs_raw))};",
        "    static constexpr std::array       entries    = {",
    ]
    for entry in spec.entries:
        doc_text = collapse_doc_lines([entry.doc_raw]) if entry.doc_raw else ""
        canonical = "false" if entry.kind == "alias" else "true"
        lines.append(
            f"        enum_entry<{fq_name}>{{{fq_name}::{entry.name}, {cpp_string(entry.name)}, {cpp_string(doc_text)}, {canonical}}},"
        )
    lines += [
        "    };",
        "};",
    ]
    return lines


def emit_header(specs: list[EnumSpec], source_path: Path) -> str:
    lines = [
        "#pragma once",
        "",
        '#include "enum_support.h"',
        "#include <array>",
        "#include <string_view>",
        "",
        f"// Generated from {source_path.name}. Do not edit by hand.",
        "",
        "namespace test::enumgen_demo {",
        "",
    ]

    for index, spec in enumerate(specs):
        if index > 0:
            lines.append("")
        lines.extend(emit_enum(spec))

    lines += [
        "",
        "} // namespace test::enumgen_demo",
        "",
        "namespace test::enum_support {",
        "",
    ]

    for index, spec in enumerate(specs):
        if index > 0:
            lines.append("")
        lines.extend(emit_traits(spec))

    lines += [
        "",
        "} // namespace test::enum_support",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: generate_enums.py <input.def> <output.h>", file=sys.stderr)
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    specs = parse_def(input_path)
    output_path.write_text(emit_header(specs, input_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
