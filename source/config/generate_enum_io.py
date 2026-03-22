#!/usr/bin/env python3
"""! Generate explicit enum2sv, sv2enum, and flag2str specializations.

The handwritten headers in ``source/config/enums/*.h`` are the source of truth.
This generator parses a small, controlled subset of C++:

- exactly one ``enum class`` per header
- one enumerator per line
- optional trailing ``/*!< ... */`` member docs, including multiline docs
- optional explicit values such as ``1 << 3`` or ``DEFAULT = INIT | STUCK``
- optional ``allow_bitops`` sentinel as the final enumerator

The generated output is intentionally simple:

- one ``.cpp`` per enum
- no declarations header
- no lookup tables
- no helper classes
- only direct specializations of ``enum2sv``, ``sv2enum``, and ``flag2str``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

ENUM_DECL_RE = re.compile(r"^\s*enum\s+class\s+([A-Za-z_]\w*)(?:\s*:\s*([^{]+))?\s*\{\s*$")
ITEM_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:=\s*(.*?))?\s*,?\s*$")
IDENT_RE = re.compile(r"[A-Za-z_]\w*")
ZERO_EXPR_RE = re.compile(r"^0(?:[uUlL]|[uU][lL]?|[lL][uU]?)?$")


@dataclass(frozen=True)
class EnumItem:
    """! Parsed metadata for one enumerator."""

    name: str
    expr: str | None
    doc: str
    is_bitops_sentinel: bool
    is_alias: bool
    is_zero_value: bool


@dataclass(frozen=True)
class EnumSpec:
    """! Parsed metadata for one enum header."""

    name: str
    header_name: str
    underlying: str | None
    is_bitflag: bool
    items: list[EnumItem]


def clean_doc_fragment(text: str) -> str:
    """! Strip doxygen markers and fold multiline item docs into one line."""
    text = text.replace("/*!<", "").replace("*/", "")
    parts: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("*"):
            line = line[1:].lstrip()
        if line:
            parts.append(line)
    return " ".join(parts)


def cpp_string(text: str) -> str:
    """! Escape a Python string as a C++ string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def comment_suffix(doc: str) -> str:
    """! Render parsed docs as a trailing C++ comment when available."""
    if not doc:
        return ""
    return f" // {doc}"


def expr_looks_like_alias(expr: str | None) -> bool:
    """! Treat bitflag expressions that reference other enumerators as aliases."""
    if expr is None:
        return False
    return IDENT_RE.search(expr) is not None


def expr_is_zero(expr: str | None, index: int) -> bool:
    """! Detect zero-valued canonical flags without evaluating arbitrary C++."""
    if expr is None:
        return index == 0
    return ZERO_EXPR_RE.fullmatch(expr.strip()) is not None


def parse_header(path: Path) -> EnumSpec:
    """! Parse one handwritten enum header into a normalized EnumSpec."""
    lines = path.read_text().splitlines()
    enum_name = None
    underlying = None
    items: list[EnumItem] = []
    in_enum = False
    current_name: str | None = None
    current_expr: str | None = None
    current_doc_parts: list[str] = []
    collecting_doc = False

    def flush_current() -> None:
        """! Finalize the current enumerator once its doc block has ended."""
        nonlocal current_name, current_expr, current_doc_parts
        if current_name is None:
            return
        is_sentinel = current_name == "allow_bitops"
        is_alias = expr_looks_like_alias(current_expr)
        is_zero = expr_is_zero(current_expr, len(items))
        items.append(
            EnumItem(
                name=current_name,
                expr=current_expr,
                doc=" ".join(part for part in current_doc_parts if part).strip(),
                is_bitops_sentinel=is_sentinel,
                is_alias=is_alias,
                is_zero_value=is_zero,
            )
        )
        current_name = None
        current_expr = None
        current_doc_parts = []

    for lineno, raw in enumerate(lines, start=1):
        if not in_enum:
            match = ENUM_DECL_RE.match(raw)
            if match:
                enum_name = match.group(1)
                underlying = match.group(2).strip() if match.group(2) else None
                in_enum = True
            continue

        if collecting_doc:
            current_doc_parts.append(clean_doc_fragment(raw))
            if "*/" in raw:
                collecting_doc = False
                flush_current()
            continue

        stripped = raw.strip()
        if not stripped:
            continue
        if stripped == "};":
            flush_current()
            break
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue

        code = raw
        inline_doc = None
        if "/*!<" in raw:
            code, inline_doc = raw.split("/*!<", 1)
            inline_doc = "/*!<" + inline_doc

        match = ITEM_RE.match(code)
        if not match:
            raise RuntimeError(f"{path}:{lineno}: unsupported enum item format: {raw}")

        current_name = match.group(1)
        current_expr = match.group(2).strip() if match.group(2) else None

        if inline_doc is None:
            flush_current()
            continue

        current_doc_parts.append(clean_doc_fragment(inline_doc))
        if "*/" in inline_doc:
            flush_current()
        else:
            collecting_doc = True

    if enum_name is None or not items:
        raise RuntimeError(f"{path}: failed to parse enum header")

    is_bitflag = any(item.is_bitops_sentinel for item in items)
    return EnumSpec(
        name=enum_name,
        header_name=path.name,
        underlying=underlying,
        is_bitflag=is_bitflag,
        items=items,
    )


def enum_items(spec: EnumSpec) -> list[EnumItem]:
    """! Return all non-sentinel enumerators in source order."""
    return [item for item in spec.items if not item.is_bitops_sentinel]


def canonical_flag_items(spec: EnumSpec) -> list[EnumItem]:
    """! Return the bitflag members that should appear in expanded flag2str output."""
    return [
        item
        for item in enum_items(spec)
        if not item.is_alias and not item.is_zero_value
    ]


def enum2sv_items(spec: EnumSpec) -> list[EnumItem]:
    """! Return the members that should be addressable through enum2sv."""
    if not spec.is_bitflag:
        return enum_items(spec)
    return [item for item in enum_items(spec) if not item.is_alias]


def zero_item(spec: EnumSpec) -> EnumItem | None:
    """! Return the canonical zero-valued item for a bitflag enum."""
    for item in enum_items(spec):
        if not item.is_alias and item.is_zero_value:
            return item
    return None


def emit_enum2sv(spec: EnumSpec) -> list[str]:
    """! Emit the explicit enum2sv specialization for one enum."""
    lines = [
        "template<>",
        f"std::string_view enum2sv({spec.name} item) noexcept {{",
    ]

    for member in enum2sv_items(spec):
        lines.append(
            f"    if(item == {spec.name}::{member.name}) return {cpp_string(member.name)};{comment_suffix(member.doc)}"
        )

    lines.append(f'    return "{spec.name}::UNDEFINED";')
    lines.append("}")
    return lines


def emit_sv2enum(spec: EnumSpec) -> list[str]:
    """! Emit the explicit sv2enum specialization for one enum."""
    lines = [
        "template<>",
        f"{spec.name} sv2enum<{spec.name}>(std::string_view item) {{",
    ]

    for member in enum_items(spec):
        lines.append(
            f"    if(item == {cpp_string(member.name)}) return {spec.name}::{member.name};{comment_suffix(member.doc)}"
        )

    if spec.is_bitflag:
        lines += [
            "",
            f"    auto value = static_cast<{spec.name}>(0);",
            "    while(!item.empty()) {",
            "        auto pos   = item.find('|');",
            "        auto token = item.substr(0, pos);",
            '        if(token.empty()) throw std::runtime_error("sv2enum given invalid string item: " + std::string(token));',
        ]

        first = True
        for member in enum_items(spec):
            keyword = "if" if first else "else if"
            first = False
            lines.append(
                f"        {keyword}(token == {cpp_string(member.name)}) value |= {spec.name}::{member.name};{comment_suffix(member.doc)}"
            )

        lines += [
            '        else throw std::runtime_error("sv2enum given invalid string item: " + std::string(token));',
            "        if(pos == std::string_view::npos) return value;",
            "        item.remove_prefix(pos + 1);",
            "    }",
        ]

    lines += [
        '    throw std::runtime_error("sv2enum given invalid string item: " + std::string(item));',
        "}",
    ]
    return lines


def emit_flag2str(spec: EnumSpec) -> list[str]:
    """! Emit the explicit flag2str specialization for one allow_bitops enum."""
    zero = zero_item(spec)
    lines = [
        "template<>",
        f"std::string flag2str(const {spec.name} &item) noexcept {{",
    ]

    if zero is not None:
        lines.append(
            f"    if(item == {spec.name}::{zero.name}) return {cpp_string(zero.name)};{comment_suffix(zero.doc)}"
        )

    lines += [
        "    std::string value;",
    ]

    for member in canonical_flag_items(spec):
        lines += [
            f"    if(has_flag(item, {spec.name}::{member.name})) {{{comment_suffix(member.doc)}",
            '        if(!value.empty()) value += "|";',
            f"        value += {cpp_string(member.name)};",
            "    }",
        ]

    lines += [
        '    if(value.empty()) return std::string(enum2sv(item));',
        "    return value;",
        "}",
    ]
    return lines


def emit_cpp(spec: EnumSpec) -> str:
    """! Emit the generated implementation file for one enum."""
    lines = [
        "// Generated from source/config/enums/{}. Do not edit by hand.".format(spec.header_name),
        "",
        '#include "config/enum_utils.h"',
        f'#include "config/enums/{spec.header_name}"',
        "",
        "#include <stdexcept>",
        "#include <string>",
        "#include <string_view>",
        "",
    ]

    lines += emit_enum2sv(spec)
    lines.append("")
    lines += emit_sv2enum(spec)
    if spec.is_bitflag:
        lines.append("")
        lines += emit_flag2str(spec)
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """! Parse input headers and write one generated implementation file per enum."""
    if len(argv) < 3:
        print(f"Usage: {argv[0]} <output-dir> <enum-header> [<enum-header> ...]", file=sys.stderr)
        return 1

    output_dir = Path(argv[1])
    headers = [Path(arg) for arg in argv[2:]]

    # Parse every header up front so generation either succeeds completely or fails loudly.
    specs = [parse_header(path) for path in headers]

    output_dir.mkdir(parents=True, exist_ok=True)
    # Emit one direct specialization unit per enum to keep rebuild scope narrow.
    for spec in specs:
        (output_dir / f"{spec.name}_enum.cpp").write_text(emit_cpp(spec))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
