#!/usr/bin/env python3
"""! Generate direct enum2sv, sv2enum, and flag2sv specializations for enumcppgen.

This generator intentionally parses only a very small C++ subset:
- one enum per header
- one enumerator per line
- optional trailing `/*!< ... */` member docs
- optional `// enumgen: bitflag` marker before the enum declaration

The goal is to keep the handwritten enum headers as the source of truth and move
only the conversion code out of headers and into generated `.cpp` files.
The generated parsers intentionally do not trim whitespace. Inputs must match the
handwritten enumerator spellings exactly, except that bitflags may still be split
on the raw `|` character.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys


ENUM_RE = re.compile(r"enum\s+class\s+([A-Za-z_]\w*)(?:\s*:\s*(.+?))?\s*\{")
ITEM_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:=\s*([^,]+))?\s*,\s*(?:/\*!\<\s*(.*?)\s*\*/)?\s*$")
NAMESPACE_RE = re.compile(r"namespace\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\{")


@dataclass(frozen=True)
class EnumItem:
    """! Metadata for one enumerator line parsed from a handwritten header."""

    name: str
    expr: str | None
    doc: str
    canonical: bool


@dataclass(frozen=True)
class EnumSpec:
    """! Metadata for one parsed enum header."""

    namespace: str
    name: str
    underlying: str | None
    is_bitflag: bool
    items: list[EnumItem]


def cpp_string(text: str) -> str:
    """! Escape a Python string as a C++ string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def with_trailing_comment(code: str, doc: str, indent: str = "") -> str:
    """! Attach the parsed enumerator documentation as a trailing C++ line comment."""
    if not doc:
        return f"{indent}{code}"
    return f"{indent}{code} // {doc}"


def parse_header(path: Path) -> EnumSpec:
    """! Parse one handwritten enum header using the constrained enumcppgen format."""
    namespace = None
    enum_name = None
    underlying = None
    is_bitflag = False
    items: list[EnumItem] = []
    in_enum = False

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        # Pick up the simple out-of-band marker that says this enum supports bitwise flags.
        if line == "// enumgen: bitflag":
            is_bitflag = True
            continue

        # Capture the namespace once so generated specializations land in the same scope.
        if namespace is None:
            if match := NAMESPACE_RE.match(line):
                namespace = match.group(1)
                continue

        # Ignore everything until we find the actual enum declaration.
        if not in_enum:
            if match := ENUM_RE.search(line):
                enum_name = match.group(1)
                underlying = match.group(2).strip() if match.group(2) else None
                in_enum = True
            continue

        # Stop once the enum closes.
        if line.startswith("};"):
            break

        # Skip comment-only lines inside the enum body.
        if line.startswith("/*") or line.startswith("*") or line.startswith("//"):
            continue

        # Parse one enumerator per line, with an optional explicit value and trailing doc.
        match = ITEM_RE.match(raw)
        if not match:
            raise RuntimeError(f"{path}:{lineno}: unsupported enum item format: {raw}")

        expr = match.group(2).strip() if match.group(2) else None
        doc = match.group(3).strip() if match.group(3) else ""

        # For the demo, any flag expression containing '|' is treated as an alias/composite.
        canonical = not (is_bitflag and expr is not None and "|" in expr)
        items.append(EnumItem(name=match.group(1), expr=expr, doc=doc, canonical=canonical))

    if namespace is None or enum_name is None or not items:
        raise RuntimeError(f"{path}: failed to parse enum header")

    return EnumSpec(namespace=namespace, name=enum_name, underlying=underlying, is_bitflag=is_bitflag, items=items)


def emit_decls(specs: list[EnumSpec]) -> str:
    """! Emit explicit specialization declarations used by the stable public API."""
    lines = [
        "#pragma once",
        "",
        "#include <string>",
        "#include <string_view>",
        "#include <type_traits>",
        "",
        "// Generated from tests/enumcppgen/enums/*.h. Do not edit by hand.",
        "",
        "namespace test::enumcppgen_demo {",
        "",
    ]

    for spec in specs:
        if spec.is_bitflag:
            lines.append(f"template<> struct enable_bitops<{spec.name}> : std::true_type {{}};")
        lines += [
            f"template<> std::string_view enum2sv({spec.name} value) noexcept;",
            f"template<> {spec.name}       sv2enum<{spec.name}>(std::string_view text);",
            f"template<> std::string       flag2sv({spec.name} value);",
            "",
        ]

    lines += [
        "}",
        "",
    ]
    return "\n".join(lines)


def emit_enum2sv(spec: EnumSpec) -> list[str]:
    """! Emit the direct enum2sv specialization for one enum."""
    lines = [
        "template<>",
        f"std::string_view enum2sv({spec.name} value) noexcept {{",
        "    switch(value) {",
    ]

    for item in spec.items:
        lines.append(with_trailing_comment(f"case {spec.name}::{item.name}: return {cpp_string(item.name)};", item.doc, "        "))

    lines += [
        f'        default: return "{spec.name}::UNDEFINED";',
        "    }",
        "}",
    ]
    return lines


def emit_exact_match_chain(spec: EnumSpec, input_name: str, return_statement: str) -> list[str]:
    """! Emit a linear exact-string match chain for one enum."""
    lines: list[str] = []
    for item in spec.items:
        lines.append(with_trailing_comment(f"if({input_name} == {cpp_string(item.name)}) {return_statement.format(name=item.name)}", item.doc, "    "))
    return lines


def emit_sv2enum(spec: EnumSpec) -> list[str]:
    """! Emit the direct sv2enum specialization for one enum."""
    lines = [
        "template<>",
        f"{spec.name} sv2enum<{spec.name}>(std::string_view text) {{",
    ]

    # Always prefer exact-name matches first so aliases like DEFAULT remain available.
    lines += emit_exact_match_chain(spec, "text", f"return {spec.name}::{{name}};")

    if spec.is_bitflag:
        lines += [
            "",
            f"    auto value = static_cast<{spec.name}>(0);",
            "    while(!text.empty()) {",
            "        auto pos   = text.find('|');",
            "        auto token = text.substr(0, pos);",
            f'        if(token.empty()) throw std::runtime_error("Invalid {spec.name} token: empty");',
        ]

        # Emit the token parser inline instead of generating lookup helpers.
        for index, item in enumerate(spec.items):
            prefix = "if" if index == 0 else "else if"
            lines.append(
                with_trailing_comment(f"{prefix}(token == {cpp_string(item.name)}) value |= {spec.name}::{item.name};", item.doc, "        ")
            )

        lines += [
            f'        else throw std::runtime_error("Invalid {spec.name} token: " + std::string(token));',
            "        if(pos == std::string_view::npos) return value;",
            "        text.remove_prefix(pos + 1);",
            "    }",
            f'    throw std::runtime_error("Invalid {spec.name}: empty string");',
        ]
    else:
        lines += [
            f'    throw std::runtime_error("Invalid {spec.name}: " + std::string(text));',
        ]

    lines.append("}")
    return lines


def emit_flag2sv(spec: EnumSpec) -> list[str]:
    """! Emit the direct flag2sv specialization for one enum."""
    lines = [
        "template<>",
        f"std::string flag2sv({spec.name} value) {{",
    ]

    if not spec.is_bitflag:
        lines += [
            "    return std::string(enum2sv(value));",
            "}",
        ]
        return lines

    lines += [
        f"    using U = std::underlying_type_t<{spec.name}>;",
        "    if(static_cast<U>(value) == 0) return std::string(enum2sv(value));",
        "    std::string out;",
    ]

    # Emit the flag expansion inline instead of generating tables or helper functions.
    for item in spec.items:
        if not item.canonical:
            continue
        lines += [
            with_trailing_comment(f"if(has_flag(value, {spec.name}::{item.name})) {{", item.doc, "    "),
            '        if(!out.empty()) out += "|";',
            f"        out += {cpp_string(item.name)};",
            "    }",
        ]

    lines += [
        "    if(out.empty()) return std::string(enum2sv(value));",
        "    return out;",
        "}",
    ]
    return lines


def emit_cpp(spec: EnumSpec) -> str:
    """! Emit one generated `.cpp` file for one enum header."""
    lines = [
        '#include "enums.h"',
        "",
        "#include <stdexcept>",
        "",
        f"namespace {spec.namespace} {{",
        "",
    ]

    # The generated file is intentionally just the three direct specializations.
    lines += emit_enum2sv(spec)
    lines += ["", *emit_sv2enum(spec), "", *emit_flag2sv(spec), "", "}"]
    return "\n".join(lines)


def main() -> int:
    """! Parse the input enum headers and write the generated declarations and `.cpp` files."""
    if len(sys.argv) < 3:
        print("usage: generate_enum_io.py <output_dir> <enum_header> [<enum_header>...]", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    headers = [Path(arg) for arg in sys.argv[2:]]

    # Parse every handwritten header first so generation stays deterministic.
    specs = [parse_header(header) for header in headers]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "enum_generated_decls.h").write_text(emit_decls(specs))

    # Emit one translation unit per enum so changes stay localized.
    for spec in specs:
        (output_dir / f"{spec.name}_enum.cpp").write_text(emit_cpp(spec))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
