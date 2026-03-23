#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tomllib


@dataclass(frozen=True)
class Spec:
    ctype: str
    var: str
    default_toml: str
    cli: str
    doc: str
    toml_path: str
    section: str
    key: str
    leading_comments: tuple[str, ...]


VECTOR_RE = re.compile(r"^std::vector<\s*(.+?)\s*>$")
ARRAY_RE = re.compile(r"^std::array<\s*(.+?)\s*,\s*(\d+)\s*>$")
COMPLEX_RE = re.compile(r"^std::complex<\s*(.+?)\s*>$")
ENUM_RE = re.compile(r"enum\s+class\s+([A-Za-z_]\w*)\s*(?:\s*:\s*[^ {]+)?\s*\{(.*?)\};", re.DOTALL)
INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)

SCALAR_TYPES = {
    "bool",
    "int",
    "long",
    "unsigned int",
    "double",
}


def cpp_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def parse_value(default_toml: str):
    return tomllib.loads(f"value = {default_toml}")["value"]


def strip_cpp_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    return text


def find_source_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if parent.name == "source":
            return parent
    return path.parent


def resolve_include(include: str, current: Path, source_root: Path) -> Path | None:
    candidates = [
        current.parent / include,
        source_root / include,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_project_headers(path: str) -> str:
    source_root = find_source_root(Path(path).resolve())
    visited: set[Path] = set()
    chunks: list[str] = []

    def visit(header: Path) -> None:
        header = header.resolve()
        if header in visited or not header.exists():
            return
        visited.add(header)

        text = header.read_text()
        chunks.append(text)
        for include in INCLUDE_RE.findall(text):
            resolved = resolve_include(include, header, source_root)
            if resolved is None:
                continue
            if source_root == resolved or source_root in resolved.parents:
                visit(resolved)

    visit(Path(path))
    return "\n".join(chunks)


def split_var(var: str) -> tuple[str, str]:
    path = var.removeprefix("settings::")
    return path.rsplit("::", 1)


def toml_path(var: str) -> str:
    return var.removeprefix("settings::").replace("::", ".")


def auto_cli(var: str) -> str:
    return "--" + var.removeprefix("settings::").replace("::", "-").replace("_", "-")


def parse_specs(path: str) -> list[Spec]:
    specs: list[Spec] = []
    pending_comments: list[str] = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            if line.startswith("##"):
                pending_comments.append(line[2:].lstrip())
            continue
        separators = [idx for idx, char in enumerate(raw) if char == ";"]
        if len(separators) < 4:
            raise RuntimeError(f"{path}:{lineno}: expected 5 semicolon-separated fields")

        first, second, third, fourth = separators[:4]
        ctype = raw[:first].strip()
        var = raw[first + 1:second].strip()
        default_toml = raw[second + 1:third].strip()
        cli = raw[third + 1:fourth].strip()
        doc = raw[fourth + 1:].strip()
        path_toml = toml_path(var)
        section, key = path_toml.rsplit(".", 1)
        resolved_cli = cli if cli else auto_cli(var)
        parse_value(default_toml)
        specs.append(
            Spec(
                ctype=ctype,
                var=var,
                default_toml=default_toml,
                cli=resolved_cli,
                doc=doc,
                toml_path=path_toml,
                section=section,
                key=key,
                leading_comments=tuple(pending_comments),
            )
        )
        pending_comments.clear()
    return specs


def parse_enum_definitions(path: str, used_types: set[str]) -> dict[str, tuple[list[str], bool]]:
    text = strip_cpp_comments(load_project_headers(path))
    enums: dict[str, tuple[list[str], bool]] = {}

    for match in ENUM_RE.finditer(text):
        enum_name = match.group(1)
        if enum_name not in used_types:
            continue

        entries: list[str] = []
        is_bitflag = False
        for raw_entry in match.group(2).split(","):
            entry = raw_entry.strip()
            if not entry:
                continue
            name = entry.split("=", 1)[0].strip()
            if not name:
                continue
            if name == "allow_bitops":
                is_bitflag = True
                continue
            entries.append(name)

        if entries:
            enums[enum_name] = (entries, is_bitflag)

    missing = sorted(used_types.difference(enums.keys()))
    if missing:
        raise RuntimeError(f"Could not derive enum metadata for: {', '.join(missing)}")
    return enums


def cpp_scalar(type_name: str, value) -> str:
    if type_name == "bool":
        return "true" if value else "false"
    if type_name in {"int", "long", "unsigned int"}:
        return str(value)
    if type_name == "double":
        return format(value, ".17g")
    if type_name == "std::string":
        return cpp_string(value)
    return cpp_string(str(value))


def cpp_initializer(spec: Spec) -> str:
    parsed = parse_value(spec.default_toml)
    if spec.ctype in SCALAR_TYPES:
        return spec.default_toml
    if spec.ctype == "std::string":
        return spec.default_toml
    if match := VECTOR_RE.match(spec.ctype):
        elem_type = match.group(1)
        values = ", ".join(cpp_scalar(elem_type, item) for item in parsed)
        return "{" + values + "}"
    if match := ARRAY_RE.match(spec.ctype):
        elem_type = match.group(1)
        values = ", ".join(cpp_scalar(elem_type, item) for item in parsed)
        return "{" + values + "}"
    if match := COMPLEX_RE.match(spec.ctype):
        elem_type = match.group(1)
        return "{" + cpp_scalar(elem_type, parsed[0]) + ", " + cpp_scalar(elem_type, parsed[1]) + "}"
    return f"sv2enum<{spec.ctype}>({spec.default_toml})"


def is_generated_enum_type(type_name: str) -> bool:
    if type_name in SCALAR_TYPES:
        return False
    if type_name == "std::string":
        return False
    if VECTOR_RE.match(type_name):
        return False
    if ARRAY_RE.match(type_name):
        return False
    if COMPLEX_RE.match(type_name):
        return False
    return True


def emit_settings_generated_h(specs: list[Spec]) -> str:
    class NamespaceNode:
        def __init__(self) -> None:
            self.children: dict[str, NamespaceNode] = {}
            self.specs: list[Spec] = []

    root = NamespaceNode()
    for spec in specs:
        parts = spec.var.removeprefix("settings::").split("::")
        node = root
        for namespace in parts[:-1]:
            node = node.children.setdefault(namespace, NamespaceNode())
        node.specs.append(spec)

    def spec_name(spec: Spec) -> str:
        return spec.var.rsplit("::", 1)[1]

    def emit_node(namespace: str, node: NamespaceNode, indent: int) -> list[str]:
        pad = " " * indent
        lines = [f"{pad}namespace {namespace} {{"]

        if node.specs:
            type_width = max(len(spec.ctype) for spec in node.specs)
            name_width = max(len(spec_name(spec)) for spec in node.specs)
            for spec in node.specs:
                line = (
                    f"{pad}    inline {spec.ctype:<{type_width}} "
                    f"{spec_name(spec):<{name_width}} = {cpp_initializer(spec)};"
                )
                if spec.doc:
                    line += f" /*!< {spec.doc} */"
                lines.append(line)

        child_names = list(node.children.keys())
        for child_index, child_name in enumerate(child_names):
            if node.specs or child_index > 0:
                lines.append("")
            lines.extend(emit_node(child_name, node.children[child_name], indent + 4))

        lines.append(f"{pad}}}")
        return lines

    def emit_root(node: NamespaceNode) -> list[str]:
        lines = ["namespace settings {"]
        child_names = list(node.children.keys())
        for child_index, child_name in enumerate(child_names):
            if child_index > 0:
                lines.append("")
            lines.extend(emit_node(child_name, node.children[child_name], 4))
        if child_names:
            lines.append("")
        lines.append("    void load(std::string_view path);")
        lines.append("    int  parse(int argc, char **argv);")
        lines.append("}")
        return lines

    out = [
        "#pragma once",
        "",
        '#include "config/enums.h"',
        "#include <array>",
        "#include <complex>",
        "#include <string>",
        "#include <string_view>",
        "#include <vector>",
        "",
        "// Generated from tests/tomlpp_parser/settings.def. Do not edit by hand.",
        "",
        "/* clang-format off */",
        "",
    ]
    out.extend(emit_root(root))
    out += [
        "",
        "/* clang-format on */",
        "",
    ]
    return "\n".join(out)


def emit_setting_specs_generated_h(specs: list[Spec]) -> str:
    out = [
        "#pragma once",
        "",
        '#include "settings.h"',
        "#include <string_view>",
        "#include <tuple>",
        "",
        "// Generated from tests/tomlpp_parser/settings.def. Do not edit by hand.",
        "",
        "namespace test::tomlpp::generated {",
        "    template<typename T>",
        "    struct SettingSpec {",
        "        T               *value;",
        "        std::string_view toml_path;",
        "        std::string_view doc;",
        "        std::string_view cli;",
        "    };",
        "",
        "    template<typename T>",
        "    constexpr auto make_setting_spec(T &value, std::string_view toml_path, std::string_view doc, std::string_view cli) {",
        "        return SettingSpec<T>{.value = &value, .toml_path = toml_path, .doc = doc, .cli = cli};",
        "    }",
        "",
        "    inline const auto &setting_specs() {",
        "        static const auto specs = std::tuple{",
    ]
    for index, spec in enumerate(specs):
        comma = "," if index + 1 < len(specs) else ""
        out.append(
            f"            make_setting_spec({spec.var}, {cpp_string(spec.toml_path)}, {cpp_string(spec.doc)}, {cpp_string(spec.cli)}){comma}"
        )
    out += [
        "        };",
        "        return specs;",
        "    }",
        "",
        "    template<typename F>",
        "    void for_each_setting(F &&func) {",
        "        std::apply([&](const auto &... spec) { (func(spec), ...); }, setting_specs());",
        "    }",
        "}",
        "",
    ]
    return "\n".join(out)


def emit_enum_choices_generated_h(enum_defs: dict[str, tuple[list[str], bool]]) -> str:
    out = [
        "#pragma once",
        "",
        '#include "config/enums.h"',
        "#include <array>",
        "#include <string_view>",
        "",
        "// Generated from source/config/enums.h. Do not edit by hand.",
        "",
        "namespace test::tomlpp::generated {",
        "    template<typename Enum>",
        "    struct EnumInfo;",
    ]

    for enum_name, (choices, is_bitflag) in enum_defs.items():
        joined_choices = ", ".join(cpp_string(choice) for choice in choices)
        out += [
            "",
            "    template<>",
            f"    struct EnumInfo<{enum_name}> {{",
            f"        static constexpr bool is_bitflag = {'true' if is_bitflag else 'false'};",
            f"        static constexpr std::array<std::string_view, {len(choices)}> choices = {{{joined_choices}}};",
            "    };",
        ]

    out += [
        "}",
        "",
    ]
    return "\n".join(out)


def emit_parse_generated_cpp() -> str:
    out = [
        '#include "parse_generated_helpers.h"',
        '#include "setting_specs_generated.h"',
        "#include <CLI/CLI.hpp>",
        "",
        "// Generated from tests/tomlpp_parser/settings.def. Do not edit by hand.",
        "",
        "int settings::parse(int argc, char **argv) {",
        '    CLI::App app{"Generated CLI11 bindings for the tomlpp parser experiment"};',
        "    app.get_formatter()->column_width(100);",
        "    app.option_defaults()->always_capture_default();",
        "    app.allow_extras(false);",
        "    test::tomlpp::generated::for_each_setting([&](const auto &spec) { test::tomlpp::bind_option(app, spec); });",
        "    try {",
        "        app.parse(argc, argv);",
        "    } catch(const CLI::ParseError &e) {",
        "        throw test::tomlpp::CliExit{app.exit(e)};",
        "    }",
        "    return 0;",
        "}",
        "",
    ]
    return "\n".join(out)


def emit_input_toml(specs: list[Spec]) -> str:
    out = [
        "# Generated from tests/tomlpp_parser/settings.def. Do not edit by hand.",
    ]
    idx = 0
    while idx < len(specs):
        section = specs[idx].section
        group: list[Spec] = []
        while idx < len(specs) and specs[idx].section == section:
            group.append(specs[idx])
            idx += 1

        if out:
            out.append("")
        for comment in group[0].leading_comments:
            out.append(f"# {comment}")
        out.append(f"[{section}]")
        key_width = max(len(spec.key) for spec in group)
        lhs_values = [f"{spec.key:<{key_width}} = {spec.default_toml}" for spec in group]
        lhs_width = max(len(lhs) for lhs in lhs_values)
        for spec, lhs in zip(group, lhs_values):
            extra_comments = spec.leading_comments if spec is not group[0] else ()
            for comment in extra_comments:
                out.append(f"# {comment}")
            if spec.doc:
                out.append(f"{lhs:<{lhs_width}}  # {spec.doc}")
            else:
                out.append(lhs)
    out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if len(argv) != 8:
        raise SystemExit(
            "usage: generate_settings.py <settings.def> <source/config/enums.h> <settings_generated.h> "
            "<setting_specs_generated.h> <enum_choices_generated.h> <parse_generated.cpp> <input.test.toml>"
        )
    spec_path, enums_h, settings_h, specs_h, enum_choices_h, parse_cpp, toml_path_out = argv[1:]
    specs = parse_specs(spec_path)
    enum_defs = parse_enum_definitions(enums_h, {spec.ctype for spec in specs if is_generated_enum_type(spec.ctype)})
    Path(settings_h).write_text(emit_settings_generated_h(specs))
    Path(specs_h).write_text(emit_setting_specs_generated_h(specs))
    Path(enum_choices_h).write_text(emit_enum_choices_generated_h(enum_defs))
    Path(parse_cpp).write_text(emit_parse_generated_cpp())
    Path(toml_path_out).write_text(emit_input_toml(specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
