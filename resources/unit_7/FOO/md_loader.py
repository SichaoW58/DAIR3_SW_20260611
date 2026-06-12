"""
md_loader.py
Load and compose system-prompt content from Markdown files.

The single-agent scripts and the multi-agent orchestrator both read their
instructions from disk: a common .md (shared persona conventions) plus a
role .md (task-specific behavior). Variables like {user} and {name} are
substituted in. Code fences and other literal '{...}' content in the
markdown are NOT touched -- substitution is by exact placeholder names.

By Juan B. Gutiérrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import os


def read_md_file(path):
    if not path:
        return ""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Markdown file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def substitute(text, variables):
    for key, value in (variables or {}).items():
        text = text.replace("{" + key + "}", str(value))
    return text


def load_persona(common_path, role_path, variables=None):
    common = read_md_file(common_path)
    role = read_md_file(role_path)
    if common and role:
        combined = f"{common}\n\n---\n\n{role}"
    else:
        combined = common or role
    return substitute(combined, variables)
