import yaml
from pathlib import Path
import re

# Load yaml
def yaml_load(append_filename=False):
  file = "config.yaml"
  assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
  with open(file, errors="ignore", encoding="utf-8") as f:
    s = f.read()  # string
    if not s.isprintable():
      s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
  data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
  if append_filename:
    data["yaml_file"] = str(file)
  return data
