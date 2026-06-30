from pathlib import Path
import hashlib
import py_compile


ROOT = Path(__file__).resolve().parents[1]


def test_supported_python_sources_parse(tmp_path):
    for base in ("src", "scripts", "tests"):
        for path in sorted((ROOT / base).rglob("*.py")):
            digest = hashlib.sha1(str(path.relative_to(ROOT)).encode("utf-8")).hexdigest()
            py_compile.compile(str(path), cfile=str(tmp_path / f"{digest}.pyc"), doraise=True)
