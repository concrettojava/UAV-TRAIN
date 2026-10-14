"""Static source audit for the standalone training package."""
import ast
from pathlib import Path


def test_self_contained_project_imports():
    root = Path(__file__).resolve().parents[2]
    files = sorted(root.glob("envs/*.py")) + sorted(root.glob("models/*.py"))
    files += sorted(root.glob("training/**/*.py"))
    assert files
    for path in files:
        assert not path.is_symlink()
        tree = ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                assert not any(alias.name.startswith("uavmarl") for alias in node.names), path
            if isinstance(node,ast.ImportFrom) and node.module:
                assert not node.module.startswith("uavmarl"), path
    assert not (root / "requirements.txt").exists()
    assert (root / "run.sh").is_file()
    assert not list((root / "outputs/runs/MBDT-DualScale-256").glob("*.pt"))
