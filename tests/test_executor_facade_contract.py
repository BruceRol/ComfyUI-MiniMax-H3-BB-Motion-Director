import ast
from pathlib import Path


def _legacy_kwargs():
    tree = ast.parse(Path("director/executor_core_legacy.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_director_plan_core":
            return {arg.arg for arg in node.args.args + node.args.kwonlyargs}
    raise AssertionError("legacy execute_director_plan_core not found")


def _facade_call_kwargs():
    tree = ast.parse(Path("director/executor_core.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == "call_kwargs":
                return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    raise AssertionError("facade call_kwargs dict not found")


def test_facade_only_passes_arguments_the_legacy_contract_accepts():
    # The facade layers new behaviour over the frozen legacy signature; a key the
    # legacy does not declare raises TypeError at run time (width/height did).
    missing = _facade_call_kwargs() - _legacy_kwargs()
    assert not missing, f"facade passes kwargs the legacy does not accept: {sorted(missing)}"


def test_facade_does_not_declare_unused_canvas_arguments():
    # The canvas comes from the plan (resolve_h3_canvas), not from the node widgets,
    # so the facade must not accept width/height as if it forwarded them.
    facade = Path("director/executor_core.py").read_text(encoding="utf-8")
    signature = facade.split("def execute_director_plan_core", 1)[1].split("):", 1)[0]
    assert "width" not in signature
    assert "height" not in signature

    # The node still passes width/height to the plan builder; the executor call must not.
    node = Path("nodes/director.py").read_text(encoding="utf-8")
    call = node.split("execute_director_plan_core(", 1)[1].split("\n        )", 1)[0]
    assert "width=width," not in call
    assert "height=height," not in call
    assert "dirname_prefix=dirname_prefix," in call
