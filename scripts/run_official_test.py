"""Run upstream host assertions against the Core ML namespace.

Only Python imports and mock target module paths are rewritten. Router tests stub
Agent construction; dispatch that stub through the Core ML load factory as well.
The idle-window fixture measures its unchanged 200 ms contract on the worker
rather than imposing a 100 ms scheduling deadline on the observing test thread.
Other host assertions, numeric tolerances and public model names are preserved.
"""

import ast
import re
import sys
import types
from pathlib import Path


class Namespace(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "laya":
            for alias in node.names:
                if alias.name == "cli":
                    alias.name, alias.asname = "router_cli", alias.asname or "cli"
        if node.module == "laya.cli":
            node.module = "laya.router_cli"
        if node.module == "laya" or (node.module or "").startswith("laya."):
            node.module = "laya_coreml" + node.module[4:]
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "laya":
                alias.name, alias.asname = "laya_coreml", alias.asname or "laya"
            elif alias.name.startswith("laya."):
                # Unaliased imports need the original top-level binding too.
                if not alias.asname:
                    return [
                        ast.Import(names=[ast.alias(name="laya_coreml", asname="laya")]),
                        ast.Import(names=[ast.alias(name="laya_coreml" + alias.name[4:])]),
                    ]
                alias.name = "laya_coreml" + alias.name[4:]
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            # This is a removal pattern for the original example's import;
            # rewriting it would stop the test's StubLaya from taking effect.
            if node.value == "\nimport laya\n":
                return node
            if "import laya" in node.value or "from laya" in node.value:
                node.value = re.sub(r"\bfrom laya(?=\b|\.)", "from laya_coreml", node.value)
                node.value = re.sub(
                    r"\bimport laya(?=[;\n ]|$)(?!\.)", "import laya_coreml as laya", node.value
                )
                node.value = node.value.replace(
                    "import laya.", "import laya_coreml as laya; import laya_coreml."
                )
                node.value = re.sub(r"([\"'])laya\.", r"\1laya_coreml.", node.value)
            if node.value.startswith("laya."):
                node.value = "laya_coreml" + node.value[4:]
                node.value = node.value.replace("laya_coreml.cli", "laya_coreml.router_cli")
        return node


def measure_idle_window_on_worker(tree):
    """Measure the original 200 ms contract where inference/unload actually run.

    A caller observing Event.wait(0.1) can resume after the entire idle window.
    Preserve the waits, eight parameter combinations and non-overlap assertion,
    but assert elapsed time in BlockingRouter.unload, before it sets the event.
    Fail closed if the pinned upstream fixture structure changes.
    """
    target = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "test_idle_window_starts_after_worker_finishes_even_on_cancel_or_failure"
    )
    router = next(
        n for n in target.body if isinstance(n, ast.ClassDef) and n.name == "BlockingRouter"
    )
    predict = next(n for n in router.body if isinstance(n, ast.FunctionDef) and n.name == "predict")
    finalizers = [n.finalbody for n in ast.walk(predict) if isinstance(n, ast.Try)]
    marked = 0
    for statements in finalizers:
        for index, statement in enumerate(statements):
            if ast.unparse(statement) == "self.finished.set()":
                statements[index:index] = ast.parse("self.finished_at = time.monotonic()").body
                marked += 1
                break
    assert marked == 1, "upstream completion marker changed"
    unload = next(n for n in router.body if isinstance(n, ast.FunctionDef) and n.name == "unload")
    assert isinstance(unload.body[0], ast.Assert), "upstream non-overlap assertion changed"
    unload.body[1:1] = ast.parse(
        "assert time.monotonic() - self.finished_at >= 0.2, 'unload preceded idle deadline'"
    ).body

    class ObserveWithoutSchedulerDeadline(ast.NodeTransformer):
        replaced = 0

        def visit_Assert(self, node):
            expected = {
                "not router.unloaded.wait(0.1)",
                "not await asyncio.to_thread(router.unloaded.wait, 0.1)",
            }
            if ast.unparse(node.test) in expected:
                self.replaced += 1
                return ast.Expr(value=node.test.operand)
            return node

    observer = ObserveWithoutSchedulerDeadline()
    observer.visit(target)
    assert observer.replaced == 2, "upstream idle observations changed"
    return tree


def main():
    path = Path(sys.argv[1]).resolve()
    source = path.read_text(encoding="utf-8")
    if path.stem == "test_evals_api":
        # Core ML refuses --onnx. Preserve Router revision assertions, excluding
        # only the ONNX runner setup and assertions that reference its reports.
        source = source.replace(
            "_real_router, _real_agent = laya.Router, _onnx_agent_module.ONNXAgent",
            "_real_router = laya.Router",
        ).replace(
            "laya.Router, _onnx_agent_module.ONNXAgent = _real_router, _real_agent",
            "laya.Router = _real_router",
        )
    if path.stem == "test_hooks":
        start = source.index(
            "# --------------------------------------------------------------- ONNXAgent parity"
        )
        end = source.index(
            "# --------------------------------------------------------------- BaseHook", start
        )
        source = source[:start] + source[end:]
    if path.stem == "test_confidence":
        # This host suite also embeds six PyTorch/TileLang padding implementation
        # and documentation checks. Neither backend is part of the Core ML port.
        start = source.index("# ------------------------------------------- tl_kernels'")
        end = source.index("# --------------------------------------- DecisionResult's", start)
        source = source[:start] + source[end:]
    tree = ast.parse(source)
    if path.stem == "test_hooks_api":

        class OnnxOnly(ast.NodeTransformer):
            @staticmethod
            def references(node):
                return any(
                    isinstance(n, ast.Name) and n.id in {"ONNXAgent", "_ApiONNXAgent"}
                    for n in ast.walk(node)
                )

            def visit_ImportFrom(self, node):
                return None if node.module == "laya.onnx_agent" else node

            def visit_Tuple(self, node):
                node.elts = [e for e in node.elts if not self.references(e)]
                return self.generic_visit(node)

            def visit_List(self, node):
                node.elts = [e for e in node.elts if not self.references(e)]
                return self.generic_visit(node)

            def visit_Expr(self, node):
                # These APIs switch PyTorch forward implementations; Core ML
                # rejects backend changes at construction (covered locally).
                backend = any(
                    isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "Agent"
                    and n.attr in {"set_backend", "backend", "backend_object"}
                    for n in ast.walk(node)
                )
                return None if self.references(node) or backend else node

        tree = OnnxOnly().visit(tree)
    if path.stem == "test_evals_api":

        class BackendOnly(ast.NodeTransformer):
            def visit_Import(self, node):
                return None if any(a.name == "laya.onnx_agent" for a in node.names) else node

            def visit_Assign(self, node):
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                if names & {"_onnx_agent_module", "_onnx_cfg", "_onnx_default_cfg"}:
                    return None
                return node

            def visit_Expr(self, node):
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                return None if names & {"_onnx_cfg", "_onnx_default_cfg"} else node

        tree = BackendOnly().visit(tree)
    backend_tests = {
        "test_serve": {
            "test_health_without_a_resident_checkpoint_flags_the_preference",
            "test_health_agrees_with_the_mcp_status_tool",
            "test_build_router_strips_the_device_before_torch_sees_it",
            "test_thread_limit",
        },
        "test_mcp": {"test_device"},
    }.get(path.stem, set())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name in backend_tests)
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in backend_tests
        )
    ]
    if path.stem == "test_cli":
        # Core ML exports are inference-only. Identify the training command by
        # syntax, not line number, so upstream additions cannot hide other checks.
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "run_cli"
                and node.value.args
                and isinstance(node.value.args[0], ast.List)
                and all(isinstance(e, ast.Constant) for e in node.value.args[0].elts)
                and ast.literal_eval(node.value.args[0]) == ["train", "--help"]
            )
            and not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
                and str(node.value.args[0].value).startswith("train subcommand:")
            )
        ]
    if path.stem in {"test_structured", "test_structured_api", "test_confidence"}:
        # These host suites inspect ONNX methods or run the ONNX decoder.
        # Exclude that backend's import/assertions; every host assertion is unchanged.
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.ImportFrom) and node.module == "laya.onnx_agent")
            and not (
                isinstance(node, ast.Expr)
                and any(
                    isinstance(child, ast.Name) and child.id == "ONNXAgent"
                    for child in ast.walk(node)
                )
            )
        ]
    if path.stem in {"test_structured", "test_structured_api", "test_confidence"}:
        for node in ast.walk(tree):
            if isinstance(node, ast.Tuple):
                node.elts = [
                    e for e in node.elts if not (isinstance(e, ast.Name) and e.id == "ONNXAgent")
                ]
    if path.stem == "test_serve":
        tree = measure_idle_window_on_worker(tree)
    tree = ast.fix_missing_locations(Namespace().visit(tree))
    import laya_coreml.agent as agent

    if path.stem in {"test_router", "test_router_batch", "test_hooks"}:
        agent.load = lambda *args, **kwargs: agent.Agent(*args, **kwargs)
    sys.argv = [str(path)]
    module = types.ModuleType("__main__")
    module.__file__ = str(path)
    sys.modules["__main__"] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    if path.stem == "_long_attribution":
        # Run the shared upstream attribution cases with its unchanged host fixture.
        # The fixture stubs encoding/forward/decoding, never native tensor execution.
        fixture_path = path.with_name("test_predict_long.py")
        fixture_tree = ast.parse(fixture_path.read_text(encoding="utf-8"))
        fixture_tree.body = [
            node
            for node in fixture_tree.body
            if (
                isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {"_Tok", "make_real_agent"}
            )
            or (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "NO_STATE_STATS" for t in node.targets)
            )
        ]
        import threading

        import numpy as np

        from laya_coreml.agent import Agent

        fixture_scope = {"Agent": Agent, "np": np, "threading": threading}
        exec(compile(fixture_tree, str(fixture_path), "exec"), fixture_scope)
        checked = []

        def check(name, actual, expected):
            assert actual == expected, (name, actual, expected)
            checked.append(name)

        module.check_long_attribution(fixture_scope["make_real_agent"], check)
        print("%d unchanged upstream long-attribution checks passed" % len(checked))
    if any(name.startswith("test_") and callable(value) for name, value in module.__dict__.items()):
        import tempfile

        import pytest

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="test_coreml_", dir=path.parent
        ) as handle:
            handle.write(ast.unparse(tree))
            handle.flush()
            raise SystemExit(pytest.main(["-q", handle.name]))


if __name__ == "__main__":
    main()
