"""autorun.py rebuilds experiment.py as header + agent config + frozen code; that must not lose anything."""
import ast

import pytest

pytest.importorskip("openai")  # autorun.py exits at import time without it
import autorun


def test_rebuild_keeps_every_import_and_compiles():
    code = autorun.read_file(autorun.EXPERIMENT_FILE)
    header, config, frozen = autorun.split_experiment(code)
    rebuilt = header + "\n\n" + config + frozen  # what one iteration writes when the agent changes nothing

    def imports(src):
        return {ast.unparse(n) for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))}

    assert imports(rebuilt) == imports(code)
    compile(rebuilt, "experiment.py", "exec")
