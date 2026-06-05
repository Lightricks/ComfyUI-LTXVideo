import ast
from pathlib import Path


def load_config_value_matches():
    source_path = Path(__file__).resolve().parents[1] / "text_embeddings_connectors.py"
    source = ast.parse(source_path.read_text())
    helper = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_config_value_matches"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_config_value_matches"]


def test_text_encoder_norm_type_matches_both_casings():
    matches = load_config_value_matches()

    assert matches("per_token_rms", "per_token_rms")
    assert matches("PER_TOKEN_RMS", "per_token_rms")


def test_non_string_expectations_remain_strict():
    matches = load_config_value_matches()

    assert matches(False, False)
    assert not matches("False", False)
