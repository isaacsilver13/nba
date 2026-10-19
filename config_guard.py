"""Validate generated configuration without evaluating Python or importing models."""
import ast
import math

FIELDS = {
    "TRAIN_SIZE", "TEST_SIZE", "EV_THRESHOLD", "FRACTIONAL_KELLY", "MAX_KELLY",
    "FOLD_WORKERS", "TOP_N_BETS_PER_DAY", "DAILY_MAX_EXPOSURE", "MODEL_SPECS",
    "ENSEMBLE_WEIGHTS", "EXTRA_FEATURE_EXCLUSIONS", "EXTRA_FEATURE_INCLUSIONS",
    "PYTH_EXPONENT", "FF_WEIGHTS",
}
CONSTRUCTORS = {"lgb.LGBMRegressor", "ElasticNet", "XGBRegressor",
                "RandomForestRegressor", "GradientBoostingRegressor"}
PARAMETERS = {
    "n_estimators", "learning_rate", "max_depth", "num_leaves", "min_child_samples",
    "subsample", "colsample_bytree", "random_state", "verbose", "verbosity", "n_jobs",
    "deterministic", "force_row_wise", "alpha", "l1_ratio", "max_iter", "tol",
    "selection", "min_samples_leaf", "min_samples_split", "max_features",
    "reg_alpha", "reg_lambda", "gamma", "min_child_weight",
}


def _literal(node):
    # literal_eval accepts no names, calls, attributes or comprehensions.
    if any(isinstance(n, (ast.Call, ast.Attribute, ast.Name, ast.Set, ast.BinOp))
           for n in ast.walk(node)):
        raise ValueError("Configuration values must be literals")
    value = ast.literal_eval(node)
    if isinstance(value, (str, bytes)) and len(value) > 1000:
        raise ValueError("Configuration literal is too long")
    return value


def _number(name, value, low, high, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{name} must be {'an integer' if integer else 'numeric'} in {low}..{high}")


def _model(node):
    if not isinstance(node, ast.Call) or node.args:
        raise ValueError("Models require allowlisted constructors with literal keyword arguments")
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
          and node.func.value.id == "lgb"):
        name = "lgb." + node.func.attr
    else:
        raise ValueError("Unsupported model constructor")
    if name not in CONSTRUCTORS:
        raise ValueError("Unsupported model constructor")
    params = {}
    for keyword in node.keywords:
        if keyword.arg not in PARAMETERS or keyword.arg in params:
            raise ValueError("Unsupported or repeated model parameter")
        params[keyword.arg] = _literal(keyword.value)
    if name in {"lgb.LGBMRegressor", "XGBRegressor", "RandomForestRegressor"} and "n_jobs" not in params:
        raise ValueError("Parallel models require an explicit bounded n_jobs")
    for key, bounds in {"n_estimators": (1, 2000), "num_leaves": (2, 256),
                        "max_depth": (1, 32), "max_iter": (1, 50000),
                        "n_jobs": (1, 8)}.items():
        if key in params:
            _number(key, params[key], *bounds, integer=True)


def _models(node):
    if not isinstance(node, (ast.List, ast.Tuple)) or not 1 <= len(node.elts) <= 6:
        raise ValueError("MODEL_SPECS must contain 1..6 model definitions")
    ids = set()
    for spec in node.elts:
        if not isinstance(spec, ast.Dict):
            raise ValueError("Each model specification must be a literal dictionary")
        keys = [_literal(key) for key in spec.keys]
        if len(keys) != len(set(keys)) or set(keys) != {"id", "model", "needs_imputer"}:
            raise ValueError("Model requires unique id/model/needs_imputer fields")
        values = dict(zip(keys, spec.values))
        identifier = _literal(values["id"])
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("Model ids must be nonempty unique strings")
        ids.add(identifier)
        if not isinstance(_literal(values["needs_imputer"]), bool):
            raise ValueError("needs_imputer must be a boolean")
        _model(values["model"])


def validate_config(source):
    """Reject executable values, target overrides and unbounded resource knobs."""
    if len(source) > 40000:
        raise ValueError("Configuration is too large")
    tree = ast.parse(source)
    values = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
            if ast.unparse(statement.annotation) != "list[str]":
                raise ValueError("Only list[str] annotations are supported")
        else:
            raise ValueError("Only configuration assignments are allowed")
        if not isinstance(target, ast.Name) or target.id not in FIELDS or target.id in values:
            raise ValueError("Unknown, repeated or non-name configuration target")
        if target.id == "MODEL_SPECS":
            _models(value)
            values[target.id] = True
        else:
            values[target.id] = _literal(value)
    if set(values) != FIELDS:
        raise ValueError("Missing configuration variables: " + ", ".join(sorted(FIELDS - set(values))))
    for key, low, high, integer in (
        ("TRAIN_SIZE", 1, 50000, True), ("TEST_SIZE", 1, 10000, True),
        ("FOLD_WORKERS", 1, 8, True), ("TOP_N_BETS_PER_DAY", 1, 50, True),
        ("EV_THRESHOLD", 0, 1, False), ("FRACTIONAL_KELLY", 0, 1, False),
        ("MAX_KELLY", 0, 1, False), ("DAILY_MAX_EXPOSURE", 0, 1, False),
        ("PYTH_EXPONENT", 1, 30, False),
    ):
        _number(key, values[key], low, high, integer)
    if values["ENSEMBLE_WEIGHTS"] not in ("inverse_rmse", "equal"):
        raise ValueError("Unsupported ensemble weights")
    for field in ("EXTRA_FEATURE_EXCLUSIONS", "EXTRA_FEATURE_INCLUSIONS"):
        if not isinstance(values[field], list) or not all(isinstance(x, str) for x in values[field]):
            raise ValueError("Feature configuration must be a list of strings")
    weights = values["FF_WEIGHTS"]
    if not isinstance(weights, dict) or set(weights) != {"efg", "tov", "oreb", "ftr"}:
        raise ValueError("FF_WEIGHTS requires the four factor names")
    for weight in weights.values():
        _number("factor weight", weight, 0, 1)
