REGISTRY = {}

from .run_robust import run_robust
REGISTRY["robust"] = run_robust

from .run_attack import run_attack
REGISTRY["attack"] = run_attack

from .eval_na import run_eval_na
REGISTRY["eval_na"] = run_eval_na