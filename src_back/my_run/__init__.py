REGISTRY = {}

from .run_robust import run_robust
REGISTRY["robust"] = run_robust

from .run_attack import run_attack
REGISTRY["attack"] = run_attack

from .run_robust_na import run_robust_na
REGISTRY["robust_na"] = run_robust_na

from .run_attack_na import run_attack_na
REGISTRY["attack_na"] = run_attack_na

from .eval_na import run_eval_na
REGISTRY["eval_na"] = run_eval_na
