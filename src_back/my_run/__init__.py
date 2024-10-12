REGISTRY = {}

from .run_robust import run_robust
REGISTRY["robust"] = run_robust

from .run_attack import run_attack
REGISTRY["attack"] = run_attack
