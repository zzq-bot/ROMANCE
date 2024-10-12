REGISTRY = {}

from .basic_controller import BasicMAC
from .attack_controller import AttackMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["attack_mac"] = AttackMAC