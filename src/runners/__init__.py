REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_robust import EpisodeRobustRunner
REGISTRY["episode_robust"] = EpisodeRobustRunner
