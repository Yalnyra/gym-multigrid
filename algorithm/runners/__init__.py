REGISTRY = {}

from algorithm.runners.episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from algorithm.runners.parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
