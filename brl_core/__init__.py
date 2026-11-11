"""Framework 2: Balanced Reinforcement Learning (BRL) Tri-Agent Curriculum Architecture."""

from brl_core.interfaces import AbstractBRLSolverAgent, WinEvaluator
from brl_core.knapsack_proj import DifferentiableKnapsackProjection, KnapsackProjectionLayer
from brl_core.tweaker import StochasticTweakerActorCritic, TweakerTransition
from brl_core.balancer import DDPGBalancer, BalancerReplayBuffer
from brl_core.orchestrator import BRLTrainingOrchestrator
from brl_core.perception import ModalitySliceSpec, PerceptionDistortionFilter

__all__ = [
    "AbstractBRLSolverAgent",
    "WinEvaluator",
    "DifferentiableKnapsackProjection",
    "KnapsackProjectionLayer",
    "StochasticTweakerActorCritic",
    "TweakerTransition",
    "DDPGBalancer",
    "BalancerReplayBuffer",
    "BRLTrainingOrchestrator",
    "ModalitySliceSpec",
    "PerceptionDistortionFilter",
]
