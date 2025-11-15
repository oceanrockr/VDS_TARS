"""Reward aggregation module for multi-agent coordination."""

from .global_reward import (
    GlobalRewardAggregator,
    GlobalRewardConfig,
    AgentReward,
    AgentType,
)

__all__ = [
    "GlobalRewardAggregator",
    "GlobalRewardConfig",
    "AgentReward",
    "AgentType",
]
