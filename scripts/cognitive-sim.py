#!/usr/bin/env python3
"""
T.A.R.S. Cognitive Federation Simulator
Tests adaptive policy learning and meta-consensus optimization
"""
import argparse
import asyncio
import httpx
import logging
from typing import List, Dict, Any
from datetime import datetime
import json
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CognitiveSimulator:
    """Simulate cognitive federation operations"""

    def __init__(
        self,
        insight_engine_url: str = "http://localhost:8090",
        policy_learner_url: str = "http://localhost:8091",
        meta_optimizer_url: str = "http://localhost:8092",
        ethical_trainer_url: str = "http://localhost:8093"
    ):
        self.insight_engine_url = insight_engine_url
        self.policy_learner_url = policy_learner_url
        self.meta_optimizer_url = meta_optimizer_url
        self.ethical_trainer_url = ethical_trainer_url

        self.client = httpx.AsyncClient(timeout=30.0)

        self.metrics = {
            "insights_generated": 0,
            "proposals_created": 0,
            "consensus_optimizations": 0,
            "ethical_predictions": 0,
            "total_latency_ms": 0.0,
            "successful_adaptations": 0,
            "failed_adaptations": 0
        }

    async def check_services(self) -> bool:
        """Check if all services are healthy"""

        services = {
            "Insight Engine": f"{self.insight_engine_url}/health",
            "Policy Learner": f"{self.policy_learner_url}/health",
            "Meta Optimizer": f"{self.meta_optimizer_url}/health",
            "Ethical Trainer": f"{self.ethical_trainer_url}/health"
        }

        all_healthy = True

        for name, url in services.items():
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    logger.info(f"✓ {name} is healthy")
                else:
                    logger.error(f"✗ {name} returned status {response.status_code}")
                    all_healthy = False
            except Exception as e:
                logger.error(f"✗ {name} is unreachable: {e}")
                all_healthy = False

        return all_healthy

    async def trigger_insight_generation(self) -> List[Dict[str, Any]]:
        """Trigger manual insight generation"""

        try:
            start = datetime.utcnow()

            response = await self.client.post(
                f"{self.insight_engine_url}/api/v1/insights/trigger"
            )
            response.raise_for_status()

            latency = (datetime.utcnow() - start).total_seconds() * 1000
            self.metrics["total_latency_ms"] += latency

            data = response.json()
            count = data.get("insights_generated", 0)
            self.metrics["insights_generated"] += count

            logger.info(f"Generated {count} insights (latency: {latency:.0f}ms)")

            # Fetch insights
            fetch_response = await self.client.post(
                f"{self.insight_engine_url}/api/v1/insights/recommendations",
                json={
                    "min_confidence": 0.7,
                    "limit": 10
                }
            )
            fetch_response.raise_for_status()

            insights = fetch_response.json().get("insights", [])
            return insights

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []

    async def trigger_policy_adaptation(self) -> Dict[str, Any]:
        """Trigger policy adaptation"""

        try:
            start = datetime.utcnow()

            response = await self.client.post(
                f"{self.policy_learner_url}/api/v1/adapt/trigger"
            )
            response.raise_for_status()

            latency = (datetime.utcnow() - start).total_seconds() * 1000
            self.metrics["total_latency_ms"] += latency

            data = response.json()
            proposals = data.get("proposals_created", 0)
            self.metrics["proposals_created"] += proposals

            logger.info(f"Created {proposals} policy proposals (latency: {latency:.0f}ms)")

            return data

        except Exception as e:
            logger.error(f"Error triggering adaptation: {e}")
            return {}

    async def simulate_consensus_optimization(self) -> Dict[str, Any]:
        """Simulate consensus optimization"""

        try:
            # Simulate consensus state with some variation
            avg_latency = random.uniform(280, 450)
            success_rate = random.uniform(0.93, 0.99)
            quorum_failures = random.randint(0, 8)

            start = datetime.utcnow()

            response = await self.client.post(
                f"{self.meta_optimizer_url}/api/v1/consensus/optimize",
                json={
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": avg_latency * 1.5,
                    "success_rate": success_rate,
                    "quorum_failures": quorum_failures,
                    "total_votes": 150,
                    "algorithm": "raft",
                    "current_timeout_ms": 500
                }
            )
            response.raise_for_status()

            latency = (datetime.utcnow() - start).total_seconds() * 1000
            self.metrics["total_latency_ms"] += latency
            self.metrics["consensus_optimizations"] += 1

            data = response.json()

            logger.info(
                f"Consensus optimization: {data['action_type']} → "
                f"{data['new_value']:.0f}ms (reward: {data['reward_signal']:.3f}, latency: {latency:.0f}ms)"
            )

            return data

        except Exception as e:
            logger.error(f"Error in consensus optimization: {e}")
            return {}

    async def simulate_ethical_prediction(self) -> Dict[str, Any]:
        """Simulate ethical fairness prediction"""

        try:
            # Simulate ethical decision input
            fairness_score = random.uniform(0.65, 0.90)
            training_dist = {
                "age": random.uniform(10.0, 25.0),
                "gender": random.uniform(40.0, 60.0),
                "race": random.uniform(5.0, 30.0),
                "disability": random.uniform(2.0, 15.0),
                "religion": random.uniform(5.0, 20.0)
            }

            outcome_dist = {
                group: random.uniform(0.4, 0.8)
                for group in training_dist.keys()
            }

            start = datetime.utcnow()

            response = await self.client.post(
                f"{self.ethical_trainer_url}/api/v1/predict",
                json={
                    "fairness_score": fairness_score,
                    "training_data_distribution": training_dist,
                    "outcome_by_group": outcome_dist,
                    "sample_size": random.randint(500, 2000)
                }
            )
            response.raise_for_status()

            latency = (datetime.utcnow() - start).total_seconds() * 1000
            self.metrics["total_latency_ms"] += latency
            self.metrics["ethical_predictions"] += 1

            data = response.json()

            logger.info(
                f"Ethical prediction: {data['prediction']} "
                f"(confidence: {data['confidence']:.2%}, latency: {latency:.0f}ms)"
            )

            return data

        except Exception as e:
            logger.error(f"Error in ethical prediction: {e}")
            return {}

    async def run_simulation(self, duration_seconds: int, interval_seconds: int = 10):
        """Run full cognitive simulation"""

        logger.info("="*60)
        logger.info("T.A.R.S. Cognitive Federation Simulator")
        logger.info("="*60)

        # Check services
        logger.info("\nChecking service health...")
        if not await self.check_services():
            logger.error("Some services are unhealthy. Exiting.")
            return

        logger.info(f"\nStarting simulation (duration: {duration_seconds}s, interval: {interval_seconds}s)")

        start_time = datetime.utcnow()
        iteration = 0

        while (datetime.utcnow() - start_time).total_seconds() < duration_seconds:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")

            # 1. Generate insights
            insights = await self.trigger_insight_generation()

            # 2. Trigger policy adaptation if insights exist
            if insights:
                await self.trigger_policy_adaptation()

            # 3. Optimize consensus
            await self.simulate_consensus_optimization()

            # 4. Simulate ethical prediction
            await self.simulate_ethical_prediction()

            # Wait for next iteration
            await asyncio.sleep(interval_seconds)

        # Print final statistics
        await self.print_statistics()

    async def print_statistics(self):
        """Print simulation statistics"""

        logger.info("\n" + "="*60)
        logger.info("SIMULATION STATISTICS")
        logger.info("="*60)

        total_operations = (
            self.metrics["insights_generated"] +
            self.metrics["proposals_created"] +
            self.metrics["consensus_optimizations"] +
            self.metrics["ethical_predictions"]
        )

        avg_latency = (
            self.metrics["total_latency_ms"] / total_operations
            if total_operations > 0 else 0
        )

        print(f"\nInsights Generated:        {self.metrics['insights_generated']}")
        print(f"Policy Proposals Created:  {self.metrics['proposals_created']}")
        print(f"Consensus Optimizations:   {self.metrics['consensus_optimizations']}")
        print(f"Ethical Predictions:       {self.metrics['ethical_predictions']}")
        print(f"\nTotal Operations:          {total_operations}")
        print(f"Average Latency:           {avg_latency:.0f}ms")

        # Fetch cognitive state
        try:
            response = await self.client.get(
                f"{self.insight_engine_url}/api/v1/state"
            )
            if response.status_code == 200:
                state = response.json()
                print(f"\nCognitive State:")
                print(f"  Total Insights:          {state['total_insights_generated']}")
                print(f"  Applied:                 {state['insights_applied']}")
                print(f"  Rejected:                {state['insights_rejected']}")
                print(f"  Success Rate:            {state['policy_adaptation_success_rate']:.1%}")
                print(f"  Avg Confidence:          {state['avg_confidence_score']:.2f}")

        except Exception as e:
            logger.warning(f"Could not fetch cognitive state: {e}")

        # Fetch optimizer statistics
        try:
            response = await self.client.get(
                f"{self.meta_optimizer_url}/api/v1/consensus/statistics"
            )
            if response.status_code == 200:
                stats = response.json()
                print(f"\nMeta-Optimizer Statistics:")
                print(f"  Total Updates:           {stats['total_updates']}")
                print(f"  Recent Avg Reward:       {stats['recent_avg_reward']:.3f}")
                print(f"  Q-Table Size:            {stats['q_table_size']}")

        except Exception as e:
            logger.warning(f"Could not fetch optimizer statistics: {e}")

        logger.info("\n" + "="*60)

        # Check success criteria
        print("\nSUCCESS CRITERIA VALIDATION:")
        print("-" * 60)

        criteria = {
            "Insight Latency ≤ 5s": avg_latency <= 5000,
            "Insights Generated > 0": self.metrics["insights_generated"] > 0,
            "Proposals Created > 0": self.metrics["proposals_created"] > 0,
            "Consensus Optimizations > 0": self.metrics["consensus_optimizations"] > 0,
            "Ethical Predictions > 0": self.metrics["ethical_predictions"] > 0
        }

        passed = 0
        total = len(criteria)

        for criterion, result in criteria.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{criterion:40s} {status}")
            if result:
                passed += 1

        print("-" * 60)
        print(f"OVERALL: {passed}/{total} criteria passed ({passed/total*100:.0f}%)")

        if passed == total:
            print("\n🎉 ALL SUCCESS CRITERIA MET!")
        else:
            print(f"\n⚠️  {total - passed} criteria failed")

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


async def main():
    parser = argparse.ArgumentParser(
        description="T.A.R.S. Cognitive Federation Simulator"
    )
    parser.add_argument(
        "--insight-engine",
        default="http://localhost:8090",
        help="Insight Engine URL"
    )
    parser.add_argument(
        "--policy-learner",
        default="http://localhost:8091",
        help="Policy Learner URL"
    )
    parser.add_argument(
        "--meta-optimizer",
        default="http://localhost:8092",
        help="Meta-Optimizer URL"
    )
    parser.add_argument(
        "--ethical-trainer",
        default="http://localhost:8093",
        help="Ethical Trainer URL"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Simulation duration in seconds"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Interval between iterations in seconds"
    )

    args = parser.parse_args()

    simulator = CognitiveSimulator(
        insight_engine_url=args.insight_engine,
        policy_learner_url=args.policy_learner,
        meta_optimizer_url=args.meta_optimizer,
        ethical_trainer_url=args.ethical_trainer
    )

    try:
        await simulator.run_simulation(
            duration_seconds=args.duration,
            interval_seconds=args.interval
        )
    finally:
        await simulator.close()


if __name__ == "__main__":
    asyncio.run(main())
