-- Phase 13.2 Evaluation Engine - Baseline Management
-- Migration: 007_eval_baselines.sql
-- Date: 2025-11-19
-- Description: Create eval_baselines table for storing agent performance baselines

CREATE TABLE IF NOT EXISTS eval_baselines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(50) NOT NULL,
    environment VARCHAR(100) NOT NULL,
    mean_reward DOUBLE PRECISION NOT NULL,
    std_reward DOUBLE PRECISION NOT NULL,
    min_reward DOUBLE PRECISION,
    max_reward DOUBLE PRECISION,
    success_rate DOUBLE PRECISION NOT NULL,
    mean_steps DOUBLE PRECISION,
    hyperparameters JSONB NOT NULL,
    version INTEGER NOT NULL,
    rank INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT eval_baselines_agent_env_rank_unique UNIQUE (agent_type, environment, rank),
    CONSTRAINT eval_baselines_version_check CHECK (version > 0),
    CONSTRAINT eval_baselines_rank_check CHECK (rank > 0),
    CONSTRAINT eval_baselines_success_rate_check CHECK (success_rate >= 0 AND success_rate <= 1)
);

-- Indexes for efficient queries
CREATE INDEX idx_eval_baselines_agent_env ON eval_baselines(agent_type, environment);
CREATE INDEX idx_eval_baselines_rank ON eval_baselines(agent_type, environment, rank);
CREATE INDEX idx_eval_baselines_created_at ON eval_baselines(created_at DESC);

-- Trigger for updated_at timestamp
CREATE OR REPLACE FUNCTION update_eval_baselines_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER eval_baselines_updated_at
BEFORE UPDATE ON eval_baselines
FOR EACH ROW
EXECUTE FUNCTION update_eval_baselines_updated_at();

-- Table and column comments
COMMENT ON TABLE eval_baselines IS 'Performance baselines for RL agents across environments';
COMMENT ON COLUMN eval_baselines.id IS 'Unique baseline identifier';
COMMENT ON COLUMN eval_baselines.agent_type IS 'Agent type: DQN, A2C, PPO, DDPG';
COMMENT ON COLUMN eval_baselines.environment IS 'Environment ID (e.g., CartPole-v1)';
COMMENT ON COLUMN eval_baselines.mean_reward IS 'Mean reward across evaluation episodes';
COMMENT ON COLUMN eval_baselines.std_reward IS 'Standard deviation of rewards';
COMMENT ON COLUMN eval_baselines.success_rate IS 'Success rate (0.0-1.0)';
COMMENT ON COLUMN eval_baselines.hyperparameters IS 'JSONB snapshot of hyperparameters used';
COMMENT ON COLUMN eval_baselines.version IS 'Agent configuration version number';
COMMENT ON COLUMN eval_baselines.rank IS '1 = current best, 2 = previous best, etc.';
COMMENT ON COLUMN eval_baselines.created_at IS 'Baseline creation timestamp';
COMMENT ON COLUMN eval_baselines.updated_at IS 'Last update timestamp';
