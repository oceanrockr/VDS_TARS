-- Rollback migration 007
-- Phase 13.2 Evaluation Engine - Remove eval_baselines table

DROP TRIGGER IF EXISTS eval_baselines_updated_at ON eval_baselines;
DROP FUNCTION IF EXISTS update_eval_baselines_updated_at();
DROP TABLE IF EXISTS eval_baselines;
