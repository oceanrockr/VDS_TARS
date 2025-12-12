#!/bin/bash
#
# Slack Integration for Phase 14.6
# Sends retrospective summary to Slack channel
#
# Usage:
#   ./slack_integration.sh [WEBHOOK_URL] [RETROSPECTIVE_PATH]
#
# Example:
#   ./slack_integration.sh https://hooks.slack.com/services/... /data/output/GA_7DAY_RETROSPECTIVE.json

set -euo pipefail

# Configuration
WEBHOOK_URL="${1:-${SLACK_WEBHOOK_URL}}"
RETRO_PATH="${2:-/data/output/GA_7DAY_RETROSPECTIVE.json}"

if [ -z "$WEBHOOK_URL" ]; then
  echo "Error: SLACK_WEBHOOK_URL not set"
  echo "Usage: $0 [WEBHOOK_URL] [RETROSPECTIVE_PATH]"
  exit 1
fi

if [ ! -f "$RETRO_PATH" ]; then
  echo "Error: Retrospective file not found: $RETRO_PATH"
  exit 1
fi

# Extract key metrics from JSON
SUCCESSES=$(jq -r '.successes | length' "$RETRO_PATH")
DEGRADATIONS=$(jq -r '.degradations | length' "$RETRO_PATH")
DRIFTS=$(jq -r '.unexpected_drifts | length' "$RETRO_PATH")
RECOMMENDATIONS=$(jq -r '.recommendations_v1_0_2 | length' "$RETRO_PATH")
COST_TREND=$(jq -r '.cost_analysis.cost_trend' "$RETRO_PATH")

# Build Slack message
MESSAGE=$(cat <<EOF
{
  "text": "🎉 *T.A.R.S. GA +7 Day Retrospective Ready!*",
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "📊 T.A.R.S. GA +7 Day Retrospective"
      }
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Phase 14.6 - 7-Day Stabilization Report*\nGenerated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
      }
    },
    {
      "type": "divider"
    },
    {
      "type": "section",
      "fields": [
        {
          "type": "mrkdwn",
          "text": "*Successes:*\n✅ $SUCCESSES achievements"
        },
        {
          "type": "mrkdwn",
          "text": "*Degradations:*\n⚠️ $DEGRADATIONS issues"
        },
        {
          "type": "mrkdwn",
          "text": "*Unexpected Drifts:*\n📊 $DRIFTS drifts detected"
        },
        {
          "type": "mrkdwn",
          "text": "*Recommendations:*\n🚀 $RECOMMENDATIONS for v1.0.2"
        }
      ]
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Cost Trend:* $COST_TREND"
      }
    },
    {
      "type": "divider"
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "📄 *Full Report:* \`/data/output/GA_7DAY_RETROSPECTIVE.md\`"
      }
    },
    {
      "type": "actions",
      "elements": [
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "View Retrospective"
          },
          "url": "https://github.com/veleron-dev/tars/tree/main/docs/final"
        }
      ]
    }
  ]
}
EOF
)

# Send to Slack
echo "Sending retrospective summary to Slack..."
RESPONSE=$(curl -s -X POST -H 'Content-type: application/json' --data "$MESSAGE" "$WEBHOOK_URL")

if [ "$RESPONSE" = "ok" ]; then
  echo "✅ Successfully sent to Slack"
else
  echo "❌ Failed to send to Slack: $RESPONSE"
  exit 1
fi
