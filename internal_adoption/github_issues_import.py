#!/usr/bin/env python3
"""
GitHub Issues Import for Phase 14.6

Automatically creates GitHub issues from retrospective action items.

Usage:
    python github_issues_import.py [RETROSPECTIVE_JSON] [GITHUB_TOKEN]

Environment Variables:
    GITHUB_TOKEN: GitHub personal access token
    GITHUB_REPO: Repository in format "owner/repo"

Example:
    export GITHUB_TOKEN="ghp_..."
    export GITHUB_REPO="veleron-dev/tars"
    python github_issues_import.py /data/output/GA_7DAY_RETROSPECTIVE.json
"""

import json
import os
import sys
from typing import Dict, List
import requests


def load_retrospective(path: str) -> dict:
    """Load retrospective JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_github_issue(
    token: str,
    repo: str,
    title: str,
    body: str,
    labels: List[str]
) -> Dict:
    """Create a GitHub issue."""
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body,
        "labels": labels
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


def format_issue_body(item: dict, retro_metadata: dict) -> str:
    """Format issue body with retrospective context."""
    body = f"""## Retrospective Action Item

**Generated from:** GA +7 Day Retrospective
**GA Day:** {retro_metadata.get('ga_day_timestamp', 'Unknown')}
**Report Generated:** {retro_metadata.get('generation_timestamp', 'Unknown')}

---

### Description

{item['description']}

---

### Context

- **Priority:** {item['priority']}
- **Status:** {item['status']}
- **Source:** Phase 14.6 Retrospective

---

### Acceptance Criteria

- [ ] Issue investigated and root cause identified
- [ ] Fix implemented and tested
- [ ] Metrics verify improvement
- [ ] Documentation updated

---

### Related Data

See full retrospective: `/data/output/GA_7DAY_RETROSPECTIVE.md`

---

*Automatically created by T.A.R.S. Phase 14.6 Retrospective Generator*
"""
    return body


def main():
    """Main entry point."""
    # Get configuration
    retro_path = sys.argv[1] if len(sys.argv) > 1 else "/data/output/GA_7DAY_RETROSPECTIVE.json"
    github_token = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPO", "veleron-dev/tars")

    if not github_token:
        print("Error: GITHUB_TOKEN not set", file=sys.stderr)
        print("Usage: python github_issues_import.py [RETROSPECTIVE_JSON] [GITHUB_TOKEN]")
        sys.exit(1)

    # Load retrospective
    print(f"Loading retrospective from {retro_path}...")
    try:
        retro = load_retrospective(retro_path)
    except FileNotFoundError:
        print(f"Error: Retrospective file not found: {retro_path}", file=sys.stderr)
        sys.exit(1)

    # Extract action items
    action_items = retro.get('action_items', [])
    print(f"Found {len(action_items)} action items")

    # Filter by priority (only P0 and P1 by default)
    priorities_to_import = os.environ.get("IMPORT_PRIORITIES", "P0,P1").split(",")
    filtered_items = [
        item for item in action_items
        if item['priority'] in priorities_to_import
    ]
    print(f"Importing {len(filtered_items)} items with priorities: {', '.join(priorities_to_import)}")

    # Create issues
    created_issues = []
    for item in filtered_items:
        title = f"{item['priority']}: {item['description']}"
        body = format_issue_body(item, retro)
        labels = [
            item['priority'].lower(),
            'retrospective',
            'phase-14.6',
            'ga-plus-7'
        ]

        print(f"Creating issue: {title}")
        try:
            issue = create_github_issue(github_token, github_repo, title, body, labels)
            created_issues.append(issue)
            print(f"  ✅ Created: {issue['html_url']}")
        except requests.exceptions.HTTPError as e:
            print(f"  ❌ Failed: {e}", file=sys.stderr)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Action items processed: {len(filtered_items)}")
    print(f"  Issues created: {len(created_issues)}")
    print(f"  Repository: {github_repo}")
    print(f"{'='*60}")

    # Output issue URLs
    if created_issues:
        print("\nCreated issues:")
        for issue in created_issues:
            print(f"  - {issue['html_url']}")


if __name__ == "__main__":
    main()
