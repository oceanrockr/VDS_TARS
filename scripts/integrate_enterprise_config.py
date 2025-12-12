#!/usr/bin/env python3
"""
Enterprise Config Integration Script

This script updates existing T.A.R.S. observability scripts to use the enterprise
configuration system while maintaining backward compatibility with existing CLI flags.

Usage:
    # Dry run (show changes without applying)
    python scripts/integrate_enterprise_config.py --dry-run

    # Apply changes
    python scripts/integrate_enterprise_config.py

    # Apply to specific files only
    python scripts/integrate_enterprise_config.py --files ga_kpi_collector.py

    # Rollback changes
    python scripts/integrate_enterprise_config.py --rollback
"""

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

SCRIPTS_TO_UPDATE = [
    "observability/ga_kpi_collector.py",
    "observability/stability_monitor_7day.py",
    "observability/anomaly_detector_lightweight.py",
    "observability/regression_analyzer.py",
    "scripts/generate_retrospective.py",
]

BACKUP_SUFFIX = ".pre_enterprise"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Integrate enterprise_config into observability scripts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without applying them"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to update (default: all)"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to pre-integration versions"
    )
    return parser.parse_args()


def backup_file(file_path: Path) -> Path:
    """Create backup of file before modification."""
    backup_path = file_path.with_suffix(file_path.suffix + BACKUP_SUFFIX)
    shutil.copy2(file_path, backup_path)
    print(f"  ✓ Backup created: {backup_path.name}")
    return backup_path


def rollback_file(file_path: Path) -> bool:
    """Restore file from backup."""
    backup_path = file_path.with_suffix(file_path.suffix + BACKUP_SUFFIX)
    if backup_path.exists():
        shutil.copy2(backup_path, file_path)
        backup_path.unlink()
        print(f"  ✓ Rolled back: {file_path.name}")
        return True
    else:
        print(f"  ✗ No backup found: {file_path.name}")
        return False


def add_enterprise_config_import(content: str) -> Tuple[str, bool]:
    """Add enterprise_config import if not present."""
    if "from enterprise_config import" in content:
        return content, False  # Already imported

    # Find the import section (after docstring, before other imports)
    import_pattern = r'(""".*?"""\s*\n)(import |from )'
    match = re.search(import_pattern, content, re.DOTALL)

    if match:
        # Insert after docstring
        insert_pos = match.start(2)
        new_import = "from enterprise_config import load_enterprise_config\n"
        content = content[:insert_pos] + new_import + content[insert_pos:]
        return content, True

    # Fallback: add at top after shebang/docstring
    lines = content.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('"""') and '"""' in line[3:]:
            insert_idx = i + 1
            break
        elif line.strip() == '"""' and i > 0:
            # Find closing docstring
            for j in range(i + 1, len(lines)):
                if '"""' in lines[j]:
                    insert_idx = j + 1
                    break
            break

    lines.insert(insert_idx, "from enterprise_config import load_enterprise_config")
    lines.insert(insert_idx + 1, "")
    return "\n".join(lines), True


def add_config_argument(content: str) -> Tuple[str, bool]:
    """Add --profile and --config arguments to argparse."""
    if "--profile" in content:
        return content, False  # Already added

    # Find parser.add_argument section
    parser_pattern = r'(parser\.add_argument\([^)]+\)\s*\n)(\s*return parser\.parse_args)'

    match = re.search(parser_pattern, content, re.DOTALL)
    if not match:
        return content, False

    # Insert before return statement
    insert_pos = match.start(2)
    new_args = '''    parser.add_argument(
        "--profile",
        type=str,
        default="local",
        help="Enterprise config profile (local, dev, staging, prod)"
    )
    parser.add_argument(
        "--config",
        action="append",
        help="Override config value (e.g., --config api.port=8200)"
    )
    '''

    content = content[:insert_pos] + new_args + "\n" + content[insert_pos:]
    return content, True


def add_config_loading(content: str, script_type: str) -> Tuple[str, bool]:
    """Add enterprise config loading logic."""
    if "load_enterprise_config(" in content:
        return content, False  # Already added

    # Find main() function or equivalent
    main_pattern = r'def main\(\):\s*\n(.*?)args = parse_args\(\)\s*\n'
    match = re.search(main_pattern, content, re.DOTALL)

    if not match:
        return content, False

    # Insert after args = parse_args()
    insert_pos = match.end()

    # Generate config loading code based on script type
    config_load_code = '''
    # Load enterprise configuration
    overrides = {}
    if hasattr(args, 'config') and args.config:
        for override in args.config:
            key, value = override.split('=', 1)
            overrides[key] = value

    try:
        config = load_enterprise_config(
            profile=getattr(args, 'profile', 'local'),
            overrides=overrides
        )
    except Exception as e:
        print(f"Warning: Failed to load enterprise config: {e}")
        print("Falling back to legacy configuration")
        config = None

    # Use config values with fallback to CLI args
    if config:
'''

    # Add script-specific config usage
    if script_type == "prometheus":
        config_load_code += '''        prometheus_url = config.observability.prometheus_url
        if not hasattr(args, 'prometheus_url') or not args.prometheus_url:
            args.prometheus_url = prometheus_url
'''
    elif script_type == "general":
        config_load_code += '''        # Config loaded successfully
        print(f"Using enterprise config (profile: {args.profile})")
'''

    content = content[:insert_pos] + config_load_code + "\n" + content[insert_pos:]
    return content, True


def update_script(file_path: Path, dry_run: bool = False) -> Dict[str, any]:
    """Update a single script to use enterprise_config."""
    print(f"\nProcessing: {file_path}")

    if not file_path.exists():
        print(f"  ✗ File not found")
        return {"success": False, "reason": "not_found"}

    # Read content
    content = file_path.read_text()
    original_content = content
    changes_made = []

    # Determine script type
    if "prometheus" in content.lower():
        script_type = "prometheus"
    else:
        script_type = "general"

    # Step 1: Add enterprise_config import
    content, changed = add_enterprise_config_import(content)
    if changed:
        changes_made.append("Added enterprise_config import")

    # Step 2: Add config CLI arguments
    content, changed = add_config_argument(content)
    if changed:
        changes_made.append("Added --profile and --config arguments")

    # Step 3: Add config loading logic
    content, changed = add_config_loading(content, script_type)
    if changed:
        changes_made.append("Added config loading logic")

    if not changes_made:
        print("  ℹ Already integrated or no changes needed")
        return {"success": True, "changes": []}

    # Show changes
    for change in changes_made:
        print(f"  • {change}")

    if dry_run:
        print("  ⚠ DRY RUN: Changes not applied")
        return {"success": True, "changes": changes_made, "dry_run": True}

    # Backup and write
    backup_file(file_path)
    file_path.write_text(content)
    print(f"  ✓ Updated successfully")

    return {
        "success": True,
        "changes": changes_made,
        "dry_run": False
    }


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("Enterprise Config Integration")
    print("=" * 80)

    # Determine files to process
    if args.files:
        # Filter to specified files
        files_to_process = [
            Path(f) for f in SCRIPTS_TO_UPDATE
            if any(specified in f for specified in args.files)
        ]
    else:
        files_to_process = [Path(f) for f in SCRIPTS_TO_UPDATE]

    if not files_to_process:
        print("\n✗ No files to process")
        return

    print(f"\nFiles to process: {len(files_to_process)}")
    for f in files_to_process:
        print(f"  - {f}")

    if args.dry_run:
        print("\n⚠ DRY RUN MODE: No files will be modified")

    # Rollback mode
    if args.rollback:
        print("\n" + "=" * 80)
        print("ROLLBACK MODE")
        print("=" * 80)

        success_count = 0
        for file_path in files_to_process:
            if rollback_file(file_path):
                success_count += 1

        print(f"\n✓ Rolled back {success_count}/{len(files_to_process)} files")
        return

    # Update mode
    print("\n" + "=" * 80)
    print("UPDATING FILES")
    print("=" * 80)

    results = []
    for file_path in files_to_process:
        result = update_script(file_path, dry_run=args.dry_run)
        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    success_count = sum(1 for r in results if r["success"])
    total_changes = sum(len(r.get("changes", [])) for r in results)

    print(f"\nFiles processed: {len(files_to_process)}")
    print(f"Successful:      {success_count}")
    print(f"Total changes:   {total_changes}")

    if args.dry_run:
        print("\n⚠ This was a DRY RUN. Run without --dry-run to apply changes.")
    else:
        print("\n✓ Integration complete!")
        print(f"\nBackups created with '{BACKUP_SUFFIX}' suffix")
        print("To rollback: python scripts/integrate_enterprise_config.py --rollback")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
