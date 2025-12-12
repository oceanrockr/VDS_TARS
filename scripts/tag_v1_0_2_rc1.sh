#!/usr/bin/env bash
#
# scripts/tag_v1_0_2_rc1.sh
#
# Create and push Git tag for T.A.R.S. v1.0.2-rc1 release.
#
# This script:
# - Validates VERSION file
# - Checks for clean git status
# - Creates annotated tag v1.0.2-rc1
# - Pushes tag to remote (with confirmation)
# - Optionally creates GitHub release
#
# Exit Codes:
#   0 - Success
#   1 - Validation failed (VERSION mismatch, dirty git status)
#   2 - Git operations failed
#
# Usage:
#   ./scripts/tag_v1_0_2_rc1.sh [--push] [--github-release]

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPECTED_VERSION="1.0.2-rc1"
TAG_NAME="v${EXPECTED_VERSION}"
VERSION_FILE="$PROJECT_ROOT/VERSION"

AUTO_PUSH=0
GITHUB_RELEASE=0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Create and push Git tag for T.A.R.S. v1.0.2-rc1.

Options:
  --push              Automatically push tag to remote (no confirmation)
  --github-release    Create GitHub release (requires gh CLI)
  --help              Show this help message

Examples:
  # Create tag locally only
  ./scripts/tag_v1_0_2_rc1.sh

  # Create tag and push to remote
  ./scripts/tag_v1_0_2_rc1.sh --push

  # Create tag, push, and create GitHub release
  ./scripts/tag_v1_0_2_rc1.sh --push --github-release

EOF
}

print_error() {
    echo "ERROR: $*" >&2
}

print_info() {
    echo "INFO: $*"
}

print_success() {
    echo "SUCCESS: $*"
}

# ============================================================================
# VALIDATION
# ============================================================================

validate_version() {
    print_info "Validating VERSION file..."

    if [[ ! -f "$VERSION_FILE" ]]; then
        print_error "VERSION file not found: $VERSION_FILE"
        return 1
    fi

    local actual_version
    actual_version=$(cat "$VERSION_FILE" | tr -d '\n\r')

    if [[ "$actual_version" != "$EXPECTED_VERSION" ]]; then
        print_error "VERSION mismatch"
        print_error "  Expected: $EXPECTED_VERSION"
        print_error "  Actual:   $actual_version"
        return 1
    fi

    print_success "VERSION file valid: $actual_version"
    return 0
}

validate_git_status() {
    print_info "Checking git status..."

    # Check if in git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not a git repository"
        return 1
    fi

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        print_error "Uncommitted changes detected"
        print_error "Please commit or stash changes before tagging"
        git status --short
        return 1
    fi

    # Check for untracked files that should be committed
    local untracked_files
    untracked_files=$(git ls-files --others --exclude-standard)

    if [[ -n "$untracked_files" ]]; then
        print_error "Untracked files detected"
        print_error "Please add or ignore the following files:"
        echo "$untracked_files"
        return 1
    fi

    print_success "Git status clean"
    return 0
}

validate_tag_not_exists() {
    print_info "Checking if tag already exists..."

    if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
        print_error "Tag already exists: $TAG_NAME"
        print_error "To re-tag, first delete the existing tag:"
        print_error "  git tag -d $TAG_NAME"
        print_error "  git push origin :refs/tags/$TAG_NAME"
        return 1
    fi

    print_success "Tag does not exist: $TAG_NAME"
    return 0
}

# ============================================================================
# GIT OPERATIONS
# ============================================================================

create_tag() {
    print_info "Creating annotated tag: $TAG_NAME"

    local tag_message
    tag_message="T.A.R.S. v${EXPECTED_VERSION} - Release Candidate 1

Enterprise-grade observability, compliance, and security features.

Major Features:
- Enterprise configuration system (multi-source precedence)
- SOC 2 Type II + ISO 27001 + GDPR compliance framework
- AES-256-GCM encryption + RSA-PSS signing
- 12-endpoint REST API with JWT + API key authentication + RBAC
- 5 observability CLI tools upgraded to enterprise mode
- Comprehensive documentation (7,000+ LOC)
- Production-ready release packaging

For full release notes, see: RELEASE_NOTES_v${EXPECTED_VERSION}.md"

    if ! git tag -a "$TAG_NAME" -m "$tag_message"; then
        print_error "Failed to create tag"
        return 2
    fi

    print_success "Tag created: $TAG_NAME"

    # Show tag info
    echo ""
    print_info "Tag information:"
    git show "$TAG_NAME" --no-patch
    echo ""

    return 0
}

push_tag() {
    local push_confirmed=0

    if [[ $AUTO_PUSH -eq 1 ]]; then
        push_confirmed=1
    else
        echo ""
        echo "Tag created locally: $TAG_NAME"
        echo ""
        read -p "Push tag to remote origin? (y/N): " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            push_confirmed=1
        fi
    fi

    if [[ $push_confirmed -eq 1 ]]; then
        print_info "Pushing tag to remote..."

        if ! git push origin "$TAG_NAME"; then
            print_error "Failed to push tag"
            print_error "To push manually: git push origin $TAG_NAME"
            return 2
        fi

        print_success "Tag pushed to origin: $TAG_NAME"
    else
        print_info "Tag not pushed (use --push to auto-push)"
        print_info "To push manually: git push origin $TAG_NAME"
    fi

    return 0
}

create_github_release() {
    if [[ $GITHUB_RELEASE -ne 1 ]]; then
        return 0
    fi

    print_info "Creating GitHub release..."

    # Check if gh CLI is available
    if ! command -v gh &> /dev/null; then
        print_error "GitHub CLI (gh) not found"
        print_error "Install from: https://cli.github.com/"
        print_error "Skipping GitHub release creation"
        return 0
    fi

    # Check if authenticated
    if ! gh auth status &> /dev/null; then
        print_error "Not authenticated with GitHub CLI"
        print_error "Run: gh auth login"
        print_error "Skipping GitHub release creation"
        return 0
    fi

    local release_notes
    release_notes="See [RELEASE_NOTES_v${EXPECTED_VERSION}.md](RELEASE_NOTES_v${EXPECTED_VERSION}.md) for complete release notes."

    if ! gh release create "$TAG_NAME" \
        --title "T.A.R.S. $TAG_NAME - Release Candidate 1" \
        --notes "$release_notes" \
        --prerelease; then
        print_error "Failed to create GitHub release"
        print_error "Create manually at: https://github.com/oceanrockr/VDS_TARS/releases/new?tag=$TAG_NAME"
        return 0
    fi

    print_success "GitHub release created: $TAG_NAME"
    return 0
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --push)
                AUTO_PUSH=1
                shift
                ;;
            --github-release)
                GITHUB_RELEASE=1
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    parse_args "$@"

    echo "========================================================================"
    echo "  T.A.R.S. v1.0.2-rc1 Git Tagging"
    echo "========================================================================"
    echo ""

    # Validation
    if ! validate_version; then
        exit 1
    fi

    if ! validate_git_status; then
        exit 1
    fi

    if ! validate_tag_not_exists; then
        exit 1
    fi

    # Create tag
    if ! create_tag; then
        exit 2
    fi

    # Push tag
    if ! push_tag; then
        exit 2
    fi

    # Create GitHub release
    if ! create_github_release; then
        # Non-fatal, continue
        :
    fi

    # Success summary
    echo ""
    echo "========================================================================"
    echo "  TAGGING COMPLETE"
    echo "========================================================================"
    echo ""
    echo "Tag:     $TAG_NAME"
    echo "Version: $EXPECTED_VERSION"
    echo "Commit:  $(git rev-parse HEAD)"
    echo ""

    if [[ $AUTO_PUSH -eq 1 ]] || git ls-remote origin "refs/tags/$TAG_NAME" &> /dev/null; then
        echo "Status:  Pushed to origin"
    else
        echo "Status:  Local only (not pushed)"
        echo ""
        echo "To push: git push origin $TAG_NAME"
    fi

    echo ""
    print_success "Release tagging complete!"

    return 0
}

# Run main
main "$@"
exit $?
