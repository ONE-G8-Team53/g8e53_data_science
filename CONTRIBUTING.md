# Contribution and Branch Flow

## Branches
- `main`: stable releases. Merge only via approved PRs.
- `release-<X.X.X>`: integration. Merge via PR from features and later to `main.

## Naming for branches and commits (related to issues)
- Features: `feature/DATA-<XXX>
- Fixes: `fix/DATA-<XXX>
- Docs: `docs/DATA-<XXX>

## Pull Requests
- ≥ 1 review required (preferably 2).
- Respect CODEOWNERS for critical paths.
- CI must pass (required status checks).
- Keep PRs small, with description and checklist.

## Commits
- Clear, imperative messages; reference issues when applicable.
- Optional: signed commits if the repo requires them.
