# Contributing to claude-code-gateway

Thank you for your interest in contributing. This document describes how to get set up locally and submit changes.

## Prerequisites

- Python 3.12+
- Node.js 22+
- Claude Code CLI installed and available on your PATH
- Git

## Local Setup

1. Fork the repository and clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/claude-code-gateway.git
cd claude-code-gateway
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

## Running Tests

Run the full test suite with:

```bash
pytest tests/ -v
```

To run a specific test file:

```bash
pytest tests/test_gateway.py -v
```

## Pull Request Process

1. Create a new branch from `main` with a descriptive name:

```bash
git checkout -b feat/your-feature-name
```

2. Make your changes, keeping commits focused and atomic.

3. Run the test suite and confirm everything passes:

```bash
pytest tests/ -v
```

4. Push your branch and open a pull request against `main` on the upstream repository.

5. Fill in the pull request template. Describe what changed and why.

6. A maintainer will review your PR. Address any feedback, then it will be merged once approved.

## Code Style

- Follow the existing patterns in the codebase — consistency matters more than personal preference.
- Use type hints on all function signatures.
- Keep functions small and focused on a single responsibility.
- Prefer simple, readable code over clever one-liners.
- Add a brief comment for any non-obvious logic.
- Do not introduce new dependencies without discussing it in an issue first.
