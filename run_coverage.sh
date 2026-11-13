#!/usr/bin/env bash
# Run the full py-gnc test suite (including slow tests) with coverage.
# Extra args pass through to pytest, e.g. ./run_coverage.sh tests/guidance
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=python
[[ -x venv/bin/python ]] && PYTHON="$PWD/venv/bin/python"

cd py-gnc  # coverage reads its config from py-gnc/pyproject.toml
"$PYTHON" -m pytest -q -o addopts= -m "" --cov --cov-report=term --cov-report=lcov "$@"

echo "LCOV report: $PWD/lcov.info"

# lcov-style HTML (brew install lcov)
if command -v genhtml >/dev/null; then
    # coverage.py marks `def` lines hit (run at import) even for uncalled functions,
    # which lcov 2.x flags as inconsistent (listed twice = also silence the warnings)
    genhtml lcov.info --branch-coverage --quiet \
        --ignore-errors inconsistent,inconsistent,corrupt -o lcov-html
    echo "LCOV HTML:   $PWD/lcov-html/index.html"
else
    echo "genhtml not found (brew install lcov); skipping lcov-html"
fi


# ./run_coverage.sh                     # everything
# ./run_coverage.sh tests/guidance      # just one folder
# ./run_coverage.sh -k rpo              # any other pytest options also work
