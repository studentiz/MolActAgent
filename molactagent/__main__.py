"""Allow ``python -m molactagent`` as an alternative to the ``molactagent``
console script. This is handy in environments where the ``console_scripts``
entry point isn't on ``PATH``."""

from molactagent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
