import sys

from sentinel.api.settings import Settings
from sentinel.audit.chain import verify
from sentinel.store.db import make_engine


def main() -> None:
    ok, broken = verify(make_engine(Settings().database_url))
    print("ok" if ok else f"broken at entry {broken}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
