import argparse

from sqlalchemy.orm import Session

from sentinel.api.auth import SCOPES, create_key
from sentinel.api.settings import Settings
from sentinel.store.db import make_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--scopes", required=True)
    create.add_argument("--name", required=True)
    args = parser.parse_args()
    scopes = set(args.scopes.split(","))
    if not scopes or not scopes <= SCOPES:
        parser.error("scopes must be a comma-separated subset of ingest,read,admin")
    with Session(make_engine(Settings().database_url)) as session:
        print(create_key(session, args.name, scopes))


if __name__ == "__main__":
    main()
