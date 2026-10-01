import argparse
import json

from sqlalchemy.orm import Session

from sentinel.agent.explainer import HttpChatClient, explain_flag
from sentinel.agent.tools import BoundTools
from sentinel.api.settings import Settings
from sentinel.store.db import make_engine
from sentinel.store.models import Flag, Transaction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag-id", type=int, required=True)
    args = parser.parse_args()
    engine = make_engine(Settings().database_url)
    with Session(engine) as session:
        flag = session.get(Flag, args.flag_id)
        if flag is None:
            parser.error("flag not found")
        transaction = session.get(Transaction, flag.transaction_id)
        result = explain_flag(BoundTools(session, flag, transaction), HttpChatClient())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
