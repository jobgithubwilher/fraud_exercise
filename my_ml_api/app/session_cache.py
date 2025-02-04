import pandas as pd
from typing import Dict


class SessionCache:
    def __init__(self):
        self.cache = {}

    def add_to_session(self, session_id: str, data: Dict):
        if session_id not in self.cache:
            self.cache[session_id] = []
        self.cache[session_id].append(data)

    def get_session_data(self, session_id: str) -> pd.DataFrame:
        if session_id in self.cache:
            return pd.DataFrame(self.cache[session_id])
        return pd.DataFrame(
            columns=["timestamp", "session_id", "device", "price"]
        )
