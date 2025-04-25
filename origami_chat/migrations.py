from mautrix.util.async_db import Scheme, UpgradeTable

upgrade_table = UpgradeTable()


@upgrade_table.register(description="Create usage log table")  # type: ignore
async def upgrade_v1(conn, scheme: Scheme) -> None:
    if scheme == Scheme.SQLITE:
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event_ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
    elif scheme == Scheme.POSTGRES:
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS usage_log (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                event_ts TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
            )"""
        )
