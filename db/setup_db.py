"""Creates and seeds inventory.db. The only file that defines the inventory
schema; agents/validation.py queries it but never creates tables.

Safe to call on every run: INSERT OR REPLACE means re-seeding is a no-op rather
than a UNIQUE constraint error. (The starter snippet in the brief uses a plain
INSERT, which crashes the second time you run it.)
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict

# Allows `python db/setup_db.py` to work as well as being imported from main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import INVENTORY_DB_PATH, SEED_INVENTORY  # noqa: E402

logger = logging.getLogger(__name__)


def seed(db_path: str = INVENTORY_DB_PATH,
         inventory: Dict[str, int] | None = None) -> None:
    """Create the inventory table and load seed rows. Idempotent."""
    inventory = SEED_INVENTORY if inventory is None else inventory
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS inventory ("
            "  item TEXT PRIMARY KEY,"
            "  stock INTEGER NOT NULL"
            ")"
        )
        conn.executemany(
            "INSERT OR REPLACE INTO inventory (item, stock) VALUES (?, ?)",
            sorted(inventory.items()),
        )
    logger.info("Seeded %s with %d items", db_path, len(inventory))


def ensure_db(db_path: str = INVENTORY_DB_PATH) -> bool:
    """Seed only if the DB file is absent. Returns True if it created one."""
    if Path(db_path).exists():
        return False
    seed(db_path)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seed()
