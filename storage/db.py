import logging
import os
from dataclasses import dataclass

import psycopg


logger = logging.getLogger(__name__)


@dataclass
class PostgresConfig:
    """
    Connection configuration for the Postgres backing store.
    """

    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "postgres")
    dbname: str = os.getenv("PGDATABASE", "epstien_files_db")


def connect(cfg: PostgresConfig) -> psycopg.Connection:
    """
    Create a new psycopg connection from a PostgresConfig.
    """
    logger.debug(
        "Connecting to Postgres host=%s port=%s dbname=%s user=%s",
        cfg.host,
        cfg.port,
        cfg.dbname,
        cfg.user,
    )
    return psycopg.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        dbname=cfg.dbname,
    )

