"""Alembic environment.

Used two ways:
* programmatically by ``app.db.migrate.upgrade_database`` – the open connection is
  passed in ``config.attributes["connection"]``;
* from the command line for development (``alembic -c alembic.ini revision
  --autogenerate -m "..."``) – the URL comes from ``sqlalchemy.url`` or the
  application settings.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from app import models  # noqa: F401  (registers every table on Base.metadata)
from app.db.base import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from app.core.config import get_settings

    return get_settings().sqlalchemy_url


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = engine_from_config({"sqlalchemy.url": _url()}, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
