"""Local PostgreSQL + pgvector adapter.

Wraps ``langchain-postgres`` ``PGVector`` to provide a vector store backed by
a locally-running PostgreSQL instance (e.g. ``docker run -e POSTGRES_PASSWORD=...
pgvector/pgvector:pg16``).

Environment variables (all have defaults suitable for local dev)
-----------------------------------------------------------------
PG_HOST         PostgreSQL host        (default: localhost)
PG_PORT         PostgreSQL port        (default: 5432)
PG_USER         Database user          (default: postgres)
PG_PASSWORD     Database password      (default: postgres)
PG_DATABASE     Database name          (default: paper_research)
PG_COLLECTION   pgvector collection    (default: papers)

Usage
-----
    from langchain_openai import OpenAIEmbeddings
    from src.adapters.postgres import LocalPostgresAdapter

    adapter = LocalPostgresAdapter(embeddings=OpenAIEmbeddings())
    vectorstore = adapter.vectorstore()

    # add documents
    adapter.add_documents(docs)

    # similarity search
    results = adapter.similarity_search("attention mechanisms", k=4)
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy


_DEFAULTS = {
    "host": "localhost",
    "port": "5432",
    "user": "postgres",
    "password": "postgres",
    "database": "paper_research",
    "collection": "papers",
}


def _build_connection_string(
    host: str,
    port: str,
    user: str,
    password: str,
    database: str,
    *,
    ssl: bool = False,
) -> str:
    """Return a psycopg3-compatible connection string."""
    scheme = "postgresql+psycopg"
    dsn = f"{scheme}://{user}:{password}@{host}:{port}/{database}"
    if ssl:
        dsn += "?sslmode=require"
    return dsn


class LocalPostgresAdapter:
    """Vector store adapter for a local PostgreSQL + pgvector instance.

    Parameters
    ----------
    embeddings:
        Any LangChain ``Embeddings`` implementation.
    host / port / user / password / database:
        Connection parameters. When *None* the corresponding ``PG_*``
        environment variable is read; falls back to the dev defaults above.
    collection_name:
        Name of the pgvector collection (i.e. logical namespace inside the DB).
    pre_delete_collection:
        When *True* drops and recreates the collection on initialisation.
        Useful for tests; never set in production.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        collection_name: Optional[str] = None,
        pre_delete_collection: bool = False,
    ) -> None:
        self._embeddings = embeddings
        self._connection_string = _build_connection_string(
            host=host or os.getenv("PG_HOST", _DEFAULTS["host"]),
            port=port or os.getenv("PG_PORT", _DEFAULTS["port"]),
            user=user or os.getenv("PG_USER", _DEFAULTS["user"]),
            password=password or os.getenv("PG_PASSWORD", _DEFAULTS["password"]),
            database=database or os.getenv("PG_DATABASE", _DEFAULTS["database"]),
        )
        self._collection = collection_name or os.getenv(
            "PG_COLLECTION", _DEFAULTS["collection"]
        )
        self._pre_delete = pre_delete_collection
        self._store: Optional[PGVector] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def vectorstore(self) -> PGVector:
        """Return (and lazily initialise) the ``PGVector`` instance."""
        if self._store is None:
            self._store = PGVector(
                embeddings=self._embeddings,
                collection_name=self._collection,
                connection=self._connection_string,
                distance_strategy=DistanceStrategy.COSINE,
                pre_delete_collection=self._pre_delete,
            )
        return self._store

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and insert *documents* into the vector store."""
        return self.vectorstore().add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return the *k* most similar documents to *query*."""
        return self.vectorstore().similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """Return documents with their cosine-distance scores."""
        return self.vectorstore().similarity_search_with_score(query, k=k)

    def as_retriever(self, **kwargs):
        """Return a LangChain ``VectorStoreRetriever`` for use in chains."""
        return self.vectorstore().as_retriever(**kwargs)
