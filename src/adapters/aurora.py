"""AWS Aurora PostgreSQL + pgvector adapter.

Supports two authentication modes:

1. **IAM authentication** (recommended for production) — uses ``boto3`` to
   generate a short-lived RDS auth token, which replaces the database password.
   The IAM role / user must have the ``rds-db:connect`` permission and the DB
   cluster must have ``enableIAMDatabaseAuthentication`` turned on.

2. **Standard password authentication** — falls back to a regular username /
   password when IAM auth is not required (e.g. non-production Aurora clusters
   or local Aurora-compatible instances).

SSL is always enforced for Aurora connections.

Environment variables
---------------------
AURORA_HOST         Aurora cluster endpoint (writer or reader)
AURORA_PORT         Port                        (default: 5432)
AURORA_USER         Database user
AURORA_PASSWORD     Database password           (IAM auth: leave empty)
AURORA_DATABASE     Database name               (default: paper_research)
AURORA_COLLECTION   pgvector collection         (default: papers)
AURORA_USE_IAM      'true' to enable IAM auth   (default: false)
AWS_REGION          AWS region for IAM token     (default: us-east-1)

Usage — IAM auth
----------------
    import os
    from langchain_openai import OpenAIEmbeddings
    from src.adapters.aurora import AuroraAdapter

    os.environ["AURORA_USE_IAM"] = "true"
    os.environ["AURORA_HOST"]    = "my-cluster.cluster-xyz.us-east-1.rds.amazonaws.com"
    os.environ["AURORA_USER"]    = "db_user"
    os.environ["AURORA_DATABASE"] = "paper_research"
    os.environ["AWS_REGION"]     = "us-east-1"

    adapter = AuroraAdapter(embeddings=OpenAIEmbeddings())
    vectorstore = adapter.vectorstore()

Usage — password auth
---------------------
    adapter = AuroraAdapter(
        embeddings=OpenAIEmbeddings(),
        host="my-cluster.cluster-xyz.us-east-1.rds.amazonaws.com",
        user="admin",
        password="secret",
    )
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import quote_plus

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy


_DEFAULTS = {
    "port": "5432",
    "database": "paper_research",
    "collection": "papers",
    "region": "us-east-1",
}


def _generate_iam_token(host: str, port: str, user: str, region: str) -> str:
    """Generate a short-lived RDS IAM auth token via boto3."""
    import boto3

    client = boto3.client("rds", region_name=region)
    token: str = client.generate_db_auth_token(
        DBHostname=host,
        Port=int(port),
        DBUsername=user,
        Region=region,
    )
    return token


def _build_aurora_connection_string(
    host: str,
    port: str,
    user: str,
    password: str,
    database: str,
) -> str:
    """Return a psycopg3 DSN with SSL required (mandatory for Aurora)."""
    encoded_pw = quote_plus(password)
    return (
        f"postgresql+psycopg://{user}:{encoded_pw}@{host}:{port}/{database}"
        "?sslmode=require"
    )


class AuroraAdapter:
    """Vector store adapter for AWS Aurora PostgreSQL + pgvector.

    Parameters
    ----------
    embeddings:
        Any LangChain ``Embeddings`` implementation.
    host / port / user / password / database:
        Connection parameters. Each falls back to the corresponding
        ``AURORA_*`` environment variable when *None*.
    use_iam:
        When *True* (or when ``AURORA_USE_IAM=true``) an IAM auth token is
        generated via boto3 and used as the password. The ``password``
        parameter / ``AURORA_PASSWORD`` env var is ignored in this mode.
    aws_region:
        AWS region used for IAM token generation. Defaults to ``AWS_REGION``
        env var, then ``us-east-1``.
    collection_name:
        Name of the pgvector collection.
    pre_delete_collection:
        Drop and recreate the collection on init. For tests only.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        use_iam: Optional[bool] = None,
        aws_region: Optional[str] = None,
        collection_name: Optional[str] = None,
        pre_delete_collection: bool = False,
    ) -> None:
        self._embeddings = embeddings

        resolved_host = host or os.environ["AURORA_HOST"]
        resolved_port = port or os.getenv("AURORA_PORT", _DEFAULTS["port"])
        resolved_user = user or os.environ["AURORA_USER"]
        resolved_db = database or os.getenv("AURORA_DATABASE", _DEFAULTS["database"])
        resolved_region = aws_region or os.getenv("AWS_REGION", _DEFAULTS["region"])

        _use_iam = use_iam
        if _use_iam is None:
            _use_iam = os.getenv("AURORA_USE_IAM", "false").lower() == "true"

        if _use_iam:
            resolved_password = _generate_iam_token(
                host=resolved_host,
                port=resolved_port,
                user=resolved_user,
                region=resolved_region,
            )
        else:
            resolved_password = password or os.getenv("AURORA_PASSWORD", "")

        self._connection_string = _build_aurora_connection_string(
            host=resolved_host,
            port=resolved_port,
            user=resolved_user,
            password=resolved_password,
            database=resolved_db,
        )
        self._collection = collection_name or os.getenv(
            "AURORA_COLLECTION", _DEFAULTS["collection"]
        )
        self._pre_delete = pre_delete_collection

        # IAM tokens expire after 15 minutes; store metadata so callers can
        # refresh the adapter before long-running jobs.
        self.use_iam = _use_iam
        self._host = resolved_host
        self._port = resolved_port
        self._user = resolved_user
        self._region = resolved_region

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

    def refresh_iam_token(self) -> None:
        """Regenerate the IAM auth token and reset the store connection.

        IAM tokens are valid for 15 minutes. Call this method before running
        long ingestion jobs or on a background schedule.
        """
        if not self.use_iam:
            return
        new_token = _generate_iam_token(
            host=self._host,
            port=self._port,
            user=self._user,
            region=self._region,
        )
        self._connection_string = _build_aurora_connection_string(
            host=self._host,
            port=self._port,
            user=self._user,
            password=new_token,
            database=self._connection_string.split("/")[-1].split("?")[0],
        )
        self._store = None  # force re-init on next access

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and insert *documents* into the Aurora vector store."""
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
