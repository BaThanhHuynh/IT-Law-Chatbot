import mysql.connector
from mysql.connector import pooling
from config import Config

_pool = None


def get_pool():
    """Get or create MySQL connection pool."""
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="it_law_pool",
            pool_size=5,
            pool_reset_session=True,
            **Config.DB_CONFIG,
        )
    return _pool


def get_connection():
    """Get a connection from the pool."""
    return get_pool().get_connection()


def execute_query(query, params=None, fetch=False):
    """Execute a query and optionally fetch results."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query, params or ())
        if fetch:
            result = cursor.fetchall()
            return result
        conn.commit()
        return cursor.lastrowid
    finally:
        cursor.close()
        conn.close()


def execute_many(query, data):
    """Execute a query with multiple data sets."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.executemany(query, data)
        conn.commit()
        return cursor.rowcount
    finally:
        cursor.close()
        conn.close()


def fetch_one(query, params=None):
    """Fetch a single row."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query, params or ())
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()


def fetch_all(query, params=None):
    """Fetch all rows."""
    return execute_query(query, params, fetch=True)
