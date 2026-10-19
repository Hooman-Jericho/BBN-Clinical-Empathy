"""Database connection, schema definition, and initialization."""

import sqlite3
from pathlib import Path
from contextlib import contextmanager

# Database file location
DB_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DB_DIR / "annotations.db"


def get_connection() -> sqlite3.Connection:
    """Get an optimized SQLite database connection with WAL mode and foreign keys enabled."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
    conn.execute("PRAGMA journal_mode = WAL")  # Fast concurrent reads and writes
    conn.execute("PRAGMA synchronous = NORMAL")  # Recommended for WAL mode
    return conn


@contextmanager
def get_db():
    """Context manager for database connections ensuring atomic transactions."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database tables and indexes."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Experts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                email TEXT,
                role TEXT DEFAULT 'annotator',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT NOT NULL UNIQUE,
                scenario TEXT,
                language TEXT DEFAULT 'en',
                date TEXT,
                source_file TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Turns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                turn_number INTEGER NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                UNIQUE(conversation_id, turn_number)
            )
        """)

        # Span annotations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS span_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL,
                expert_id INTEGER,
                span_id TEXT NOT NULL,
                text TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                label TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (turn_id) REFERENCES turns(id) ON DELETE CASCADE,
                FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE SET NULL,
                UNIQUE(turn_id, span_id)
            )
        """)

        # Relations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL,
                expert_id INTEGER,
                relation_id TEXT NOT NULL UNIQUE,
                from_span_id TEXT NOT NULL,
                to_span_id TEXT NOT NULL,
                to_turn_id INTEGER,
                relation_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (turn_id) REFERENCES turns(id) ON DELETE CASCADE,
                FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE SET NULL
            )
        """)

        # SPIKES annotations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spikes_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL,
                expert_id INTEGER,
                stage TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (turn_id) REFERENCES turns(id) ON DELETE CASCADE,
                FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE SET NULL,
                UNIQUE(turn_id, expert_id)
            )
        """)

        # AI suggestions table (for tracking agent performance and human reviews)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL,
                span_id TEXT NOT NULL,
                text TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                suggested_label TEXT NOT NULL,
                confidence REAL,
                reasoning TEXT,
                agent_type TEXT,
                model TEXT,
                status TEXT DEFAULT 'pending', -- pending, accepted, rejected, modified
                expert_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP,
                FOREIGN KEY (turn_id) REFERENCES turns(id) ON DELETE CASCADE,
                FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE SET NULL
            )
        """)

        # Indexes for optimized query execution
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_spans_turn ON span_annotations(turn_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_spans_span_id ON span_annotations(span_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_spans_expert ON span_annotations(expert_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_turn ON relations(turn_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_span_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_span_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_turn ON ai_suggestions(turn_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_status ON ai_suggestions(status)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_spikes_turn_expert ON spikes_annotations(turn_id, IFNULL(expert_id, -1))")


def close_db(conn):
    """Close a database connection safely."""
    if conn:
        try:
            conn.close()
        except Exception:
            pass


def reset_db():
    """Reset database (drop all tables and recreate)."""
    if DB_PATH.exists():
        try:
            DB_PATH.unlink()
        except Exception:
            pass
    init_db()
