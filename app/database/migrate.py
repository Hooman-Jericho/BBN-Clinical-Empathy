"""Migration script to import JSON data into SQLite database."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import init_db, DB_PATH
from database.crud import import_conversation_from_json, get_or_create_expert


def migrate_json_files(samples_dir: Path, create_default_expert: bool = True):
    """Migrate all JSON conversation files to the database.

    Args:
        samples_dir: Path to the samples directory containing JSON files
        create_default_expert: Whether to create a default expert
    """
    print(f"Initializing database at {DB_PATH}")
    init_db()

    if create_default_expert:
        expert = get_or_create_expert("Default Expert", "expert@example.com")
        print(f"Created/found default expert: {expert.name} (ID: {expert.id})")

    if not samples_dir.exists():
        print(f"Samples directory not found: {samples_dir}")
        return

    json_files = list(samples_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to import")

    imported = 0
    skipped = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            conv = import_conversation_from_json(data, source_file=str(json_file))
            print(f"  Imported: {json_file.name} -> conversation ID {conv.id}")
            imported += 1

        except json.JSONDecodeError as e:
            print(f"  Skipped (invalid JSON): {json_file.name} - {e}")
            skipped += 1
        except Exception as e:
            print(f"  Error importing {json_file.name}: {e}")
            skipped += 1

    print(f"\nMigration complete: {imported} imported, {skipped} skipped")


def main():
    """Run the migration."""
    # Get the samples directory
    app_dir = Path(__file__).parent.parent
    samples_dir = app_dir.parent / "data" / "samples"

    print("=" * 50)
    print("BBN Annotation Tool - Database Migration")
    print("=" * 50)

    migrate_json_files(samples_dir)

    print("\nDatabase ready at:", DB_PATH)


if __name__ == "__main__":
    main()
