# coding=utf-8
import os
import sqlite3
from config import SQLITE_DB_DIR, SQLITE_DB_NAME


if not os.path.exists(SQLITE_DB_DIR):
    os.makedirs(SQLITE_DB_DIR)

# Create or connect to a SQLite database
conn = sqlite3.connect(f'{SQLITE_DB_DIR}/{SQLITE_DB_NAME}')

# Cursor object
c = conn.cursor()

# Execute SQL statement to create t_raw_tab table
c.execute('''
CREATE TABLE IF NOT EXISTS t_raw_tab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL,
    content_length INTEGER,
    crawl_time INTEGER
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database and tables created successfully.")

