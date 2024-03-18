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
    content_length INTEGER NOT NULL,
    doc_status INTEGER NOT NULL,
    ctime INTEGER NOT NULL,
    mtime INTEGER NOT NULL
)
''')

#`doc_status` meanings:
#  1 - 'Web page recorded'
#  2 - 'Web page crawling'
#  3 - 'Web page crawled'
#  4 - 'Web text Embedding stored in Chroma'


c.execute('''
CREATE TABLE IF NOT EXISTS t_doc_embedding_map_tab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    embedding_id_list TEXT NOT NULL,
    ctime INTEGER NOT NULL,
    mtime INTEGER NOT NULL
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

# Creating indexes after table creation
with sqlite3.connect(f'{SQLITE_DB_DIR}/{SQLITE_DB_NAME}') as conn:
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_url ON t_raw_tab (url)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_raw_tab (ctime)')
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_id ON t_doc_embedding_map_tab (doc_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_doc_embedding_map_tab (ctime)')

print("Database and tables created successfully, with indexes added.")

