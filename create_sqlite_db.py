# coding=utf-8
import os
import sqlite3

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SQLITE_DB_DIR = os.getenv('SQLITE_DB_DIR', 'your_sqlite_db_directory')
SQLITE_DB_NAME = os.getenv('SQLITE_DB_NAME', 'your_sqlite_db_name')

if not os.path.exists(SQLITE_DB_DIR):
    os.makedirs(SQLITE_DB_DIR)

# Create or connect to a SQLite database
conn = sqlite3.connect(f'{SQLITE_DB_DIR}/{SQLITE_DB_NAME}')

# Cursor object
cur = conn.cursor()

cur.execute('''
CREATE TABLE IF NOT EXISTS t_domain_tab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    domain_status INTEGER NOT NULL,
    version INTEGER NOT NULL,
    ctime INTEGER NOT NULL,
    mtime INTEGER NOT NULL
)
''')

#`domain_status` meanings:
#  1 - 'Domain statistics gathering'
#  2 - 'Domain statistics gathering collected'
#  3 - 'Domain processing'
#  4 - 'Domain processed'


cur.execute('''
CREATE TABLE IF NOT EXISTS t_raw_tab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    url TEXT NOT NULL,
    content TEXT NOT NULL,
    content_length INTEGER NOT NULL,
    doc_status INTEGER NOT NULL,
    version INTEGER NOT NULL,
    ctime INTEGER NOT NULL,
    mtime INTEGER NOT NULL
)
''')

#`doc_status` meanings:
#  1 - 'Web page recorded'
#  2 - 'Web page crawling'
#  3 - 'Web page crawling completed'
#  4 - 'Web text Embedding stored in Chroma'
#  5 - 'Web page expired and needed crawled again'


cur.execute('''
CREATE TABLE IF NOT EXISTS t_doc_embedding_map_tab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    embedding_id_list TEXT NOT NULL,
    ctime INTEGER NOT NULL,
    mtime INTEGER NOT NULL
)
''')


cur.execute('''
CREATE TABLE IF NOT EXISTS t_user_qa_record_tab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    source TEXT NOT NULL,
    ctime INTEGER NOT NULL,
    mtime INTEGER NOT NULL
)
''')


cur.execute('''
CREATE TABLE IF NOT EXISTS t_user_qa_intervene_tab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    intervene_answer TEXT NOT NULL,
    source TEXT NOT NULL,
    ctime INTEGER NOT NULL,
    mtime INTEGER NOT NULL
)
''')


cur.execute('''
CREATE TABLE IF NOT EXISTS t_account_tab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    is_login INTEGER NOT NULL,
    ctime INTEGER NOT NULL,
    mtime INTEGER NOT NULL
)
''')


cur.execute('''
CREATE TABLE IF NOT EXISTS t_bot_setting_tab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    initial_messages TEXT NOT NULL,
    suggested_messages TEXT NOT NULL,
    bot_name TEXT NOT NULL,
    bot_avatar TEXT NOT NULL,
    chat_icon TEXT NOT NULL,
    placeholder TEXT NOT NULL,
    model TEXT NOT NULL,
    ctime INTEGER NOT NULL,
    mtime INTEGER NOT NULL
)
''')


# Commit the changes and close the connection
conn.commit()
conn.close()

# Creating indexes after table creation
with sqlite3.connect(f'{SQLITE_DB_DIR}/{SQLITE_DB_NAME}') as conn:
    # the index of t_raw_tab
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_url ON t_raw_tab (url)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_raw_tab (ctime)')

    # the index of t_doc_embedding_map_tab
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_id ON t_doc_embedding_map_tab (doc_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_doc_embedding_map_tab (ctime)')

    # the index of t_user_qa_record_tab
    conn.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON t_user_qa_record_tab (user_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_user_qa_record_tab (ctime)')

    # the index of t_user_qa_intervene_tab
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_query ON t_user_qa_intervene_tab (query)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_user_qa_intervene_tab (ctime)')

    # the index of t_account_tab
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_account_name ON t_account_tab (account_name)')

    # the index of t_domain_tab
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_domain ON t_domain_tab (domain)')


print("Database and tables created successfully, with indexes added.")

