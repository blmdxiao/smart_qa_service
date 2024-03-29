import datetime
from sqlalchemy import Column, Integer, Text, DateTime

from admin.models.db import db


class DocEmbeddingMap(db.Model):
    __tablename__ = "t_doc_embedding_map_tab"
    id = Column(Integer, primary_key=True)  # Primary key, auto-increment by default
    doc_id = Column(Integer, nullable=False)  # Document ID, cannot be null. Consider adding ForeignKey if it references another table
    embedding_id_list = Column(Text, nullable=False)  # List of embedding IDs, stored as text, cannot be null
    ctime = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)  # Creation time, auto-set to current UTC time, cannot be null
    mtime = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)  # Last modified time, auto-updates on modification, cannot be null
