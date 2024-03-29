from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean, DateTime
from sqlalchemy.orm import relationship
from werkzeug.security import check_password_hash, generate_password_hash

from admin.models.db import db
from admin.token_helper import TokenHelper

class CrawlUrlContentTask(db.Model):
    __tablename__ = 't_raw_tab'
    id = Column(Integer, primary_key=True)  # Primary key, auto-increment by default
    url = Column(Text, nullable=False)  # URL to be crawled, cannot be null
    version = Column(String(128), nullable=False)  # Version of the crawled content, cannot be null
    content = Column(Text, nullable=False)  # The crawled content, cannot be null
    content_length = Column(Integer, nullable=False)  # Length of the crawled content, cannot be null
    doc_status = Column(Integer, nullable=False)  # Status of the document (e.g., processed, unprocessed), cannot be null
    base_url_id = Column(Integer, nullable=False)  # Assuming this is a foreign key
    mtime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)  # Last modified time, auto-updates on modification, cannot be null
    ctime = Column(DateTime, default=datetime.utcnow, nullable=False)  # Creation time, auto-set to current time, cannot be null

    # Assuming there's a relationship to another table via base_url_id
    base_url = relationship("AnotherModel", backref="crawl_tasks")
