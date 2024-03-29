import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker
from werkzeug.security import check_password_hash, generate_password_hash

from admin.models.db import db
from admin.token_helper import TokenHelper


class CrawlUrlsTask(db.Model):
    __tablename__ = 'crawl_urls_task'
    id = Column(Integer, primary_key=True)  # Primary key, auto-increment by default
    site_url = Column(Text, nullable=False)  # Site URL to crawl, cannot be null
    version = Column(String(128), nullable=False)  # Version of the task or content, cannot be null
    status = Column(Integer, nullable=False)  # Status of the crawl task, cannot be null
    ctime = Column(DateTime, default=datetime.datetime.utcnow,
                   nullable=False)  # Creation time, auto-set to current UTC time, cannot be null
    mtime = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow,
                   nullable=False)  # Last modified time, auto-updates on modification, cannot be null

    def finish_task(self):
        """Mark the task as finished and commit changes."""
        self.status = 2  # Assuming status '2' means 'finished'
        self.mtime = datetime.datetime.utcnow()  # Ensure the modified time is updated
        db.session.commit()  # Use db.session directly for committing the change
