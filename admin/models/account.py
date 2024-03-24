from sqlalchemy import Column, Integer, String, Text, ForeignKey, \
    Boolean, DateTime
from sqlalchemy.orm import relationship
from werkzeug.security import check_password_hash, generate_password_hash

from smart_qa_app import db


class Account(db.Model):
    __tablename__ = 'account'
    id = Column(Integer, primary_key=True)
    account_name = Column(String(30))
    password = Column(String(128))
    login=Column(Boolean)
    ctime=Column(Integer)
    mtime=Column(Integer)



    def validate_password(self, pd):
        return self.password==pd