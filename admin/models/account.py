from datetime import datetime

from werkzeug.security import generate_password_hash, check_password_hash

from admin.models.db import db
from admin.token_helper import TokenHelper


class Account(db.Model):
    __tablename__ = 'account'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Primary key with auto increment
    account_name = db.Column(db.String(30), nullable=False)  # Ensured non-nullability
    password_hash = db.Column(db.String(128), nullable=False)  # Stores the hash of the password
    login = db.Column(db.Boolean, default=False)  # Defaults to False
    token = db.Column(db.String)  # Removed index based on requirement
    ctime = db.Column(db.DateTime, default=datetime.utcnow)  # Uses DateTime, defaults to the current time
    mtime = db.Column(db.DateTime, default=datetime.utcnow,
                      onupdate=datetime.utcnow)  # Automatically updates on record update

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        """Sets the password hash for the account."""
        self.password_hash = generate_password_hash(password, method='sha256', salt_length=10)

    def validate_password(self, password):
        """Checks if the provided password matches the stored password hash."""
        return check_password_hash(self.password_hash, password)

    def is_login(self):
        return self.login and self.token and TokenHelper.verify_token(self.token) != 'Token expired'

