from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime

from admin.models.db import db


class BotSetting(db.Model):
    __tablename__ = 'bot_setting'
    id = Column(Integer, primary_key=True)  # Primary key, auto-increment by default
    initial_messages = Column(Text, nullable=False)  # Initial bot messages, cannot be null
    suggested_messages = Column(Text, nullable=False)  # Suggested messages for quick reply, cannot be null
    bot_name = Column(String(255), nullable=False)  # Bot's name, cannot be null
    bot_avatar = Column(String(255), nullable=False)  # URL to the bot's avatar image, cannot be null
    chat_icon = Column(String(255), nullable=False)  # URL to the chat icon image, cannot be null
    placeholder = Column(String(255), nullable=False)  # Placeholder text for the chat input, cannot be null
    model = Column(String(255), nullable=False)  # Model used by the bot, cannot be null
    ctime = Column(DateTime, default=datetime.utcnow, nullable=False)  # Creation time, auto-set to current time, cannot be null
    mtime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)  # Last modified time, auto-updates on modification, cannot be null
