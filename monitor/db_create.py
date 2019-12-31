#!flask/bin/python
from migrate.versioning import api
from config import Config
from config import Config
from app import db
import os.path
db.create_all()
