from flask import Flask
from config import Config
from flask_bootstrap import Bootstrap
from datetime import datetime
from flask_moment import Moment
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)

app.config.from_object(Config)
db = SQLAlchemy(app)

bootstrap = Bootstrap(app)
moment = Moment(app)

from app import views,models
