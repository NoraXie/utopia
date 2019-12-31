from functools import wraps
from flask import session,redirect,url_for

# login required
def login_required(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        if session.get('username'):
            return func(*args, **kwargs)
        else:
            return redirect(url_for('login'))
    return wrapper
