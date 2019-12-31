from app import app
from flask_script import Manager, Server
app.run(debug = True,port=5000, host='0.0.0.0')

manager = Manager(app)
manager.add_command("runserver",
                    Server(host='0.0.0.0',
                           port=5000,
                           use_debugger=True))

manager.run()
