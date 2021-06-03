import threading
import time
from control import *

class FarmBotThread(object):
    """ Threading class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self):
        """ Constructor
        """
        self.handler = None

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def update_action(self, action, coords):
        time.sleep(3)
        self.handler.update(action, coords)

    def run(self):
        """ Method that runs forever """
        fb = start()
        self.handler = MyHandler()
        fb.connect(self.handler)

