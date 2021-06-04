from farmbot import Farmbot, FarmbotToken 
import requests
import json
import time
import wget
import os

# Before we begin, we must download an access token from the
# API. To avoid copy/pasting passwords, it is best to create
# an access token and then store that token securely:
def start():
    raw_token = FarmbotToken.download_token("markt@berkeley.edu",
                                        "a5s6d7f8fdsas",
                                        "https://my.farm.bot")

    # This token is then passed to the Farmbot constructor:
    fb = Farmbot(raw_token)
    return fb

# If you are just doing testing, such as local development,
# it is possible to skip token creation and login with email
# and password. This is not recommended for production devices:
# fb = Farmbot.login(email="em@i.l",
#                    password="pass",
#                    server="https://my.farm.bot")

# The next step is to call fb.connect(), but we are not ready
# to do that yet. Before we can call connect(), we must
# create a "handler" object. FarmBot control is event-based
# and the handler object is responsible for integrating all
# of those events into a custom application.
#
# At a minimum, the handler must respond to the following
# methods:
#     on_connect(self, bot: Farmbot, client: Mqtt) -> None
#     on_change(self, bot: Farmbot, state: Dict[Any, Any]) -> None
#     on_log(self, _bot: Farmbot, log: Dict[Any, Any]) -> None
#     on_error(self, _bot: Farmbot, _response: ErrorResponse) -> None
#     on_response(self, _bot: Farmbot, _response: OkResponse) -> None
#
# FarmBotPy will call the appropriate method whenever an event
# is triggered. For example, the method `on_log` will be
# called with the last log message every time a new log
# message is created.
# CLIENT = None

class MyHandler:
    # The `on_connect` event is called whenever the device
    # connects to the MQTT server. You can place initialization
    # logic here.
    def __init__(self):
        self.action = None
        self.coords = None

        self.CLIENT = None
        self.bot = None

    def update(self, action, coords):
        assert action in ['prune', 'move', 'photo', 'move_rel'], "Not in list of actions"
        self.action = action
        self.coords = coords
        self.execute()  

    def execute(self):
        if self.action == 'move':
            if isinstance(self.coords, list):
                for i in self.coords:
                    request_id = self.bot.move_absolute(x=i[0], y=i[1], z=i[2])
                    print("MOVE_ABS REQUEST ID: " + request_id)
            else:
                x, y, z = self.coords[0], self.coords[1], self.coords[2]
                request_id = self.bot.move_absolute(x=x, y=y, z=z)
                print("MOVE_ABS REQUEST ID: " + request_id)

        elif self.action == 'move_rel':
            request_id = self.bot.move_relative(self.coords[0],self.coords[1],self.coords[2])
            print("TOGGLE PIN REQUEST ID: " + request_id)

        elif self.action == 'prune':
            request_id = self.bot.toggle_pin(48)
            print("TOGGLE PIN REQUEST ID: " + request_id)
            # request_id = self.bot.move_relative(0,0,-390)
            # print("MOVE_REL REQUEST ID: " + request_id)
            # request_id = self.bot.move_relative(0,0, 389)
            # print("MOVE_REL REQUEST ID: " + request_id)
            # request_id = self.bot.toggle_pin(48)
            # print("TOGGLE PIN REQUEST ID: " + request_id)

        elif self.action == 'photo':
            request_id = self.bot.take_photo()
            print("PHOTO REQUEST ID: " + request_id)

    # The callback is passed a FarmBot instance, plus an MQTT
    # client object (see Paho MQTT docs to learn more).
    def on_connect(self, bot, mqtt_client):
        # Once the bot is connected, we can send RPC commands.
        # Every RPC command returns a unique, random request
        # ID. Later on, we can use this ID to track our commands
        # as they succeed/fail (via `on_response` / `on_error`
        # callbacks):
        self.CLIENT = mqtt_client
        self.bot = bot

    def on_change(self, bot, state):
        # The `on_change` event is most frequently triggered
        # event. It is called any time the device's internal
        # state changes. Example: Updating X/Y/Z position as
        # the device moves across the garden.
        # The bot maintains all this state in a single JSON
        # object that is broadcast over MQTT constantly.
        # It is a very large object, so we are printing it
        # only as an example.

        # print("NEW BOT STATE TREE AVAILABLE:")
        # print(state)

        # Since the state tree is very large, we offer
        # convenience helpers such as `bot.position()`,
        # which returns an (x, y, z) tuple of the device's
        # last known position:
        print("Current position: (%.2f, %.2f, %.2f)" % bot.position())

        # A less convenient method would be to access the state
        # tree directly:
        # pos = state["location_data"]["position"]
        # xyz = (pos["x"], pos["y"], pos["z"])
        # print("Same information as before: " + str(xyz))

    # The `on_log` event fires every time a new log is created.
    # The callback receives a FarmBot instance, plus a JSON
    # log object. The most useful piece of information is the
    # `message` attribute, though other attributes do exist.
    def on_log(self, bot, log):
        print("New message from FarmBot: " + log['message'])

    # When a response succeeds, the `on_response` callback
    # fires. This callback is passed a FarmBot object, as well
    # as a `response` object. The most important part of the
    # `response` is `response.id`. This `id` will match the
    # original request ID, which is useful for cross-checking
    # pending operations.
    def on_response(self, bot, response):
        print("ID of successful request: " + response.id)

    # If an RPC request fails (example: stalled motors, firmware
    # timeout, etc..), the `on_error` callback is called.
    # The callback receives a FarmBot object, plus an
    # ErrorResponse object.
    def on_error(self, bot, response):
        # Remember the unique ID that was returned when we
        # called `move_absolute()` earlier? We can cross-check
        # the ID by calling `response.id`:
        print("ID of failed request: " + response.id)
        # We can also retrieve a list of error message(s) by
        # calling response.errors:
        print("Reason(s) for failure: " + str(response.errors))


# Now that we have a handler class to use, let's create an
# instance of that handler and `connect()` it to the FarmBot:
# handler = MyHandler("move", (10,20,0))

# Once `connect` is called, execution of all other code will
# be pause until an event occurs, such as logs, errors,
# status updates, etc..
# If you need to run other code while `connect()` is running,
# consider using tools like system threads or processes.
# fb = start()
# fb.connect(handler)
# print("This line will not execute. `connect()` is a blocking call.")

def photo(dir_path):
    raw_token = FarmbotToken.download_token("markt@berkeley.edu",
                                        "a5s6d7f8fdsas",
                                        "https://my.farm.bot")
    API_TOKEN = str(json.loads(raw_token)["token"]["encoded"])
    headers = {'Authorization': 'Bearer ' + API_TOKEN,
               'content-type': "application/json"}
    response = requests.get('https://my.farmbot.io/api/images', headers=headers)
    images = response.json()
    imageurls = [images[i]['attachment_url'] for i in range(len(images))]
    imagetimes = [images[i]['created_at'] for i in range(len(images))]
    imagexs = [images[i]['meta']['x'] for i in range(len(images))]
    imageys = [images[i]['meta']['y'] for i in range(len(images))]
    imagezs = [images[i]['meta']['z'] for i in range(len(images))]

    newTime = {}
    for t in imagetimes:
        ret = t[:4] + t[5:7] + t[8:10] + t[11:13] + t[14:16] + t[17:19]
        newTime[t] = int(ret)
    mri = imagetimes.index(max(newTime)) # Most Recent index

    cname = wget.download(imageurls[mri])
    ### You can modify the new name uisng time of photo or location ###
    newname = imagetimes[mri] + '_' + str(imagexs[mri]) + '_' + str(imageys[mri]) + '_' + str(imagezs[mri]) + '.jpg'
    #newname = "recent.jpg"
    os.rename(cname, dir_path + newname)

def dismount_nozzle():
    # Position Correctly, Move into slot, Disconnect
    fb = start()
    handler = MyHandler("move", [(2600, 121, -385), (2690, 121, -385), (2690, 121, -350)])
    fb.connect(handler)

def mount_nozzle():
    # Position Correctly, Move into slot, Disconnect
    fb = start()
    handler = MyHandler("move", [(2690, 121, -350), (2690, 121, -385), (2600, 121, -385)])
    fb.connect(handler)

def dismount_yPruner():
    # Position Correctly, Move into slot, Disconnect
    fb = start()
    handler = MyHandler("move", [(2600, 221, -383), (2690, 221, -383), (2690, 221, -350)])
    fb.connect(handler)

def mount_yPruner():
    # Position Correctly, Move into slot, Disconnect
    fb = start()
    handler = MyHandler("move", [(2690, 221, -350), (2690, 221, -383), (2600, 221, -383)])
    fb.connect(handler)

def dismount_xPruner():
    # Position Correctly, Move into slot, Disconnect
    fb = start()
    handler = MyHandler("move", [(2600, 321, -383), (2690, 321, -383), (2690, 321, -350)])
    fb.connect(handler)

def mount_xPruner():
    # Position Correctly, Move into slot, Disconnect
    fb = start()
    handler = MyHandler("move", [(2690, 321, -350), (2690, 321, -383), (2600, 321, -383)])
    fb.connect(handler)
    
