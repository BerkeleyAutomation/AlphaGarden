from enum import Enum

class Event(Enum):
  WATER_REQUIRED = "Water required"
  WATER_ABSORBED = "Water absorbed"
  RADIUS_UPDATED = "Plant radius"
  HEIGHT_UPDATED = "Plant height"

class Logger:
  def __init__(self):
    # Map of event type to dictionary of { plant_id: [list of data points] }
    self.events = {event_type: {} for event_type in Event}

  def log(self, event_type, plant_id, data):
    if event_type in self.events:
      self.events[event_type][plant_id] = self.events[event_type].get(plant_id, []) + [data]

  def get_data(self, event_type, plant_id):
    if event_type in self.events:
      return self.events[event_type][plant_id]