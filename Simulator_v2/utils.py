from Logger import Logger, Event
import json
import time 

def export_results(plants, logger, filename=None):
  """
  plants:   a list of Plant objects from this simulator run
  logger:   a Logger object with all collected data from this run
  filename: a string for the name of the file to save results to
  """
  if not filename:
    filename = time.strftime("%Y-%m-%d-%H%M%S")
  else:
    # Ignore any custom extensions
    filename = filename.split(".")[0]

  print(f"Exporting results to outputs/{filename}.json...")

  plant_data = []
  for plant in plants:
    plant_data.append({
      "type": plant.type,
      "radii": logger.get_data(Event.RADIUS_UPDATED, plant.id),
      "heights": logger.get_data(Event.HEIGHT_UPDATED, plant.id)
    })

  with open(f"./outputs/{filename}.json", "w+") as file:
    file.write(json.dumps(plant_data))

  print("Export complete.")

def read_results(filename):
  filename = filename.split(".")[0]
  with open(f"./outputs/{filename}.json") as file:
    data = json.load(file)
    return data
