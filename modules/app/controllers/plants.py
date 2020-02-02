import os
from flask import request, jsonify
from app import app, mongo
import logger

ROOT_PATH = os.environ.get('ROOT_PATH')
LOG = logger.get_root_logger(__name__, filename=os.path.join(ROOT_PATH, 'output.log'))

def has_key(data, key):
    return data.get(key, None) is not None

@app.route('/plant', methods=['GET', 'POST', 'DELETE', 'PATCH'])
def plant():
    if request.method == 'GET':
        query = request.args
        data = mongo.db.plants.find_one(query)
        return jsonify(data), 200

    data = request.get_json()
    if request.method == 'POST':
        if has_key(data, 'location') and has_key(data, 'radius') and has_key(data, 'type') and has_key(data, 'partition'):
            mongo.db.plants.insert_one(data)
            return jsonify({'ok': True, 'message': 'Plant created successfully'}), 200
        else:
            return jsonify({'ok': False, 'message': 'Bad request parameters'}), 400

    if request.method == 'DELETE':
        if has_key(data, 'location'):
            db_response = mongo.db.plants.delete_one({'location': data['location']})
            if db_response.deleted_count == 1:
                response = {'ok': True, 'message': 'plant deleted'}
            else:
                response = {'ok': True, 'message': 'no plant found'}
            return jsonify(response), 200
        else:
            return jsonify({'ok': False, 'message': 'Bad request parameters'}), 400
    
    if request.method == 'PATCH':
        if data.get('query', {}) != {}:
            mongo.db.plants.update_one(data['query'], {'$set': data.get('payload', {})})
            return jsonify({'ok': True, 'message': 'plant updated'}), 200
        else:
            return jsonify({'ok': False, 'message': 'Bad request parameters'}), 400

@app.route('/zoom', methods=['GET'])
def zoom():
    x_click = float(request.args.get('x'))
    y_click = float(request.args.get('y'))

    image_width = int(os.environ.get('IMAGE_WIDTH'))
    image_height = int(os.environ.get('IMAGE_HEIGHT'))
    partitions_x = int(os.environ.get('IMAGE_PARTITION_X'))
    partitions_y = int(os.environ.get('IMAGE_PARTITION_Y'))
    partition_width = image_width / partitions_x
    partition_height = image_height / partitions_y

    x = int(x_click / partition_width)
    y = int(y_click / partition_height)
    partition = y * partitions_y + x
    
    cursor = mongo.db.plants.find({"partition": partition})
    data = [plant for plant in cursor]
    return jsonify(data), 200