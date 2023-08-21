#!/usr/bin/env python
# -*- coding: utf-8 -*-
import base64
import csv
import copy
import argparse
import itertools
import datetime
import json
import os
import hashlib
from collections import deque

import cv2 as cv
import mediapipe as mp
import numpy as np

from model import KeyPointClassifier

from escpos.printer import Serial
from PIL import Image


# API
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
import eventlet
from flaskext.mysql import MySQL


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=int,
        default=0.5,
    )

    args = parser.parse_args()

    return args


def GestureDetect(
    images, hands, point_history, keypoint_classifier, keypoint_classifier_labels
):
    image = cv.flip(images, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation #############################################################
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    #  ####################################################################
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Hand sign classification
            try:
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                result = {
                    "hand": handedness.classification[0].label,
                    "result": keypoint_classifier_labels[hand_sign_id],
                }
                return result
            except:
                return "null"
    else:
        return "null"


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (
            temp_point_history[index][0] - base_x
        ) / image_width
        temp_point_history[index][1] = (
            temp_point_history[index][1] - base_y
        ) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def base64_to_image(base64_string):
    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv.imdecode(image_array, cv.IMREAD_COLOR)
    return image

# Custom InputStreamHandler
class MyInputStreamHandler:
    def __init__(self):
        self.data_buffer = []

    def fileno(self):
        return self

    def readinto(self, data):
        if self.data_buffer:
            data[:len(self.data_buffer)] = self.data_buffer
            del self.data_buffer[:]
            return len(data)
        return None

    def put(self, data):
        self.data_buffer.extend(data)

def loadHistory(file_path):
    history = []

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            history.append(data)

    return history

def save_to_file(data):
    with open('data/history.txt', 'a') as file:
        json.dump(data, file)
        file.write('\n')

def toCurrency(value):
    return "Rp. {:,}".format(value).replace(',', '.')

def calculate_md5(input_string):
    md5_hash = hashlib.md5(input_string.encode()).hexdigest()
    return md5_hash

def productsToObject(arr):
    objects = [
        {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "price": row[3],
            "image": row[4],
            "category_id": row[5]
        }
        for row in arr
    ]

    return objects

def generate_unique_filename(filename):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d%H%M%S')
    unique_filename = f'{timestamp}_{filename}'
    return unique_filename

# SETUP PROGRAM!!
app = Flask(__name__, static_folder='build', static_url_path='/')

UPLOAD_FOLDER = 'build/uploads'  # Folder to save uploaded images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Connecting to database
app.config['MYSQL_DATABASE_USER'] = 'mphstar'
app.config['MYSQL_DATABASE_PASSWORD'] = '123'
app.config['MYSQL_DATABASE_DB'] = 'kiosk'
app.config['MYSQL_DATABASE_HOST'] = 'localhost' 

mysql = MySQL(app)


CORS(app)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*")
input_handler = MyInputStreamHandler()

# Argument parsing #################################################################
args = get_args()

use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence

# Model load #############################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=1,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)

keypoint_classifier = KeyPointClassifier()

# Read labels ###########################################################
with open(
    "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
) as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

# Coordinate history #################################################################
history_length = 16
point_history = deque(maxlen=history_length)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/getCategory')
def get_category():
    conn = mysql.connect()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM category")
    category = cursor.fetchall()

    cursor.close()
    conn.close()

    category_list = []
    for cat in category:
        category_item = {
            'id': cat[0],
            'name': cat[1]
        }

        category_list.append(category_item)

    return jsonify({"categories": category_list})

@app.route('/api/product/delete', methods=['POST'])
def delete():
    data = request.json
    
    conn = mysql.connect()
    cursor = conn.cursor()

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], data.get('image_path'))
    if os.path.exists(file_path):
        os.remove(file_path)

    cursor.execute('DELETE FROM product WHERE id = %s', (data.get('id')))
    
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Success delete data"}), 200

@app.route('/api/product/deleteSelection', methods=['POST'])
def deleteSelection():
    data = request.json
    
    conn = mysql.connect()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM product WHERE id IN ({})".format(','.join(['%s'] * len(data.get('id')))), (data.get('id')))
    result = cursor.fetchall()

    for row in result:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], row[4])
        if os.path.exists(file_path):
            os.remove(file_path)

    cursor.execute("DELETE FROM product WHERE id IN ({})".format(','.join(['%s'] * len(data.get('id')))), (data.get('id')))
    conn.commit()

    cursor.close()
    conn.close()

    return jsonify({"message": "Success delete data"}), 200

@app.route('/api/product/update', methods=['POST'])
def update():
    try:
        conn = mysql.connect()
        cursor = conn.cursor()

        if 'image' in request.files:
            # update images
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get('filename_old').split('/')[-1])
            if os.path.exists(file_path):
                os.remove(file_path)
            
            image = request.files['image']
            name_file = generate_unique_filename(image.filename)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], name_file)
            image.save(filename)

            cursor.execute('UPDATE product SET name = %s, description = %s, price = %s, image = %s WHERE id = %s', (request.form.get('name'), request.form.get('description'), request.form.get('price'), name_file, request.form.get('id')))

        else:
            cursor.execute('UPDATE product SET name = %s, description = %s, price = %s WHERE id = %s', (request.form.get('name'), request.form.get('description'), request.form.get('price'), request.form.get('id')))
        
        
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "success"}), 200

    except Exception as e:
        print(e)
        return str(e), 500

@app.route('/api/product/create', methods=['POST'])
def create():
    try:
        image = request.files['image']
        if(image.filename != ''):
            # Save the uploaded image to the specified folder
            name_file = generate_unique_filename(image.filename)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], name_file)
            image.save(filename)
            conn = mysql.connect()
            cursor = conn.cursor()

            cursor.execute('INSERT INTO product VALUES (null, %s, %s, %s, %s, %s)', (request.form.get('name'), request.form.get('description'), request.form.get('price'), name_file, request.form.get('category')))

            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({
                "message": "Success add data"
            }), 200

    except Exception as e:
        print(e)
        return str(e), 500

@app.route('/api/login', methods=['POST'])
def cekLogin():
    try:
        data = request.form

        conn = mysql.connect()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM account WHERE userid = %s AND password = %s", (data.get('userid'), calculate_md5(data.get('password'))))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if(result):
            return jsonify({
                "status": "success",
                "message": "Login Success",
                "result": {
                    "id": result[0],
                    "userid": result[1],
                }
            })

        return jsonify({
            "status": "failed",
            "message": "ID / password is wrong"
        })

    except Exception as e:
        return jsonify({"error": str(e)})



@app.route('/api/dashboard', methods=['GET'])
def getDashboard():
    try:
        conn = mysql.connect()
        cursor = conn.cursor()

        # get count
        cursor.execute('SELECT COUNT(*) as total FROM product WHERE id_category = 1')
        totalSnack = cursor.fetchone()

        # get count
        cursor.execute('SELECT COUNT(*) as total FROM product WHERE id_category = 2')
        totalDrink = cursor.fetchone()

        # get count
        cursor.execute('SELECT COUNT(*) as total FROM product WHERE id_category = 3')
        totalIcecream = cursor.fetchone()

        # get best seller
        cursor.execute('SELECT *, COUNT(detail_history.id) as total FROM product JOIN detail_history ON product.id = detail_history.id_product  GROUP BY product.id ORDER BY total DESC')
        bestSeller = cursor.fetchall()

        # get total / category
        cursor.execute('SELECT category.id AS category, COALESCE(COUNT(detail_history.id), 0) AS total FROM category LEFT JOIN product ON product.id_category = category.id LEFT JOIN detail_history ON detail_history.id_product = product.id GROUP BY category.name;')
        category = cursor.fetchall()


        cursor.close()
        conn.close()

        return jsonify({
            "count": {
                "snack": totalSnack[0],
                "drink": totalDrink[0],
                "icecream": totalIcecream[0]
            },
            "best_seller": bestSeller,
            "category": {
                "snack": category[2][1],
                "drink": category[0][1],
                "icecream": category[1][1]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/getHistory', methods=['GET'])
def getHistory():
    try:
        page = request.args.get('page', default=1, type=int)
        limit = request.args.get('limit', default=6, type=int)
        search = request.args.get('search', default='')

        offset = (page - 1) * limit
        
        conn = mysql.connect()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) as total FROM history WHERE id LIKE %s", (f'%{search}%'))
        total = cur.fetchone()

        # Ambil data dari history
        cur.execute("SELECT * FROM history WHERE id LIKE %s ORDER BY date DESC LIMIT %s OFFSET %s", (f'%{search}%', limit, offset))
        history_data = cur.fetchall()

        combined_data = []
        
        for history in history_data:
            transaction_id = history[0]
            total_items = history[2]
            price = history[1]
            datetime = history[3]

            # Ambil detail_data berdasarkan ID transaksi
            cur.execute("SELECT * FROM detail_history JOIN product ON detail_history.id_product = product.id WHERE detail_history.id = %s", (transaction_id,))
            detail_data = cur.fetchall()

            details_list = []
            for detail in detail_data:
                item_id = detail[0]
                subtotal = detail[2]
                qty = detail[1]
                id_product = detail[3]
                name_product = detail[5]
                image = detail[8]

                details_list.append({
                    'item_id': item_id,
                    'qty': qty,
                    'subtotal': subtotal,
                    'id_product': id_product,
                    'name': name_product,
                    'image': image
                })

            combined_data.append({
                'transaction_id': transaction_id,
                'total_items': total_items,
                'price': price,
                'datetime': datetime,
                'details': details_list
            })

        cur.close()
        conn.close()

        return jsonify({
            "total": total,
            "data": combined_data
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/getProduct')
def get_data():
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=6, type=int)
    search = request.args.get('search', default='')

    offset = (page - 1) * limit

    conn = mysql.connect()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM product WHERE id_category = 1 AND name LIKE %s", (f'%{search}%'))
    total_snack = cursor.fetchone()

    if 'page' in request.args or 'limit' in request.args:
        cursor.execute('SELECT * FROM product WHERE id_category = 1 AND product.name LIKE %s ORDER BY product.id DESC LIMIT %s OFFSET %s', (f'%{search}%', limit, offset))
    else:
        cursor.execute('SELECT * FROM product WHERE id_category = 1 AND product.name LIKE %s ORDER BY product.id DESC', (f'%{search}%'))

    snack = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) as total FROM product WHERE id_category = 2 AND name LIKE %s", (f'%{search}%'))
    total_drink = cursor.fetchone()

    if 'page' in request.args or 'limit' in request.args:
        cursor.execute('SELECT * FROM product WHERE id_category = 2 AND product.name LIKE %s ORDER BY product.id DESC LIMIT %s OFFSET %s', (f'%{search}%', limit, offset))
    else:
        cursor.execute('SELECT * FROM product WHERE id_category = 2 AND product.name LIKE %s ORDER BY product.id DESC', (f'%{search}%'))
    drink = cursor.fetchall()
    

    cursor.execute("SELECT COUNT(*) as total FROM product WHERE id_category = 3 AND name LIKE %s", (f'%{search}%'))
    total_ice = cursor.fetchone()

    if 'page' in request.args or 'limit' in request.args:
        cursor.execute('SELECT * FROM product WHERE id_category = 3 AND product.name LIKE %s ORDER BY product.id DESC LIMIT %s OFFSET %s', (f'%{search}%', limit, offset))
    else:
        cursor.execute('SELECT * FROM product WHERE id_category = 3 AND product.name LIKE %s ORDER BY product.id DESC', (f'%{search}%'))
    icecream = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify({"products": {
        "snack": {
            "total_data": total_snack[0],
            "data": productsToObject(snack)
        },
        "drink": {
            "total_data": total_drink[0],
            "data": productsToObject(drink)
        },
        "icecream": {
            "total_data": total_ice[0],
            "data": productsToObject(icecream)
        }
    }})


@app.route('/transaction', methods=['POST'])
def transaction():
    data = request.json
    # save_to_file(data.get('transaction'))
    # Result data in below
    # print(data.get('transaction'))

    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.strftime('%Y%m%d%H%M%S')

    idtransaction = f'TR{timestamp}'

    transaction = data.get('transaction')

    conn = mysql.connect()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO history (id, total_price, total_items, date) VALUES (%s, %s, %s, %s)", (idtransaction, transaction['price'], transaction['total_items'], timestamp))
    conn.commit()
    
    detail_data = transaction['data']

    for detail in detail_data:
        item_id = detail['data']['id']
        subtotal = detail['subtotal']
        qty = detail['qty']

        cursor.execute("INSERT INTO detail_history (id, quantity, subtotal, id_product) VALUES (%s, %s, %s, %s)",
                        (idtransaction, qty, subtotal, item_id))
        conn.commit()
    
    cursor.close()
    conn.close()

    if 'isPrinting' in data:
        try:
            printer = Serial(devfile='/dev/ttyUSB0',
            baudrate=9600,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1.00,
            dsrdtr=True)
            
            total_items = data.get('transaction')['total_items']
            price = data.get('transaction')['price']
        

            printer.text("                Order List\n")
            printer.text(f'            {idtransaction}\n')
            printer.text("-----------------------------------------\n")
            
            for transaction_data in data.get('transaction')['data']:
                print_text = "{:<30} {:>10}\n".format(
                    transaction_data['data']['name'],
                    transaction_data.get('qty')
                )
                printer.text(print_text)
                printer.text(toCurrency(transaction_data['subtotal']))
                printer.text("\n\n")

            printer.text("-----------------------------------------\n")
            printer.text("Total Items: {}\n".format(total_items))
            printer.text("Price: {}\n".format(toCurrency(price)))

            printer.cut()
            printer.close()
            
            response = {'message': 'Transaction success'}
            return jsonify(response), 200
        
        except Exception as e:
            print(e)
            error_response = {'message': 'Transaction printing failed'}
            return jsonify(error_response), 500
    else:
        response = {'message': 'Transaction success'}
        return jsonify(response), 200

@app.route('/history')
def history():
    return jsonify({'history': loadHistory('data/history.txt')})

@socketio.on("tesconnect")
def connect(text):
    print(text)
    # emit("processed_image", {"result": "success"})

# Function socket
@socketio.on("image")
def receive_image(image):
    # Decode the base64-encoded image data
    image = base64_to_image(image)

    input_handler.put(image)

    result = GestureDetect(
        images=image,
        hands=hands,
        point_history=point_history,
        keypoint_classifier=keypoint_classifier,
        keypoint_classifier_labels=keypoint_classifier_labels,
    )

    if result != 'null':
        # end gesture recognition
        print(result)
        emit("processed_image", {"result": result})


if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
