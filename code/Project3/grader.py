import os 
import requests

import tensorflow as tf 
from PIL import Image
import numpy as np 


# base URL for the student inference server
base_url = "http://172.17.0.1:5000"


# GET ----
def make_get_request():
    rsp = requests.get(f"{base_url}/summary")
    try:
        rsp.raise_for_status()
        data = rsp.json()
        print(f"GET /summary format correct; response: {data}")
    except Exception as e:
        print(f"ERROR: GET /summary is INVALID. Non-200 status code; Status code received: {rsp.status_code}")
    

# POST -----

def get_paths():
    damage_paths = os.listdir("/data/damage")
    damage_paths = [f"/data/damage/{p}" for p in damage_paths]
    no_damage_paths = os.listdir("/data/no_damage")
    no_damage_paths = [f"/data/no_damage/{p}" for p in no_damage_paths]
    return damage_paths, no_damage_paths


def do_image_preprocessing(path):
    img = Image.open(path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_list = np.expand_dims(img_array, axis=0).tolist()
    return img_list


def make_post_request(path):
    # image = do_image_preprocessing(path)

    url = f"{base_url}/inference"
    
    # send multipart POST
    data = {"image": open(path, 'rb')}
    rsp = requests.post(url, files=data)
    # ------

    try:
        rsp.raise_for_status()
    except Exception as e:
        print(f"ERROR: POST /inference is INVALID. Non-200 status code; Status code received: {rsp.status_code}")
        return None
    return rsp 
    

def get_prediction(response, label):
    try:
        prediction = response.json()['prediction'].lower()
    except Exception as e:
        print(f"ERROR: POST /inference is INVALID. Could not parse the response; Exception: {e}")
        return None 
    if prediction == label:
        return 1
    else:
        return 0


def do_full_post_test():
    print("Starting full POST test suite...")
    total_correct = 0
    total = 0

    for p in damage_paths:
        total += 1
        response = make_post_request(p)
        if response:
            prediction = get_prediction(response, "damage")
            if prediction == 1 or prediction == 0:
                if prediction == 1:
                    print(f"POST /inference format correct for input {p} AND prediction was correct!")
                if prediction == 0:
                    print(f"POST /inference format correct for input {p} BUT prediction was not-correct!")
                    p = response.json()['prediction'].lower()
                    print(f"--> Your prediction: {p}; correct prediction: damage")

                total_correct = total_correct + prediction                 

    for p in no_damage_paths:
        total += 1
        response = make_post_request(p)
        if response:
            prediction = get_prediction(response, "no_damage")
            if prediction == 1 or prediction == 0:
                if prediction == 1:
                    print(f"POST /inference format correct for input {p} AND prediction was correct!")
                if prediction == 0:
                    print(f"POST /inference format correct for input {p} BUT prediction was not-correct!")
                    p = response.json()['prediction'].lower()
                    print(f"--> Your prediction: {p}; correct prediction: no_damage")
                    
                total_correct = total_correct + prediction  

    accuracy = float(total_correct)/float(total)

    print("Final results:")
    print(f"Total correct: {total_correct}")
    print(f"Total Inferences: {total}")
    print(f"Accuracy: {accuracy}")


print("\n\n\n**** STARTING GRADING ****\n")
make_get_request()

damage_paths, no_damage_paths = get_paths()
do_full_post_test()

