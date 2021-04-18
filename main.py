from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import face_recognition
import datetime
import json
import requests
import os
import shutil
import pickle

app = FastAPI()
dataset = ""
dataset_filename = "2021-04-10_16-05-26.json"
graph_id = "QmR36gmoXm1LtAm3yKZcEcHrdRz9MH2GARtx1YCb48Vz1g"
graph_url = "https://api.thegraph.com/subgraphs/name/kleros/proof-of-humanity-mainnet"
kleros_ipfs = "https://ipfs.kleros.io"



templates = Jinja2Templates(directory="templates")
app.mount("/faces", StaticFiles(directory="faces"), name="faces")

@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# @app.get("/json")
# async def read_json_file():
#     global dataset_filename
#     return FileResponse(dataset_filename)

def update_dataset(skip):
    global dataset
    query = "{submissions(where:{registered: true}, first:1000, skip:"+skip+"){id status registered name requests{evidence{sender URI}}}}"
    r = requests.post(graph_url, json={'query': query})
    dataset = r.json()
    save_dataset_to_file()
    return dataset

def save_dataset_to_file():
    global dataset
    global dataset_filename
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{now}.json"
    with open(filename, 'w') as f:
        json.dump(dataset, f)
        dataset_filename = filename
        print(f"new dataset file called {dataset_filename}")

def read_dataset():
    global dataset
    global dataset_filename
    with open(dataset_filename, 'r') as f:
        data = json.load(f)
        return data
    return False

def scrape_submission_image(path, human):
    try:
        uri = human["requests"][0]["evidence"][0]["URI"]
        file = f"https://ipfs.kleros.io{uri}"
        res = requests.get(file)
        jres = res.json()
        with open(f"{path}/evidence.json", 'w') as f:
            json.dump(jres, f)
        
        uri = jres["fileURI"]
        file = f"https://ipfs.kleros.io{uri}"
        res = requests.get(file)
        jres = res.json()
        with open(f"{path}/registration.json", 'w') as f:
            json.dump(jres, f)

        uri = jres["photo"]
        file = f"https://ipfs.kleros.io{uri}"
        res = requests.get(file)
        with open(f"{path}/image.jpg", 'wb') as f:
            f.write(res.content)

        face_folder = os.path.join(os.path.abspath(os.getcwd()), "faces")
        id = human["id"]
        with open(f"{face_folder}/{id}.jpg", 'wb') as f:
            f.write(res.content)


    except Exception as e:
        print(e)
    return

def scrape_submission_video():
    return

def check_dirs(path):
    isdir = os.path.isdir(path) 
    if isdir:
        print("already recorded")
        return True
    else:
        print("new submission")
        os.mkdir(path)
        return False

def scrape_profile(human, save=True):
    folder = "humans/{}".format(human["id"])
    path = os.path.join(os.path.abspath(os.getcwd()), folder)
    folder_exists = check_dirs(path)
    try:
        if folder_exists:
            return True
        else:
            with open(f"{path}/data.json", 'w') as f:
                json.dump(human, f)
            scraped_image = scrape_submission_image(path, human)
            return True
    except Exception as e:
        print(e)
        return False
    
@app.get("/update/dataset/{skip}")
async def updater(skip):
    update_dataset(skip)
    data = read_dataset()
    for human in data["data"]["submissions"]:
        print("---")
        print(human)
        scrape_profile(human)
        print("---")
    return data


def encode_face(dir, image):
    print(f"Encoding Face {image}")
    try:
        comp_image = face_recognition.load_image_file(f"{dir}/{image}")
        comp_image_encoded = face_recognition.face_encodings(comp_image)[0]
        return (True, image[:-4], comp_image_encoded)
    except Exception as e:
        print(e)
        return (False, False, False)

def encode_faces():
    images = os.listdir("faces")
    encodings = []
    face_ids = []
    for image in images:
        if image.startswith('.'):
            pass
        else:
            ok, face_id, encoding = encode_face("faces", image)
            if ok:
                encodings.append(encoding)
                face_ids.append(face_id)
    return (encodings, face_ids)

def update_encodings(encodings, face_ids):
    # if file isnt already in face_ids then 
    # encode and add to encodings/faceid lists
    # and save the files 
    return False

@app.get("/update/encodings")
async def write_encodings():
    encodings, face_ids = encode_faces()
    with open("encodings", 'wb') as f:
        pickle.dump(encodings, f)
        with open("face_ids", 'wb') as f:
            pickle.dump(face_ids, f)
            return True
    return False

def read_encodings_file():
    encodings, face_ids = ([], [])
    with open("encodings", 'rb') as f:
        encodings = pickle.load(f)
        with open("face_ids", 'rb') as f:
            face_ids = pickle.load(f)
            return (encodings, face_ids)
    return False

def face_comparisons(input, encodings, face_ids):
    input_image = face_recognition.load_image_file(input)
    input_image_encoded = face_recognition.face_encodings(input_image)[0]
    matches = face_recognition.compare_faces(encodings, input_image_encoded)
    match_ids = []
    for i, match in enumerate(matches):
        if match:
            print("---")
            print(match)
            print(i)
            face_id = face_ids[i]
            print(face_id)
            match_ids.append(face_id)
            # print(f"warn: input face was found in database at {face_id}")
    # print(match_ids)
    if len(match_ids) > 0:
        return True, match_ids
    return False, False

# --------------------------------------------

def get_checking_profile(id):
    query = '{submission(id:"'+id+'"){id status registered name requests{evidence{sender URI}}}}'
    print(query)
    r = requests.post(graph_url, json={'query': query})
    profile = r.json()
    print(profile)
    return r.status_code, profile

def get_profile_image(path, human):
    try:
        print(human["requests"][0]["evidence"][0]["URI"])
        uri = human["requests"][0]["evidence"][0]["URI"]
        file = f"https://ipfs.kleros.io{uri}"
        res = requests.get(file)
        jres = res.json()
        with open(f"{path}/evidence.json", 'w') as f:
            json.dump(jres, f)
        
        uri = jres["fileURI"]
        file = f"https://ipfs.kleros.io{uri}"
        res = requests.get(file)
        jres = res.json()
        with open(f"{path}/registration.json", 'w') as f:
            json.dump(jres, f)

        uri = jres["photo"]
        file = f"https://ipfs.kleros.io{uri}"
        res = requests.get(file)
        with open(f"{path}/image.jpg", 'wb') as f:
            f.write(res.content)
        return (True, f'{path}/image.jpg')
    except Exception as e:
        return (False, e)

@app.get("/check/{address}")
async def check_profile(address):
    print("checking: "+address)
    status, profile = get_checking_profile(address)
    if status == 200:
        folder = "temp/"+address
        temp_path = os.path.join(os.path.abspath(os.getcwd()), folder)
        check_dirs(temp_path)
        gpi_res, input = get_profile_image(temp_path, profile["data"]["submission"])
        if gpi_res:
            encodings, face_ids = read_encodings_file()
            ret, matches = face_comparisons(input, encodings, face_ids)
            if ret:
                return matches
            else:
                return False
        else:
            return input

# @app.get("/image/{address}")
# async def get_image(address):
#     try:
#         f = open(f"faces/{address}.jpg")
#         return FileResponse(f"faces/{address}.jpg")
#     except IOError:
#         return False
    # return FileResponse(f"faces/{address}.jpg")




# def main():
#     updater() # uncomment to get graph data and profile images and store locally
#     # write_encodings() # uncomment to write face encodings to file

#     # encodings, face_ids = read_encodings_file() # read face encodings file
#     # input = "test.png"  # src of test image to check against
#     # face_comparisons(input, encodings, face_ids) # compare input with encodings


# if __name__ == '__main__':
#     main()