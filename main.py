from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import pickle
import sqlite3

import face_recognition
import datetime
import json
import requests
import os
import shutil
import pickle
import asyncio
import umap

app = FastAPI()
graph_id = "QmR36gmoXm1LtAm3yKZcEcHrdRz9MH2GARtx1YCb48Vz1g"
graph_url = "https://api.thegraph.com/subgraphs/name/kleros/proof-of-humanity-mainnet"
kleros_ipfs = "https://ipfs.kleros.io"
conn = None
reducer = None

templates = Jinja2Templates(directory="templates")
app.mount("/faces", StaticFiles(directory="faces"), name="faces")


@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/check/{address}")
async def check_profile(address):
    print("checking: ", address)
    global conn
    # make address lowercase
    address = str(address.lower())
    c = conn.cursor()
    # check db for address
    res = c.execute("SELECT * FROM humans WHERE address = ?", (address,)).fetchone()
    # if not in db
    if not res:
        print("not in db, scraping")
        r, profile = get_checking_profile(address)
        if r is False:
            return {"status": "error", "message": "error getting profile"}
        ok = scrape_profile(profile)
        if not ok:
            return {"status": "error", "message": "error scraping profile"}
        res = c.execute("SELECT * FROM humans WHERE address = ?", (address,)).fetchone()
    try:
        print(res)
        similar = find_similar_encodings(c, res[2], res[3])
        return similar
    except Exception as e:
        print("error finding similar", e)
        return False

def find_similar_encodings(c, x, y, dist=0.08):
    c.execute("SELECT address FROM humans WHERE embedding_x BETWEEN ? AND ? AND embedding_y BETWEEN ? AND ?", (x-dist, x+dist, y-dist, y+dist))
    results = c.fetchall()
    return results


def get_checking_profile(address):
    query = '{submission(id:"'+address+'"){id status registered name requests{evidence{sender URI}}}}'
    print(query)
    try:
        r = requests.post(graph_url, json={'query': query})
        profile = r.json()
        profile = profile["data"]["submission"]
        return r.status_code, profile
    except Exception as e:
        print("error getting profile", e)
        return False, False


def check_dirs(path):
    isdir = os.path.isdir(path) 
    if isdir:
        print("already recorded")
        return True
    else:
        print("new submission")
        os.mkdir(path)
        return False


@app.get("/update/{limit}")
def update(limit=17000):
    id = "0"
    count = 0
    limit = int(limit)
    while(count <= limit):
        print(f"updating {id}  {count}")
        query = '{submissions(first: 1000, where: {id_gt:"'+id+'", registered: true}){id creationTime submissionTime status registered name vouchees{id} requests{evidence{sender URI}}}}'
        print(query)
        r = requests.post(graph_url, json={'query': query})
        data = r.json()
        for human in data["data"]["submissions"]:
            scrape_profile(human)
        try:
            id = str(data["data"]["submissions"][len(data["data"]["submissions"])-1]["id"])
        except:
            break
        count += 1000
    return "done"

def scrape_profile(human, save=True):
    try:
        folder = 'humans/{}'.format(human["id"])
        print(folder)
        path = os.path.join(os.path.abspath(os.getcwd()), folder)
        folder_exists = check_dirs(path)
        if folder_exists:
            return True
        else:
            with open(f"{path}/data.json", 'w') as f:
                json.dump(human, f)
            scraped_image = scrape_submission_image(path, human)
            return True
    except Exception as e:
        print("error scraping", e)
        return False

def scrape_submission_image(path, human):
    global conn
    print("getting submission image for ", human["id"])
    try:
        idd = human["id"]
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
        with open(f"{path}/{idd}.jpg", 'wb') as f:
            f.write(res.content)

        face_folder = os.path.join(os.path.abspath(os.getcwd()), "faces")
        with open(f"{face_folder}/{idd}.jpg", 'wb') as f:
            f.write(res.content)
        
        #  get face encoding
        print("got files, doing fancy things")
        ok, nm, enc = encode_face(face_folder, f"{idd}.jpg")
        try:
            enc_np = np.array(enc)
            np_text = str(enc_np.tolist())
            x, y = embed_encoding(enc_np)
        except Exception as e:
            print("error embedding", e)
        # write to database
        if ok:
            try:
                c = conn.cursor()
                c.execute("INSERT OR REPLACE INTO humans VALUES (?, ?, ?, ?)", (str(nm), np_text, float(x), float(y)))
                conn.commit()
            except Exception as e:
                print("error writing to db", e)
    except Exception as e:
        print("error getting files", e)
    return

def embed_encoding(enc_np):
    global reducer
    print("embedding face encoding")
    embedding = reducer.transform([enc_np])
    print(embedding)
    x = float(embedding[0][0])
    y = float(embedding[0][1])
    print(x, y)
    return x, y

def encode_face(dir, image):
    print(f"Encoding Face {image}")
    try:
        comp_image = face_recognition.load_image_file(f"{dir}/{image}")
        comp_image_encoded = face_recognition.face_encodings(comp_image)[0]
        return (True, image[:-4], comp_image_encoded)
    except Exception as e:
        print(e)
        return (False, False, False)


    
@app.on_event("startup")
async def on_startup():
    print("starting up")
    global conn
    global reducer
    conn = sqlite3.connect("data.db")
    with open("umap_reducer.pkl", 'rb') as f:
        reducer = pickle.load(f)
    
    
@app.on_event("shutdown")
async def on_shutdown():
    print("shutting down")
    conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)