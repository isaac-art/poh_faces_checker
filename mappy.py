import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sqlite3


def prepare_db():
    print("preparing")
    # make sql database
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("CREATE TABLE humans (address TEXT PRIMARY KEY, encoding TEXT, embedding_x FLOAT, embedding_y FLOAT)")
    conn.commit()
    conn.close()
    return True

def update_db_with_encodings():
    print("updating db with addresses/encodings")
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    for e in os.listdir("individual_encodings"):
        with open(f"individual_encodings/{e}", 'rb') as f:
            dat = pickle.load(f)
            dat_np = np.array(dat)
            dat_np_text = str(dat_np.tolist())
            # filename without extension
            fn = e.split(".")[0]
            # add to db address = fn, encoding = dat ignore embedding
            c.execute("INSERT OR REPLACE INTO humans VALUES (?, ?, ?, ?)", (fn, dat_np_text, None, None))
    conn.commit()
    conn.close()

def plot_umap(embedding):
    # plot 
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.show()
    
def do_umap(encodings):
    print("umapping")
    import umap 
    # 2d umap projection and save the model
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='correlation').fit(encodings)
    with open("umap_embedding.pkl", 'wb') as f:
        pickle.dump(reducer.embedding_, f)
    with open("umap_reducer.pkl", 'wb') as f:
        pickle.dump(reducer, f)
    return reducer.embedding_

    
def makey():
    print("making umaps data")
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("SELECT * FROM humans")
    encodings = []
    addresses = []
    for row in c.fetchall():
        add = row[0]
        enc = row[1].strip('][').split(', ')
        encodings.append(enc)
        addresses.append(add)
    encodings_np = np.array(encodings)
    # scaled_encodings = StandardScaler().fit_transform(encodings_np)
    embedding = do_umap(encodings_np)
    print("updating db")
    for i, add in enumerate(addresses):
        print(add, embedding[i][0], embedding[i][1])
        x = float(embedding[i][0])
        y = float(embedding[i][1])
        c.execute("UPDATE humans SET embedding_x = ?, embedding_y = ? WHERE address = ?", (x, y, add))
    conn.commit()
    conn.close()
    
def query(address="0xfff00b17888c7088dd625eeb210f58bda0dc08d9", dist=0.08):
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    # select the entry where the address = address
    data =  c.execute("SELECT * FROM humans WHERE address = ?", (address,)).fetchall()[0]
    
    x = data[2]
    y = data[3]
    # given an address find entries where embedding_x and embedding_y are within 0.1 of the address x and y
    c.execute("SELECT address FROM humans WHERE embedding_x BETWEEN ? AND ? AND embedding_y BETWEEN ? AND ?", (x-dist, x+dist, y-dist, y+dist))
    results = c.fetchall()
    conn.close()
    for res in results:
        print(res)
    return results 

def demo_fit(address, reducer):
    import umap
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    data =  c.execute("SELECT * FROM humans WHERE address = ?", (address,)).fetchall()[0]
    enc = data[1].strip('][').split(', ')
    # fit the new data to the umap reducer
    embeddint = reducer.transform([enc])
    print(embeddint)

if __name__ == "__main__":
    # prepare_db()
    # update_db_with_encodings()
    # makey()
    # with open("umap_embedding.pkl", 'rb') as f:
    #     embedding = pickle.load(f)
    #     plot_umap(embedding)
    # with open("umap_reducer.pkl", 'rb') as f:
    #     reducer = pickle.load(f)
    # demo_fit("0xfff00b17888c7088dd625eeb210f58bda0dc08d9", reducer)
    query()