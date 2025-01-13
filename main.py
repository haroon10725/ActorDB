import os
import cv2
import math
import torch
import logging
import numpy as np
import chromadb
import typer
from PIL import Image
from tqdm import tqdm
from rich import print
from patchify import patchify
import clip

app = typer.Typer()

class ClipIndex:
    def __init__(self, patch_size: int = 720 // 2): 
        self.patch_size = patch_size
        self.patch_shape = (self.patch_size, self.patch_size, 3)
        self.patch_step = self.patch_size // 2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info("Loading CLIP Model")
        self.clip_model, self.clip_prep = clip.load("RN50x4", self.device, jit=False)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="chromadb_store")
        self.collection = self.client.get_or_create_collection("video_embeddings")

    def index_and_store_embeddings(self, path: str, freq: float = 1.0):
        last_index = 0
        duration = video_duration(path)
        logging.info(f'Processing: {path}')
        with tqdm(total=math.ceil(duration)) as progress_bar:
            for frame, timestamp in video_frames(path):
                if timestamp - last_index > freq:
                    progress_bar.update(math.floor(timestamp) - progress_bar.n)
                    last_index = timestamp

                    patches = patchify(frame, self.patch_shape, self.patch_step).squeeze()
                    patches = patches.reshape(-1, *self.patch_shape)
                    pils = [self.clip_prep(Image.fromarray(p)) for p in patches]
                    tensor = torch.stack(pils).to(self.device)

                    with torch.no_grad():
                        frame_features = self.clip_model.encode_image(tensor)
                        frame_features /= frame_features.norm(dim=-1, keepdim=True)

                        # Generate unique IDs and store embeddings in ChromaDB
                        ids = [f"{path}{timestamp}{i}" for i in range(len(patches))]
                        self.collection.add(
                            ids=ids,
                            embeddings=frame_features.cpu().numpy().tolist(),
                            metadatas=[{'path': path, 't': timestamp}] * len(patches)
                        )

    def search(self, query: str, n: int = 6, threshold: int = 35):
        query_tensor = torch.cat([clip.tokenize(query)]).to(self.device)
        with torch.no_grad():
            query_features = self.clip_model.encode_text(query_tensor)
            query_features /= query_features.norm(dim=-1, keepdim=True)

        # Search embeddings in ChromaDB
        results = self.collection.query(
            query_embeddings=query_features.cpu().numpy().tolist(),
            n_results=n * 10
        )

        filtered_results = []
        time = 0

        # Process only the first list of distances and metadata
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        for score, meta in zip(distances, metadatas):
            if len(filtered_results) < n and score < threshold and abs(meta['t'] - time) > 0.1:
                time = meta['t']
                filtered_results.append({'score': float(score), 'path': meta['path'], 't': meta['t']})

        return filtered_results


def video_frames(path: str):
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    ret, frame = video.read()
    count = 0

    while ret:
        count += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame, count / fps
        ret, frame = video.read()


def video_duration(path: str) -> float:
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return frame_count / fps


@app.command()
def checkfile(filepath: str):
    if os.path.exists(filepath) and filepath.endswith(".mp4"):
        print("[green]Mp4 file exists")
    else:
        print("[bold red]Invalid file or format. Please upload a .mp4 file")


@app.command()
def processvideo(filepath: str, patch_size: int = 360, freq: float = 1.0):
    indexer = ClipIndex(patch_size)
    indexer.index_and_store_embeddings(filepath, freq)
    print(f"Processing completed for {filepath} with patch size {patch_size} and frequency {freq}s.")


@app.command()
def searchinvideo(query: str, n: int = 6, threshold: int = 35):
    indexer = ClipIndex()
    results = indexer.search(query, n, threshold)
    for result in results:
        print(f"Match at {result['t']}s in {result['path']} with score {result['score']}")


if __name__ == "__main__":  
    app()




#python main.py checkfile <path-to-movie>
#python main.py processvideo <path-to-movie> --patch-size 360 --freq 1.0
#python main.py searchinvideo "<query>" --n 6 --threshold 35

# C:\Users\hp1\Desktop\ActorDB-code\johnnyenglish.mp4