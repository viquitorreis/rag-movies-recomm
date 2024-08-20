import pymongo
import os
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

load_dotenv()

client = pymongo.MongoClient(os.environ.get('MONGO_URI'))
db = client.sample_mflix
collection = db.movies

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# client = InferenceClient(token=hf_token)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# The embedding function will take a string of text and return a list of floats
# This list of floats will represent the embeddings of the text
def generate_embedding(input: str) -> list[float]:
    embeddings = model.encode([input])

    return embeddings[0].tolist() # Convert NumPy array to list

for doc in collection.find({'plot':{"$exists": True}}).limit(50):
    # This will add a new field to the document called 'plot_embedding_hf'
    # The value of this field will be the embeddings of the plot
    doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
    collection.replace_one({'_id': doc['_id']}, doc)
    print(f"Added embeddings for {doc['title']}")

query = "imaginary characters from outer space at war"

results = collection.aggregate([
    (
        {
            "$vectorSearch": {
                "queryVector": generate_embedding(query),
                "path": "plot_embedding_hf",
                "numCandidates": 100,
                "limit": 4,
                "index": "PlotSemanticSearch"
            }
        }
    )
])

for document in results:
    print(f'Movie name: {document["title"]},\nPlot: {document["plot"]}\n')