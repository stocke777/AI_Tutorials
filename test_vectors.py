

import faiss
import numpy as np

index = faiss.IndexFlatL2(1536)
db = np.random.rand(5, 1536).astype('float32')
index.add(db)

query = np.random.rand(1, 1536).astype('float32')
distances, indices = index.search(query, k=3)
print(indices)
