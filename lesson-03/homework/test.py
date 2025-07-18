from fastembed import TextEmbedding

model = TextEmbedding("jinaai/jina-embeddings-v2-small-en")
embeddings_gen = model.embed(["Hello world!"])  # Generatore
embeddings = list(embeddings_gen)  # Forza il consumo del generatore
print(embeddings)
