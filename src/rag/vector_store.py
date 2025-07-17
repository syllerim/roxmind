import pandas as pd

from chromadb import PersistentClient
# from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def build_chroma_vector_store_from_df(
    df: pd.DataFrame,
    embedding_cols: list,
    metadata_cols: list,
    document_cols: list,
    id_cols: list,
    persist_dir: str = "chroma_store",
    collection_name: str = "hyrox_participants",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    debug_mode = True
):
    """
    Builds and saves a Chroma vector store from an existing DataFrame.
    
    - df: main dataset with all features and metadata.
    - embedding_cols: station and aggregate columns to embed.
    - metadata_cols: cols like age, gender, event_name, etc.
    - document_cols: searchable documents.
    - id_cols: unique ID per participant (e.g., 'id').
    - persist_dir: where to save the local vector store.
    - collection_name: name for the ChromaDB collection.
    - embedding_model_name: which sentence-transformers model to use.
    - debug_mode: if True, will delete existing collection before creating a new one.
    """

    # load embedding model
    model = SentenceTransformer(embedding_model_name)

    #convert embedding features to string per row
    df['embedding_input'] = df[embedding_cols].astype(str).apply(" ".join, axis=1)

    # prepare metadatas as dict per row
    df['metadata'] = df[metadata_cols].to_dict(orient='records')

    # join documents into one str per row
    df['document'] = df[document_cols].astype(str).apply(" ".join, axis=1)

    df['unique_id'] = df[id_cols].astype(str).apply("_".join, axis=1)

    # generate vector embeddings
    embeddings = model.encode(
        df['embedding_input'].tolist(),
        show_progress_bar=True,
        batch_size=1024
    )

    # initialize ChromaDB
    client = PersistentClient(path=persist_dir)

    # delete collection if in debug mode
    if debug_mode:
        try:
            client.delete_collection(collection_name)
            print(f"✅ Deleted existing collection '{collection_name}' (debug mode)")
        except Exception as e:
            print(f"⚠️ Could not delete collection (maybe it doesn't exist yet): {e}")
    
    # get or create collection
    collection = client.get_or_create_collection(collection_name)

    # insert in batches to avoid exceeding ChromaDB's max batch size
    max_batch_size = 5460
    total_rows = len(df)

    for i in range(0, total_rows, max_batch_size):
        batch_ids = df['unique_id'].tolist()[i:i+max_batch_size]
        batch_docs = df['document'].tolist()[i:i+max_batch_size]
        batch_metas = df['metadata'].tolist()[i:i+max_batch_size]
        batch_embeddings = embeddings[i:i+max_batch_size]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embeddings
        )
        print(f"✅ Added batch {i // max_batch_size + 1} ({len(batch_ids)} items)")

    print(f"✅ Vector store saved to {persist_dir} with collection '{collection_name}'")


# -------------------------------------------------------------------------------------