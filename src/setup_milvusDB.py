from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient

def setup_milvus_collection():
    """
    Connects to Milvus and sets up a collection with a predefined schema.
    
    Returns:
        collection (milvus.Collection): The created collection.
    """
    # Connect to Milvus

    connections.connect("default", host="localhost", port="19530")

    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
    ]
    schema = CollectionSchema(fields, "Tembusu Grand chunks")

    # Create collection
    collection = Collection("tembusu_grand", schema)

    # Create indexes
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    }
    collection.create_index("embedding", index_params)

    return collection

def insert_data(collection, embeddings, chunks, sources):
    """
    Insert data into the collection.

    Args:
        collection (MilvusCollection): The collection to insert data into.
        embeddings (list): List of embeddings.
        chunks (list): List of chunks.
        sources (list): List of sources.

    Returns:
        None
    """
    entities = [
        embeddings,
        chunks,
        sources
    ]
    collection.insert(entities)

# Example usage
# collection = setup_milvus_collection()
# sources = ["document1.pdf"] * len(chunks)  # Assuming all chunks are from the same source
# insert_data(collection, embeddings, chunks, sources)