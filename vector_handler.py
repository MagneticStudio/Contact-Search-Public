# AI Contacts Search Assistant
# Copyright (C) 2023 Alex Furmansky, Magnetic Ventures LLC
#
# This file is part of the AI Contacts Search Assistant.
#
# AI Contacts Search Assistant is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI Contacts Search Assistant is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with AI Contacts Search Assistant. If not, see <https://www.gnu.org/licenses/>.

import re
import os
import faiss
import logging
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any  # Import List and Dict from typing

# Load environment variables from .env file
load_dotenv()
client = OpenAI()

# Constants from .env
EMBEDDING_STORE_PATH = os.getenv('EMBEDDING_STORE_PATH', 'embeddings.index')
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', 1536))

def clean_string(input_string):
    if input_string is None:
        return ''
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', input_string)
    return cleaned_string

def chunk_text(text, chunk_size=512, overlap=256):
    """
    Chunk the text into overlapping segments.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping tokens between chunks.

    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def generate_embeddings(text_chunks, model="text-embedding-3-small"):
    """
    Generate embeddings for a list of text chunks using OpenAI's API.

    Args:
        text_chunks (list): A list of text chunks.
        model (str): The model to use for generating embeddings.

    Returns:
        list: A list of embeddings.
    """
    
    embeddings = []
    for chunk in text_chunks:
        try:
            response = client.embeddings.create(input=chunk, model=model)
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding for chunk: {chunk}\n{e}")
    
    return embeddings

class EmbeddingStore:
    def __init__(self, dimension=EMBEDDING_DIMENSION, index_path=EMBEDDING_STORE_PATH):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = self.index_path + '.meta'
        self.chunks_path = self.index_path + '.chunks'
        
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings, metadata, chunks):
        """
        Add embeddings to the FAISS index.

        Args:
            embeddings (list): A list of embeddings.
            metadata (list): A list of metadata associated with each embedding.
            chunks (list): A list of original text chunks.
        """
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        
        # Store metadata and chunks separately
        with open(self.metadata_path, 'a') as meta_file, open(self.chunks_path, 'a') as chunk_file:
            for meta, chunk in zip(metadata, chunks):
                meta_file.write(f"{meta}\n")
                chunk_file.write(f"{chunk}\n")
        
        # Save the updated index to disk
        self.save_index()

    def search(self, query_embedding, k=10, contact_ids=None, threshold=0.4):
        """
        Search for the top k similar embeddings, optionally filtering by contact IDs and a similarity threshold.

        Args:
            query_embedding (list): The query embedding.
            k (int): The number of top results to return.
            contact_ids (list, optional): A list of contact IDs to filter results.
            threshold (float): The minimum similarity threshold.

        Returns:
            list: A list of tuples containing indices, metadata, and original chunks of the top k similar embeddings.
            
            Sample Output:
            [
                (0, 'contact_id_1,other_metadata', 'Original chunk text 1'),
                (1, 'contact_id_2,other_metadata', 'Original chunk text 2'),
                ...
            ]
        """
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve metadata and chunks for the top k results
        with open(self.metadata_path, 'r') as meta_file, open(self.chunks_path, 'r') as chunk_file:
            metadata = [line.strip() for line in meta_file.readlines()]
            chunks = [line.strip() for line in chunk_file.readlines()]
        
        results = []
        for i in range(k):
            index = indices[0][i]
            distance = distances[0][i]
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            if similarity < threshold:
                continue  # Skip results below the threshold
            meta = metadata[index]
            chunk = chunks[index]
            contact_id = meta.split(',')[0]  # Assuming contact ID is the first element in metadata
            
            if contact_ids is None or contact_id in contact_ids:
                results.append((index, meta, chunk))
        
        return results
    
    def delete_embeddings_by_metadata_prefix(self, prefix):
        """
        Delete embeddings from the FAISS index based on a metadata prefix.

        Args:
            prefix (str): The prefix of the metadata to match for deletion.
        """
        if not os.path.exists(self.metadata_path) or not os.path.exists(self.chunks_path):
            # If the metadata or chunks file does not exist, there's nothing to delete
            logging.info("Metadata or chunks file does not exist. Skipping deletion.")
            return

        # Load existing metadata and chunks
        with open(self.metadata_path, 'r') as meta_file, open(self.chunks_path, 'r') as chunk_file:
            metadata = [line.strip() for line in meta_file.readlines()]
            chunks = [line.strip() for line in chunk_file.readlines()]

        # Find indices to delete
        indices_to_delete = [i for i, meta in enumerate(metadata) if meta.startswith(prefix)]

        if not indices_to_delete:
            logging.info(f"No embeddings found with prefix {prefix}. Skipping deletion.")
            return

        # Remove embeddings from the index
        self.index.remove_ids(np.array(indices_to_delete).astype('int64'))

        # Remove corresponding metadata and chunks
        metadata = [meta for i, meta in enumerate(metadata) if i not in indices_to_delete]
        chunks = [chunk for i, chunk in enumerate(chunks) if i not in indices_to_delete]

        # Save updated metadata and chunks
        with open(self.metadata_path, 'w') as meta_file, open(self.chunks_path, 'w') as chunk_file:
            for meta, chunk in zip(metadata, chunks):
                meta_file.write(f"{meta}\n")
                chunk_file.write(f"{chunk}\n")
        
        # Save the updated index to disk
        self.save_index()

    def save_index(self):
        """
        Save the FAISS index to disk.
        """
        faiss.write_index(self.index, self.index_path)

    def load_index(self):
        """
        Load the FAISS index from disk.
        """
        self.index = faiss.read_index(self.index_path)

