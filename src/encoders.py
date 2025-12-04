"""
ReACT-Drug: Protein and Molecular Encoders
ESM-2 for protein sequences, ChemBERTa for SMILES
"""

import torch
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, EsmModel, EsmTokenizer

from .config import CONFIG


class ESM2ProteinEncoder:
    """ESM-2 protein sequence encoder for similarity search"""
    
    def __init__(self):
        self.model_name = CONFIG["esm2"]["model_name"]
        self.max_length = CONFIG["esm2"]["max_length"]
        self.embedding_dim = CONFIG["esm2"]["embedding_dim"]
        self.batch_size = CONFIG["esm2"]["batch_size"]
        self.device = CONFIG["esm2"]["device"]
        
        print(f"🧬 Loading ESM-2 model: {self.model_name}")
        self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
        self.model = EsmModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("✅ ESM-2 loaded successfully")
        
        self.embedding_cache = {} if CONFIG["esm2"]["use_cached_embeddings"] else None
    
    def encode_sequences(self, sequences):
        """Encode protein sequences using ESM-2"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # Check cache
        if self.embedding_cache is not None:
            cached, uncached_seqs, uncached_idx = [], [], []
            for i, seq in enumerate(sequences):
                seq_hash = hashlib.md5(seq.encode()).hexdigest()
                if seq_hash in self.embedding_cache:
                    cached.append((i, self.embedding_cache[seq_hash]))
                else:
                    uncached_seqs.append(seq)
                    uncached_idx.append(i)
        else:
            uncached_seqs, uncached_idx, cached = sequences, list(range(len(sequences))), []
        
        # Process uncached sequences
        uncached_embeddings = []
        if uncached_seqs:
            uncached_embeddings = self._encode_batch(uncached_seqs)
            if self.embedding_cache is not None:
                for seq, emb in zip(uncached_seqs, uncached_embeddings):
                    self.embedding_cache[hashlib.md5(seq.encode()).hexdigest()] = emb
        
        # Combine results
        result = [None] * len(sequences)
        for idx, emb in cached:
            result[idx] = emb
        for i, emb in enumerate(uncached_embeddings):
            result[uncached_idx[i]] = emb
        
        return result
    
    def _encode_batch(self, sequences):
        """Encode sequences in batches"""
        embeddings = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_emb = outputs.last_hidden_state
                masked_emb = token_emb * attention_mask.unsqueeze(-1)
                pooled = masked_emb.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                pooled = pooled.cpu()
                
                for j in range(pooled.shape[0]):
                    embeddings.append(pooled[j])
        
        return embeddings
    
    def find_similar_proteins(self, target_sequence, protein_database, top_k=None):
        """Find similar proteins using ESM-2 embeddings"""
        if top_k is None:
            top_k = CONFIG["esm2"]["similarity_top_k"]
        
        target_emb = self.encode_sequences([target_sequence])[0].numpy()
        db_ids = list(protein_database.keys())
        db_seqs = [protein_database[pid]['sequence'] for pid in db_ids]
        db_embs = self.encode_sequences(db_seqs)
        
        similarities = []
        for i, db_emb in enumerate(db_embs):
            sim = cosine_similarity([target_emb], [db_emb.numpy()])[0][0]
            similarities.append({
                'protein_id': db_ids[i],
                'similarity': sim,
                'sequence': db_seqs[i]
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        min_thresh = CONFIG["esm2"]["min_similarity_threshold"]
        filtered = [s for s in similarities if s['similarity'] >= min_thresh]
        
        return filtered[:top_k]
    
    def clear_cache(self):
        if self.embedding_cache is not None:
            self.embedding_cache.clear()


class ChemBERTaSmilesEncoder:
    """ChemBERTa SMILES encoder for molecular representations"""
    
    def __init__(self):
        self.model_name = CONFIG["chemberta"]["model_name"]
        self.max_length = CONFIG["chemberta"]["max_length"]
        self.embedding_dim = CONFIG["chemberta"]["embedding_dim"]
        self.batch_size = CONFIG["chemberta"]["batch_size"]
        self.device = CONFIG["chemberta"]["device"]
        
        print(f"💊 Loading ChemBERTa model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("✅ ChemBERTa loaded successfully")
        
        self.embedding_cache = {} if CONFIG["chemberta"]["use_cached_embeddings"] else None
    
    def encode_smiles(self, smiles_list):
        """Encode SMILES strings using ChemBERTa"""
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Check cache
        if self.embedding_cache is not None:
            cached, uncached_smiles, uncached_idx = [], [], []
            for i, smi in enumerate(smiles_list):
                if smi in self.embedding_cache:
                    cached.append((i, self.embedding_cache[smi]))
                else:
                    uncached_smiles.append(smi)
                    uncached_idx.append(i)
        else:
            uncached_smiles, uncached_idx, cached = smiles_list, list(range(len(smiles_list))), []
        
        # Process uncached
        uncached_embs = []
        if uncached_smiles:
            uncached_embs = self._encode_batch(uncached_smiles)
            if self.embedding_cache is not None:
                for smi, emb in zip(uncached_smiles, uncached_embs):
                    self.embedding_cache[smi] = emb
        
        # Combine
        result = [None] * len(smiles_list)
        for idx, emb in cached:
            result[idx] = emb
        for i, emb in enumerate(uncached_embs):
            result[uncached_idx[i]] = emb
        
        return result
    
    def _encode_batch(self, smiles_list):
        """Encode SMILES in batches"""
        embeddings = []
        for i in range(0, len(smiles_list), self.batch_size):
            batch = smiles_list[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    batch_embs = outputs.last_hidden_state[:, 0, :]
                elif hasattr(outputs, 'pooler_output'):
                    batch_embs = outputs.pooler_output
                else:
                    batch_embs = outputs[0][:, 0, :]
                batch_embs = batch_embs.cpu()
                
                for j in range(batch_embs.shape[0]):
                    embeddings.append(batch_embs[j])
        
        return embeddings
    
    def find_similar_molecules(self, query_smiles, database_smiles, threshold=0.7, top_k=50):
        """Find molecules similar to query in database"""
        if isinstance(query_smiles, str):
            query_smiles = [query_smiles]
        
        query_embs = self.encode_smiles(query_smiles)
        db_embs = self.encode_smiles(database_smiles)
        
        similar = []
        for i, q_emb in enumerate(query_embs):
            q_np = q_emb.numpy()
            for j, db_emb in enumerate(db_embs):
                sim = cosine_similarity([q_np], [db_emb.numpy()])[0][0]
                if sim >= threshold:
                    similar.append({
                        'query_smiles': query_smiles[i],
                        'database_smiles': database_smiles[j],
                        'similarity': sim,
                        'database_index': j
                    })
        
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar[:top_k * len(query_smiles)]
    
    def clear_cache(self):
        if self.embedding_cache is not None:
            self.embedding_cache.clear()