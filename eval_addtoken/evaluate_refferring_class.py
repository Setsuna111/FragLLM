import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'  # restrict GPU visibility
from typing import Optional, Union, List, Dict, Any
from sentence_transformers import SentenceTransformer
from torchmetrics.classification import (
    Accuracy,
    Recall,
    Precision,
    F1Score,
    MatthewsCorrCoef,
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    MultilabelAveragePrecision,
    AveragePrecision,
    MulticlassF1Score
)


class ClsMetrics:
    """
        Multiclass classification metrics, for reference task.
    """
    
    def __init__(self, num_labels, device):

        super().__init__()
        self.num_labels = num_labels
        self.device = device
        
        self.metrics_dict = {
            'acc': Accuracy(task="multiclass", num_classes=num_labels).to(device),
            'recall': Recall(task="multiclass", num_classes=num_labels, average='macro').to(device),
            'precision': Precision(task="multiclass", num_classes=num_labels, average='macro').to(device),
            'f1': F1Score(task="multiclass", num_classes=num_labels, average='macro').to(device),
            'mcc': MatthewsCorrCoef(task="multiclass", num_classes=num_labels).to(device),
        }
    
    def update(self, pred, target):

        pred, target = pred.to(self.device), target.to(self.device)

        for metric in self.metrics_dict.values():
            metric.update(pred, target)

    def reset(self):

        for metric in self.metrics_dict.values():
            metric.reset()
    
    def compute(self):
        
        results = {}

        for name, metric in self.metrics_dict.items():
            results[name] = metric.compute()

        return results
    

class LanguageMetrics:
    """
    Placeholder for language metrics (BLEU, etc.) to be implemented.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.reset()
    
    def update(self, pred_texts: List[str], target_texts: List[str]):
        """Update metrics with prediction and target texts."""
        self.pred_texts.extend(pred_texts)
        self.target_texts.extend(target_texts)
    
    def reset(self):
        """Reset accumulated data."""
        self.pred_texts = []
        self.target_texts = []
    
    def compute(self) -> Dict[str, float]:
        """Compute language metrics. To be implemented."""
        return {
            'bleu': 0.0,
            'rouge_l': 0.0,
            'meteor': 0.0
        }


class ReferenceMetrics:
    """
    Unified metrics class for evaluating LLM outputs against InterPro protein database.
    Combines classification metrics and language metrics with InterPro-based retrieval.
    
    Supports four input modes:
    1. Label-only: Traditional classification with pred_labels + target_labels
    2. InterPro text retrieval: LLM text descriptions + true InterPro IDs 
    3. Language metrics: Predicted texts + target texts for BLEU/ROUGE
    4. Combined: Any combination of the above for comprehensive evaluation
    
    Key features:
    - Automatic InterPro database loading and embedding caching
    - Full-scale retrieval against entire InterPro database (not just input subset)
    - Dynamic class count based on InterPro database size
    - Support for both CSV and manual input
    """

    def __init__(self, 
                 device: str = 'cuda',
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 interpro_db_path: str = "/home/lfj/projects_dir/FragLLM/data/new_interpro_metadata_short_with_fragment_type_v3.json",
                 cache_dir: str = "/home/lfj/projects_dir/FragLLM/eval/cache",
                 batch_size: int = 32):
        
        self.device = device
        self.embedding_model_name = embedding_model
        self.embedding_model = None  # Lazy initialization
        self.interpro_db_path = interpro_db_path
        self.cache_dir = cache_dir
        self.batch_size = batch_size  # Batch size for embedding computation
        
        # Initialize InterPro database
        self.interpro_data = self._load_interpro_database()
        
        # Cache for label embeddings
        self.label_embeddings_cache = None
        self.interpro_ids_list = None
        self.interpro_descriptions_list = None
        
        # Get actual number of labels from InterPro database
        self.num_labels = len(self.interpro_data)
        print(f"Initialized with {self.num_labels} InterPro classes")
        
        # Initialize classification metrics with correct number of labels
        self.cls_metrics = ClsMetrics(
            num_labels=self.num_labels,
            device=device
        )
        
        # Initialize language metrics
        self.lang_metrics = LanguageMetrics(device=device)
        
        # Track what type of data has been added
        self.has_labels = False
        self.has_texts = False
        
        # Data init
        self.reset()
    
    def _load_interpro_database(self):
        """Load InterPro database from JSON file."""
        try:
            with open(self.interpro_db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} InterPro entries from {self.interpro_db_path}")
            return data
        except Exception as e:
            print(f"Error loading InterPro database: {e}")
            return {}
    
    def _initialize_embedding_model(self):
        """Lazy initialization of embedding model."""
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to(self.device)
            except Exception as e:
                print(f"Warning: Could not load embedding model {self.embedding_model_name}: {e}")
                self.embedding_model = None
    
    def _get_cache_path(self):
        """Get cache file path for label embeddings."""
        os.makedirs(self.cache_dir, exist_ok=True)
        model_name = self.embedding_model_name.replace('/', '_').replace('\\', '_')
        cache_filename = f"interpro_embeddings_{model_name}.pkl"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _load_or_compute_label_embeddings(self):
        """Load cached label embeddings or compute them if not cached."""
        if self.label_embeddings_cache is not None:
            return self.label_embeddings_cache, self.interpro_ids_list, self.interpro_descriptions_list
        
        cache_path = self._get_cache_path()
        
        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                self.label_embeddings_cache = cache_data['embeddings']
                self.interpro_ids_list = cache_data['interpro_ids']
                self.interpro_descriptions_list = cache_data['descriptions']
                print(f"Loaded label embeddings from cache: {cache_path}")
                return self.label_embeddings_cache, self.interpro_ids_list, self.interpro_descriptions_list
            except Exception as e:
                print(f"Error loading cache: {e}, recomputing embeddings...")
        
        # Compute embeddings
        print("Computing label embeddings for all InterPro entries...")
        self.interpro_ids_list = []
        self.interpro_descriptions_list = []
        
        for interpro_id, entry in self.interpro_data.items():
            self.interpro_ids_list.append(interpro_id)
            self.interpro_descriptions_list.append(entry['description_short'])
        
        # Encode all descriptions in batches to avoid OOM
        self._initialize_embedding_model()
        if self.embedding_model is None:
            raise ValueError("Embedding model not available for computing label embeddings")
        
        self.label_embeddings_cache = self._encode_texts_in_batches(
            self.interpro_descriptions_list
        )
        
        # Save to cache
        try:
            cache_data = {
                'embeddings': self.label_embeddings_cache,
                'interpro_ids': self.interpro_ids_list,
                'descriptions': self.interpro_descriptions_list
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved label embeddings to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
        
        return self.label_embeddings_cache, self.interpro_ids_list, self.interpro_descriptions_list
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence transformer."""
        self._initialize_embedding_model()
        if self.embedding_model is None:
            raise ValueError("Embedding model not available")
        
        # For small batches, encode directly
        if len(texts) <= self.batch_size:
            with torch.no_grad():  # Always disable gradients for inference
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings
        else:
            # For large batches, use batch processing
            return self._encode_texts_in_batches(texts)
    
    def _encode_texts_in_batches(self, texts: List[str]) -> np.ndarray:
        """Encode texts in batches to avoid OOM."""
        self._initialize_embedding_model()
        if self.embedding_model is None:
            raise ValueError("Embedding model not available")
        
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        print(f"Encoding {len(texts)} texts in {total_batches} batches (batch_size={self.batch_size})")
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")
            
            try:
                with torch.no_grad():  # Disable gradient computation to save memory
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts, 
                        convert_to_tensor=False,
                        show_progress_bar=False  # Disable individual batch progress
                    )
                all_embeddings.append(batch_embeddings)
                
                # Clear CUDA cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in batch {batch_num}: {e}")
                # Try with smaller batch size if OOM occurs
                if "out of memory" in str(e).lower() and self.batch_size > 1:
                    print(f"OOM detected, trying with smaller batch size...")
                    smaller_batch_size = max(1, self.batch_size // 2)
                    sub_embeddings = []
                    for j in range(0, len(batch_texts), smaller_batch_size):
                        sub_batch = batch_texts[j:j + smaller_batch_size]
                        with torch.no_grad():  # Also disable for recovery batches
                            sub_emb = self.embedding_model.encode(sub_batch, convert_to_tensor=False)
                        sub_embeddings.append(sub_emb)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    batch_embeddings = np.vstack(sub_embeddings)
                    all_embeddings.append(batch_embeddings)
                else:
                    raise e
        
        # Concatenate all batch embeddings
        final_embeddings = np.vstack(all_embeddings)
        print(f"Successfully encoded {len(texts)} texts to embeddings shape: {final_embeddings.shape}")
        
        return final_embeddings
    
    def _retrieval_based_classification(self, 
                                      pred_descriptions: List[str], 
                                      true_interpro_ids: List[str]) -> tuple:
        """
        Convert text descriptions to classifications via embedding similarity against full InterPro database.
        
        Args:
            pred_descriptions: List of predicted descriptions from LLM
            true_interpro_ids: List of true InterPro IDs
            
        Returns:
            tuple: (pred_labels, true_labels) as tensors with indices in InterPro database
        """
        # Load or compute all InterPro label embeddings
        label_embeddings, interpro_ids_list, _ = self._load_or_compute_label_embeddings()
        
        # Encode prediction descriptions
        pred_embeddings = self._encode_texts(pred_descriptions)
        
        similarities = self.embedding_model.similarity(pred_embeddings, label_embeddings)

        # # Compute similarity against all InterPro entries
        # from sklearn.metrics.pairwise import cosine_similarity
        # similarities = cosine_similarity(pred_embeddings, label_embeddings)
        
        # Find best matches for predictions
        pred_indices = np.argmax(similarities, axis=1)
        
        # Convert true InterPro IDs to indices in the database
        true_indices = []
        for interpro_id in true_interpro_ids:
            if interpro_id in interpro_ids_list:
                true_indices.append(interpro_ids_list.index(interpro_id))
            else:
                print(f"Warning: InterPro ID {interpro_id} not found in database")
                true_indices.append(0)  # Default to first entry if not found
        
        return torch.tensor(pred_indices), torch.tensor(true_indices)
    
    def _convert_labels_to_tensors(self, pred_labels, target_labels):
        """Convert label data to tensors."""
        def to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, list):
                # Handle mixed int/str labels
                converted = []
                for x in data:
                    if isinstance(x, (int, float)):
                        converted.append(int(x))
                    else:
                        # Convert string to hash-based index
                        converted.append(hash(str(x)) % 1000)
                return torch.tensor(converted)
            else:
                raise ValueError(f"Unsupported label type: {type(data)}")
        
        return to_tensor(pred_labels), to_tensor(target_labels)
    
    def load_from_csv(self, 
                     csv_path: str,
                     pred_label_col: Optional[str] = None,
                     target_label_col: Optional[str] = None,
                     pred_text_col: Optional[str] = None,
                     target_text_col: Optional[str] = None,
                     target_interpro_id_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and evaluate data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            pred_label_col: Column name for predicted labels (optional)
            target_label_col: Column name for target labels (optional)
            pred_text_col: Column name for predicted texts (optional)
            target_text_col: Column name for target texts (optional)
            target_interpro_id_col: Column name for target InterPro IDs (optional)
        
        Returns:
            Dict containing evaluation results
        """
        df = pd.read_csv(csv_path)
        
        pred_labels = None
        target_labels = None
        pred_texts = None
        target_texts = None
        target_interpro_ids = None
        
        if pred_label_col and target_label_col:
            pred_labels = df[pred_label_col].tolist()
            target_labels = df[target_label_col].tolist()
        
        if pred_text_col:
            pred_texts = df[pred_text_col].tolist()
            
        if target_text_col:
            target_texts = df[target_text_col].tolist()
            
        if target_interpro_id_col:
            target_interpro_ids = df[target_interpro_id_col].tolist()
        
        self.update(
            pred_labels=pred_labels,
            target_labels=target_labels,
            pred_texts=pred_texts,
            target_texts=target_texts,
            target_interpro_ids=target_interpro_ids
        )
        
        return self.compute()
    
    def update(self, 
               pred_labels: Optional[Union[torch.Tensor, List]] = None,
               target_labels: Optional[Union[torch.Tensor, List]] = None,
               pred_texts: Optional[List[str]] = None,
               target_texts: Optional[List[str]] = None,
               target_interpro_ids: Optional[List[str]] = None):
        """
        Update metrics with predictions and targets.
        
        Args:
            pred_labels: Predicted class labels (tensor or list) - for label-only mode
            target_labels: True class labels (tensor or list) - for label-only mode
            pred_texts: Predicted text descriptions from LLM (list of strings)
            target_texts: Target text descriptions (list of strings) - for language metrics
            target_interpro_ids: True InterPro IDs (list of strings) - for text retrieval mode
        """
        # Mode 1: Label-only input (traditional classification)
        if pred_labels is not None and target_labels is not None:
            pred_tensor, target_tensor = self._convert_labels_to_tensors(pred_labels, target_labels)
            self.cls_metrics.update(pred_tensor, target_tensor)
            self.has_labels = True
            
        # Mode 2: Text-only input with InterPro retrieval
        if pred_texts is not None and target_interpro_ids is not None:
            # Use InterPro database for retrieval-based classification
            pred_tensor, target_tensor = self._retrieval_based_classification(
                pred_texts, target_interpro_ids
            )
            self.cls_metrics.update(pred_tensor, target_tensor)
            self.has_labels = True
            
        # Language metrics (when both pred_texts and target_texts are provided)
        if pred_texts is not None and target_texts is not None:
            self.lang_metrics.update(pred_texts, target_texts)
            self.has_texts = True
        
        # Validation: ensure at least one type of data is provided
        if not self.has_labels and not self.has_texts:
            raise ValueError("Must provide either labels, texts, or InterPro IDs for evaluation")
    
    def reset(self):
        """Reset all metrics."""
        self.cls_metrics.reset()
        self.lang_metrics.reset()
        self.has_labels = False
        self.has_texts = False
    
    def compute(self) -> Dict[str, Any]:
        """
        Compute metrics based on what data has been provided.
        
        Returns:
            Dict containing classification and/or language metrics
        """
        results = {}
        
        # Always compute classification metrics if labels were provided
        if self.has_labels:
            cls_results = self.cls_metrics.compute()
            for key, value in cls_results.items():
                results[f'cls_{key}'] = value
        
        # Compute language metrics if texts were provided
        if self.has_texts:
            lang_results = self.lang_metrics.compute()
            for key, value in lang_results.items():
                results[f'lang_{key}'] = value
        
        return results


# Example usage:
if __name__ == "__main__":

    # Initialize metrics (num_labels automatically determined from InterPro database)
    print("=== Initializing InterPro-based Evaluation System ===")
    metrics = ReferenceMetrics(
        device='cuda',
        embedding_model="/home/lfj/projects_dir/pretrained_model/Qwen3-Embedding-0.6B",
        interpro_db_path="/home/lfj/projects_dir/FragLLM/data/new_interpro_metadata_short_with_fragment_type_v3.json",
        cache_dir="/home/lfj/projects_dir/FragLLM/eval/cache",
        batch_size=16
    )
    print(f"System ready with {metrics.num_labels} InterPro classes\n")

    # ============ Four Input Modes Examples ============
    
    # Mode 1: Traditional Label-only Classification
    print("=== Mode 1: Traditional Classification ===")
    metrics.reset()
    metrics.update(
        pred_labels=[0, 1, 2, 1, 0],  # Predicted class indices
        target_labels=[0, 1, 1, 1, 0]  # True class indices
    )
    results = metrics.compute()
    print("Traditional classification results:", results)
    print("✓ Only classification metrics computed\n")

    # Mode 2: InterPro Text Retrieval (Core functionality)
    print("=== Mode 2: InterPro Text Retrieval ===")
    metrics.reset()
    metrics.update(
        pred_texts=[
            "A glycine-rich domain involved in microtubule organization and cytoskeletal transport",
            "C-terminal domain found in signal-induced proliferation proteins with GTPase activity"
        ],
        target_interpro_ids=["IPR000938", "IPR021818"]  # True InterPro IDs from database
    )
    results = metrics.compute()
    print("InterPro retrieval results:", results)
    print("✓ Classification via retrieval against full InterPro database\n")

    # Mode 3: Language Metrics Only
    print("=== Mode 3: Language Metrics Only ===")
    metrics.reset()
    metrics.update(
        pred_texts=[
            "The CAP-Gly domain is involved in cytoskeletal organization",
            "SIPA1L domain regulates GTPase signaling pathways"
        ],
        target_texts=[
            "CAP-Gly domain for microtubule organization and vesicle transport",
            "SIPA1L C-terminal domain for signal-induced proliferation regulation"
        ]
    )
    results = metrics.compute()
    print("Language-only results:", results)
    print("✓ Only language metrics (BLEU, ROUGE, etc.) computed\n")

    # Mode 4: Combined Evaluation (Most comprehensive)
    print("=== Mode 4: Combined Evaluation ===")
    metrics.reset()
    metrics.update(
        pred_texts=[
            "Domain with glycine-rich sequence for microtubule binding",
            "Protein domain that activates GTPases in proliferation signaling"
        ],
        target_interpro_ids=["IPR000938", "IPR021818"],  # For classification metrics
        target_texts=[
            "CAP-Gly domain: cytoskeleton-associated protein domain",
            "SIPA1L domain: signal-induced proliferation-associated domain"
        ]  # For language metrics
    )
    results = metrics.compute()
    print("Combined evaluation results:", results)
    print("✓ Both classification (via InterPro) AND language metrics computed\n")

    # ============ CSV Usage Examples ============
    print("=== CSV Usage Examples ===")
    
    # CSV Example 1: InterPro retrieval + Language metrics
    print("# Comprehensive CSV evaluation:")
    print("# results = metrics.load_from_csv(")
    print("#     csv_path='llm_predictions.csv',")
    print("#     pred_text_col='llm_output',           # LLM generated descriptions")
    print("#     target_text_col='ground_truth_desc',  # Reference descriptions for BLEU/ROUGE")
    print("#     target_interpro_id_col='interpro_id'  # True InterPro IDs for retrieval")
    print("# )")
    print("# → Returns both classification and language metrics")
    print()
    
    # CSV Example 2: InterPro retrieval only
    print("# InterPro retrieval only:")
    print("# results = metrics.load_from_csv(")
    print("#     csv_path='predictions.csv',")
    print("#     pred_text_col='llm_description',")
    print("#     target_interpro_id_col='true_interpro_id'")
    print("# )")
    print("# → Returns classification metrics via InterPro database retrieval")
    print()
    
    print("=== System Features ===")
    print("✓ Automatic InterPro database loading")
    print("✓ Intelligent embedding caching")
    print("✓ Full-scale retrieval (entire database, not just input subset)")
    print("✓ Dynamic class count based on database size")
    print("✓ Support for mixed evaluation modes")
    print("✓ Optimized for protein domain classification tasks")

        