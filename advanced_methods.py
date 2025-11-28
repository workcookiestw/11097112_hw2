# advanced_methods.py
import numpy as np
import jieba
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import time

class Word2VecSimilarity:
    def __init__(self, sentences=None):
        """
        初始化 Word2Vec 模型。
        為了效能優化，若無預訓練模型，我們使用當下語料快速訓練一個輕量模型。
        """
        self.model = None
        if sentences:
            self.train_model(sentences)

    def train_model(self, sentences):
        # 斷詞處理
        tokenized_sentences = [list(jieba.cut(s)) for s in sentences]
        # 訓練模型 (效能優化：設定 min_count=1 確保所有詞都被考慮，vector_size=100 為平衡點)
        self.model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

    def get_sentence_vector(self, text):
        if not self.model:
            return np.zeros(100)
        
        words = list(jieba.cut(text))
        word_vecs = [self.model.wv[w] for w in words if w in self.model.wv]
        
        if not word_vecs:
            return np.zeros(self.model.vector_size)
        
        # 使用平均詞向量代表句向量 (Mean Pooling)
        return np.mean(word_vecs, axis=0)

    def calculate_similarity(self, text1, text2):
        vec1 = self.get_sentence_vector(text1).reshape(1, -1)
        vec2 = self.get_sentence_vector(text2).reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0] * 100  # 轉為 0-100 分

class PerformanceMonitor:
    """效能監控器：用於精確測量時間與估算成本"""
    def __init__(self, model_name="gpt-4o"):
        self.start_time = 0
        self.end_time = 0
        self.model_name = model_name
        # 2025年 預估價格 (每 1M tokens)
        self.pricing = {
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "traditional": {"input": 0, "output": 0}
        }

    def start(self):
        self.start_time = time.perf_counter() # 使用 perf_counter 獲得更高精度

    def stop(self):
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time

    def estimate_cost(self, input_text, output_text):
        if self.model_name == "traditional" or self.model_name == "word2vec":
            return 0.0
            
        # 簡易 Token 估算 (1 中文字約等於 1.5-2 tokens，這裡粗略估計)
        input_tokens = len(input_text) * 1.5
        output_tokens = len(str(output_text)) * 1.5
        
        rates = self.pricing.get(self.model_name, self.pricing["gpt-4o"])
        cost = (input_tokens / 1_000_000 * rates["input"]) + \
               (output_tokens / 1_000_000 * rates["output"])
        return cost