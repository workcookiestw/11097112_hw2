import math
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

# --- A-1 TF-IDF & Similarity (Manual Implementation) ---
def calculate_tf(word_dict, total_words):
    """計算詞頻 (Term Frequency)"""
    tf_dict = {}
    for word, count in word_dict.items():
        tf_dict[word] = count / total_words
    return tf_dict

def calculate_idf(documents, word):
    """計算逆文件頻率 (IDF)"""
    num_docs = len(documents)
    # 避免分母為 0，通常會 +1
    containing_docs = sum(1 for doc in documents if word in doc)
    idf = math.log10(num_docs / (1 + containing_docs))
    return idf

def calculate_tfidf_similarity(documents):
    """
    使用 scikit-learn 計算文件之間的 TF-IDF 與相似度
    修正：加入 jieba 中文斷詞
    """
    docs_segmented = [" ".join(jieba.cut(doc)) for doc in documents]
    
    # 這裡使用 sklearn 是為了矩陣運算的便利性與作業 B 部分的要求
    # 但手動計算的邏輯保留在 calculate_tf/idf 中供驗證使用
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(docs_segmented)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    return feature_names, tfidf_matrix.toarray(), similarity_matrix

# --- A-2 Rule-Based Classification ---
class RuleBasedSentimentClassifier:
    def __init__(self):
        self.positive_words = ['好', '棒', '優秀', '喜歡', '推薦', '滿意', '開心', '值得', '精彩', '完美', '不錯']
        self.negative_words = ['差', '糟', '失望', '討厭', '不推薦', '浪費', '無聊', '爛', '糟糕', '差勁', '生氣', '壞']
        self.negation_words = ['不', '沒', '無', '非', '別']
        
        # [NEW] 程度副詞 (作業要求 A-2)
        self.degree_adverbs = {
            '很': 1.5, '非常': 2.0, '超': 2.5, '太': 2.0, '真': 1.5, '特別': 1.8,
            '有點': 0.8, '稍微': 0.8
        }

    def classify(self, text):
        score = 0
        
        # 簡單斷詞以利捕捉程度副詞
        # 注意：規則式通常需要精細的 window 掃描，這裡簡化為檢查前一個詞
        segments = list(jieba.cut(text))
        
        for i, word in enumerate(segments):
            word_score = 0
            
            # 判斷基礎分數
            if word in self.positive_words:
                word_score = 1
            elif word in self.negative_words:
                word_score = -1
            
            if word_score != 0:
                # 檢查前一個詞是否為程度副詞或否定詞
                if i > 0:
                    prev_word = segments[i-1]
                    
                    # 1. 處理否定 (否定 + 正 = 負)
                    if prev_word in self.negation_words:
                        word_score *= -1
                    
                    # 2. [NEW] 處理程度副詞 (加權)
                    elif prev_word in self.degree_adverbs:
                        word_score *= self.degree_adverbs[prev_word]
                
                score += word_score

        if score > 0: return '正面'
        elif score < 0: return '負面'
        else: return '中性'

# --- A-3 Statistical Summarization ---
class StatisticalSummarizer:
    def __init__(self):
        self.stop_words = set(['的','了','在','是','我','有','和','就','不','人','都','一','上','也','很','到','說','要','去','你'])

    def sentence_score(self, sentence, word_freq, index, total_sentences):
        words = [w for w in jieba.cut(sentence) if w not in self.stop_words]
        if not words: return 0
        
        # 1. 詞頻得分
        score = sum(word_freq.get(w, 0) for w in words)
        
        # 2. 句長懲罰/獎勵
        if len(words) > 50: score *= 0.8
        elif len(words) < 5: score *= 0.7
        
        # 3. [NEW] 句子位置加權 (作業要求 A-3)
        # 首句通常是摘要，尾句通常是總結
        if index == 0:
            score *= 1.5
        elif index == total_sentences - 1:
            score *= 1.2
            
        return score

    def summarize(self, text, ratio=0.3):
        # 簡單斷句
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        
        all_words = []
        for s in sentences:
            all_words.extend([w for w in jieba.cut(s) if w not in self.stop_words])
        word_freq = Counter(all_words)

        # 傳入 index 以進行位置加權
        sentence_scores = {s: self.sentence_score(s, word_freq, i, len(sentences)) for i, s in enumerate(sentences)}
        
        select_count = max(1, int(len(sentences) * ratio))
        top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:select_count]

        # 保持原文順序輸出
        summary_sentences = []
        for s in sentences:
            if s in top_sentences:
                summary_sentences.append(s)
                
        return '。'.join(summary_sentences) + '。'