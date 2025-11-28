# NLP Models Comparison: Traditional vs. Modern

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-v1.0%2B-green)](https://openai.com/)

本專案是一個綜合性的 NLP 基準測試工具，旨在比較 **傳統方法** (TF-IDF, Rule-Based, Statistical) 與 **現代大型語言模型** (GPT-4o) 以及 **進階詞向量方法** (Word2Vec) 在不同自然語言處理任務上的表現。

專案會自動生成量化報表、視覺化圖表以及詳細的比較數據，並輸出至 `results/` 資料夾。

## ✨ 主要功能

本專案針對以下三大任務進行效能、準確率與成本的比較：

1.  **語意相似度計算 (Semantic Similarity)**
    * Traditional: TF-IDF + Cosine Similarity
    * Advanced: Word2Vec (Gensim)
    * Modern: GPT-4o (Semantic Understanding)
2.  **文本分類 (Text Classification)**
    * Traditional: Rule-Based Sentiment Analysis
    * Modern: GPT-4o (Contextual Classification)
3.  **自動摘要 (Text Summarization)**
    * Traditional: Statistical Summarizer (Frequency-based)
    * Modern: GPT-4o (Generative Summarization)

## 📂 檔案結構

```text
.
├── comparison.py    # [主程式] 執行測試、計算指標並生成所有報表
├── traditional_methods.py # 傳統演算法實作 (TF-IDF, 規則庫, 統計摘要)
├── modern_methods.py      # 現代 AI 實作 (OpenAI API v1.0+ Client)
├── advanced_methods.py    # 進階演算法實作 (Word2Vec, 效能監控器)
├── requirements.txt       # 專案依賴套件清單
├── README.md              # 說明文件
└── results/               # [自動生成] 存放所有輸出結果

## 🚀 執行方式
1. 環境設定請確保您已安裝 Python 3.8 或以上版本
2. 安裝套件：
   ```bash
   pip install -r requirements.txt

3. 設定 API Key ( 打開 comparison.py )

# 若填入 OpenAI API Key，請填入並將 MOCK_MODE 設為 False
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 
MOCK_MODE = False 

4. 執行程式執行主程式以開始基準測試：python comparison.py
程式執行完畢後，會自動建立 results/ 資料夾，並輸出結果檔案。

## 檔名類型說明
程式執行後會在 results/ 產生：

1. tfidf_similarity_matrix.png: 相似度熱力圖。
2. classification_results.csv: 分類準確度報表。
3. summarization_comparison.txt: 摘要文字比較。
4. performance_metrics.json: 效能數據 JSON 檔。