# modern_methods.py (修正版：適用 OpenAI v1.0+)
from openai import OpenAI
import json

def ai_similarity(text1, text2, api_key):

    # 建立客戶端 (v1.0+ 新寫法)
    client = OpenAI(api_key=api_key)

    prompt = f"""
    請評估以下兩段文字的語意相似度。
    文字1: {text1}
    文字2: {text2}
    請只回答一個 0 到 100 的整數數字。
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一個語意相似度評估助理。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        # 擷取回覆 (v1.0+ 改用屬性存取，非字典 key)
        reply = response.choices[0].message.content.strip()

        # 數值轉換邏輯
        score = None
        for token in reply.split():
            try:
                score = int(token)
                break
            except ValueError:
                continue
        
        return score if score is not None else 0

    except Exception as e:
        print("❌ API 錯誤：", e)
        return 0

def ai_classify(text, api_key):
    """
    使用 GPT-3.5 進行多維度分類
    """
    client = OpenAI(api_key=api_key)

    prompt = f"""
    請閱讀以下文字，並根據內容進行情緒與主題分類。
    文字內容：{text}
    請以 JSON 格式回答，格式如下：
    {{
        "sentiment": "正面/負面/中性",
        "topic": "主題類別",
        "confidence": 0.95
    }}
    僅輸出 JSON。
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一個專業的文本分類助理。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"} # v1.0+ 支援強制 JSON 模式
        )

        reply = response.choices[0].message.content.strip()
        result = json.loads(reply)
        return result

    except Exception as e:
        print("❌ API 錯誤：", e)
        return {"sentiment": "中性", "topic": "未知", "confidence": 0.0}

def ai_summarize(text, max_length, api_key):
    """
    使用 GPT-3.5 生成摘要
    """
    client = OpenAI(api_key=api_key)

    prompt = f"""
    請根據以下文字生成摘要，長度不超過 {max_length} 字：
    {text}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位專業的摘要助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ API 錯誤：", e)
        return "摘要生成失敗"