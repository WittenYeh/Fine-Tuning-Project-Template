# src/utils.py

import os
import xml.etree.ElementTree as ET
import pandas as pd
import re
from termcolor import cprint

def load_xml_data(file_path):
    """从XML文件中加载吸烟状态数据集。"""
    if not os.path.exists(file_path):
        cprint(f"错误: 文件未找到 {file_path}", "red")
        return pd.DataFrame(columns=['ID', 'text', 'status'])

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        cprint(f"错误: 解析XML文件失败 {file_path}: {e}", "red")
        return pd.DataFrame(columns=['ID', 'text', 'status'])

    records = []
    for record in root.findall('RECORD'):
        record_id = record.get('ID')
        text_element = record.find('TEXT')
        text = text_element.text.strip() if text_element is not None and text_element.text is not None else ""

        status = None
        smoking_element = record.find('SMOKING')
        if smoking_element is not None:
            status = smoking_element.get('STATUS')

        records.append({'ID': record_id, 'text': text, 'status': status})

    df = pd.DataFrame(records)
    cprint(f"成功加载 {len(df)} 条记录，来源: {os.path.basename(file_path)}", "green")
    return df

LABEL_MAP = {
    "CUERRENT SMOKER": "CURRENT SMOKER",
    "SMOKER": "SMOKER",
    "PAST SMOKER": "PAST SMOKER",
    "NON-SMOKER": "NON-SMOKER",
    "UNKNOWN": "UNKNOWN"
}

def clean_prediction(text):
    """从模型输出中提取最可能的标签"""
    text = text.upper()
    valid_labels = ["PAST SMOKER", "CURRENT SMOKER", "NON-SMOKER", "SMOKER", "UNKNOWN"]
    for label in valid_labels:
        if re.search(r'\b' + re.escape(label) + r'\b', text):
            # 这是一个特殊处理，防止 "CURRENT SMOKER" 被错误地识别为 "SMOKER"
            if label == "SMOKER":
                if "CURRENT SMOKER" in text or "PAST SMOKER" in text:
                    continue
            return label
    return "UNKNOWN"