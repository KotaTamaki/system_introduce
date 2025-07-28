
import pandas as pd
import os
import numpy as np
import json

def prepare_chart_data(csv_path):
    try:
        # --- デバッグ ---
        print(f"--- Reading CSV: {csv_path} ---")
        if not os.path.exists(csv_path):
            print("-> File does not exist.")
            return None
        
        df = pd.read_csv(csv_path)

        # ▼▼▼【重要】読み込んだデータフレームの中身を出力 ▼▼▼
        print("-> DataFrame Head:")
        print(df.head())
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        if df.empty:
            print("-> DataFrame is empty. Skipping.")
            return None

        # --- 以下、元のロジック ---
        elements = [
            "天気", "曜日", "時間帯", "イベント有無", 
            "アクセス", "来店回数", "混雑状況"
        ]
        
        # CSVの列数が足りない場合のエラーを防ぐ
        if df.shape[1] < 7:
            print(f"-> Error: CSV file has fewer than 7 columns. Found {df.shape[1]}.")
            return None

        eval_df = df.iloc[:, -8:-1]
        eval_df.columns = elements

        score_df = pd.DataFrame({
            '1': (eval_df == 1).sum() * 1,
            '2': (eval_df == 2).sum() * 2,
            '3': (eval_df == 3).sum() * 3
        }, index=elements)

        total_counts = score_df.sum(axis=1)

        chart_data = {
            'labels': elements,
            'datasets': [
                {'label': '低評価', 'data': score_df['1'].tolist(), 'backgroundColor': '#0072B2'},
                {'label': '中評価', 'data': score_df['2'].tolist(), 'backgroundColor': '#E69F00'},
                {'label': '高評価', 'data': score_df['3'].tolist(), 'backgroundColor': '#009E73'}
            ],
            'totals': total_counts.tolist()
        }
        print("-> Chart data prepared successfully.")
        return chart_data

    except Exception as e:
        print(f"-> An unexpected error occurred while processing {csv_path}: {e}")
        # エラーが発生した場合もNoneを返す
        return None