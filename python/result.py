
import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # この行を追加
import matplotlib.pyplot as plt
import japanize_matplotlib
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

def group_plot(csv,output_path):

    df = pd.read_csv(csv, on_bad_lines='skip')
    df = pd.read_csv(csv, on_bad_lines='warn')
    # -----------------------------
    # 🔽 評価対象の要素（最後の7列）
    # -----------------------------
    elements = [
        "Weather",        # 天気
        "Day of Week",    # 曜日
        "Time of Day",    # 時間帯
        "Event Presence", # イベントの有無
        "Accessibility",  # アクセスの良さ
        "Visit Frequency",# 来店回数
        "Crowdedness"     # 混雑状況
    ]

    # CSVから最後の7列を抽出
    eval_df = df.iloc[:, -8: -1]
    eval_df.columns = elements

    # -----------------------------
    # 🔽 スコア換算（1:Low=1点, 2:Medium=2点, 3:High=3点）
    # -----------------------------
    score_df = pd.DataFrame({
        '1': (eval_df == 1).sum() * 1,
        '2': (eval_df == 2).sum() * 2,
        '3': (eval_df == 3).sum() * 3
    }, index=elements)

    # 合計点
    total_scores = score_df.sum(axis=1)

    # 横棒グラフの描画
    y = np.arange(len(elements))
    height = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    bar1 = ax.barh(y, score_df['1'], height, label='低', color='#0072B2')
    bar2 = ax.barh(y, score_df['2'], height, left=score_df['1'], label='中', color='#E69F00')
    bar3 = ax.barh(y, score_df['3'], height, left=score_df['1'] + score_df['2'], label='高', color='#009E73')

    # 合計点ラベルを右側に表示
    for i, total in enumerate(total_scores):
        ax.text(total + 0.3, y[i], str(int(total)), va='center', fontsize=15, fontweight='bold')

    # y軸の順番を逆にするための処理
    elements_rev = elements[::-1]         # ラベルを逆順に
    y = np.arange(len(elements))[::-1]    # y軸の位置も逆順に合わせる

    # 軸やタイトルなどの設定
    ax.set_xlim(0, total_scores.max() * 1.2)
    ax.set_xlabel('合計点数', fontsize=14)
    ax.set_title('来店に影響する各要素の合計点数', fontsize=16)
    ax.set_yticks(y)
    ax.set_yticklabels(elements, fontsize=12)
    ax.legend(
        handles=[bar3, bar2, bar1],
        labels=['高', '中', '低'],
        title='影響度',
        title_fontsize=13,
        prop={'size': 15}
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path)
    plt.close()