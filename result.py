import matplotlib
matplotlib.use('Agg') # この行を追加
import matplotlib.pyplot as plt
import pandas as pd
import os
import japanize_matplotlib
import numpy as np
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

    # -----------------------------
    # 🔽 グラフ作成（積み上げ + 合計点表示）
    # -----------------------------
    x = np.arange(len(elements))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    bar1 = ax.bar(x, score_df['1'], width, label='低', color='skyblue')
    bar2 = ax.bar(x, score_df['2'], width, bottom=score_df['1'], label='Medium (2 points)', color='orange')
    bar3 = ax.bar(x, score_df['3'], width, bottom=score_df['1'] + score_df['2'], label='High (3 points)', color='salmon')

    # 合計点ラベル
    for i, total in enumerate(total_scores):
        ax.text(x[i], total + 0.5, str(int(total)), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, total_scores.max() + 3)

    # ラベル・装飾
    ax.set_ylabel('Total Points')
    ax.set_title('Survey Evaluation (Stacked by Score)')
    ax.set_xticks(x)
    ax.set_xticklabels(elements, rotation=30)
    ax.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path)
    plt.close()
        