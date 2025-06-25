import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime
import pandas as pd

# Flaskアプリケーションの初期化
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # flashメッセージのために必要

# アンケート結果を保存するディレクトリ
DATA_FOLDER = 'data'
# dataディレクトリが存在しない場合は作成
os.makedirs(DATA_FOLDER, exist_ok=True)

# 質問項目
QUESTIONS = [
    "質問1: この研究の新規性は評価できますか？",
    "質問2: 提案手法の有効性は明確に示されていると感じますか？",
    "質問3: 論文の構成や説明は分かりやすいですか？",
    "質問4: 実世界への応用可能性は高いと感じますか？",
    "質問5: この研究は、関連分野に貢献すると思いますか？"
]

@app.route('/')
def index():
    """研究概要ページ"""
    research_summary = """
    研究論文は、階層的な販売予測における精度向上と一貫性の確保に焦点を当てています。具体的には、限られた履歴データ、有用な情報の不足、および階層的制約の考慮不足といった課題に対処するため、特徴の分離抽出と再調整に基づく新しい予測手法を提案しています。提案されたフレームワークは、類似製品データを用いた事前学習と微調整、時系列依存および非依存特徴を独立して抽出するモジュール、そして階層的な一貫性を保証する調整モジュールと損失関数を組み合わせています。実世界の小売データを用いた実験により、この手法が既存の主流な予測モデルと比較して優れた性能を示し、供給チェーン管理における意思決定を効果的に支援することが実証されました。
    """
    return render_template('index.html', summary=research_summary)

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """アンケートページ"""
    if request.method == 'POST':
        # フォームからデータを辞書として受け取る
        data = {
            'q1': request.form.get('q1'),
            'q2': request.form.get('q2'),
            'q3': request.form.get('q3'),
            'q4': request.form.get('q4'),
            'q5': request.form.get('q5'),
            'comment': request.form.get('comment'),
            'timestamp': datetime.now().isoformat()
        }

        # 必須項目が選択されているかチェック
        if not all([data['q1'], data['q2'], data['q3'], data['q4'], data['q5']]):
            flash('すべての評価質問に回答してください。', 'danger')
            return render_template('survey.html', questions=QUESTIONS, form_data=data)

        # タイムスタンプをファイル名にしてJSONファイルとして保存
        filename = os.path.join(DATA_FOLDER, f"result_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        flash('アンケートへのご協力、ありがとうございました！', 'success')
        return redirect(url_for('results'))

    return render_template('survey.html', questions=QUESTIONS)

@app.route('/results')
def results():
    """集計結果ページ"""
    all_data = []
    # dataディレクトリ内の全jsonファイルを読み込む
    for filename in sorted(os.listdir(DATA_FOLDER)):
        if filename.endswith('.json'):
            filepath = os.path.join(DATA_FOLDER, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    all_data.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: Skipping corrupted or empty file: {filename}")


    chart_data = None
    comments = []
    total_responses = len(all_data)

    if all_data:
        # Pandas DataFrameに変換して集計を容易にする
        df = pd.DataFrame(all_data)
        
        # コメントを抽出
        comments = df['comment'].dropna().tolist()

        # グラフ用のデータを作成
        chart_labels = [q.split(':')[0] for q in QUESTIONS] # ["質問1", "質問2", ...]
        
        # 各評価(1-5)ごとにデータを集計
        datasets = []
        ratings = [1, 2, 3, 4, 5]
        # グラフの色
        colors = ['rgba(255, 99, 132, 0.7)', 'rgba(255, 159, 64, 0.7)', 'rgba(255, 205, 86, 0.7)', 'rgba(75, 192, 192, 0.7)', 'rgba(54, 162, 235, 0.7)']

        for i, rating in enumerate(ratings):
            rating_counts = []
            for q_num in range(1, 6):
                # DataFrameの列名を 'q1', 'q2', ... とする
                q_col = f'q{q_num}'
                # 各質問で現在の評価(rating)に一致する回答数をカウント
                # 文字列の'1'-'5'と比較
                count = (df[q_col] == str(rating)).sum() if q_col in df else 0
                rating_counts.append(int(count))

            datasets.append({
                'label': f'評価 {rating}',
                'data': rating_counts,
                'backgroundColor': colors[i % len(colors)],
            })
        
        chart_data = {
            'labels': chart_labels,
            'datasets': datasets
        }
        # テンプレートに渡すためにJSON文字列に変換
        chart_data = json.dumps(chart_data)

    return render_template('results.html', chart_data=chart_data, comments=comments, total_responses=total_responses)

if __name__ == '__main__':
    app.run(debug=True)