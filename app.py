import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime
import pandas as pd
import numpy as np


# --- PyMC関連のインポート ---
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import warnings

# FutureWarningを無視
warnings.simplefilter(action='ignore', category=FutureWarning)

# Flaskアプリケーションの初期化
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # flashメッセージのために必要

# --- アンケート機能関連の定数 ---
DATA_FOLDER = 'data'
os.makedirs(DATA_FOLDER, exist_ok=True)
QUESTIONS = [
    "質問1: この研究の新規性は評価できますか？",
    "質問2: 提案手法の有効性は明確に示されていると感じますか？",
    "質問3: 論文の構成や説明は分かりやすいですか？",
    "質問4: 実世界への応用可能性は高いと感じますか？",
    "質問5: この研究は、関連分野に貢献すると思いますか？"
]

# --- 売上予測モデル関連のグローバル設定 ---
MODEL_FILE = "static/sales_model_trace.nc"
N_ORDER_YEARLY = 10  # モデル定義で使用したフーリエ級数の次数

# ==============================================================================
# 【変更点】CSVを読み込む代わりに、学習時の情報を定数として保持
# ==============================================================================
# 以下の値は、事前に 'train_store1.csv' から計算したものです。
# これにより、アプリケーション実行時にCSVファイルを読み込む必要がなくなります。
N_TRAIN = 687  # フィルタリング後の学習データ総数 (例)
LAST_TRAIN_DATE = pd.to_datetime('2015-01-29')  # 学習データの最終日 (例)
INITIAL_SALES_LOG = 8.8  # 学習データの最初のlog(Sales) (例)
# ==============================================================================


# モデル構築関数
def build_model():
    """PyMCモデルの構造を定義する関数。ダミーデータで構造のみを定義する。"""
    coords = {"time": [0], "dayofweek_state": np.arange(7),
              "yearly_fourier": np.arange(2 * N_ORDER_YEARLY)}

    with pm.Model(coords=coords) as model:
        # Data containers (サイズ1のダミーで初期化)
        promo_data = pm.Data('promo_data', [0], mutable=True)
        dayofweek_idx = pm.Data('dayofweek_idx', [0], mutable=True)
        school_holiday_data = pm.Data('school_holiday_data', [0], mutable=True)
        state_holiday_a_data = pm.Data('state_holiday_a_data', [0], mutable=True)
        state_holiday_b_data = pm.Data('state_holiday_b_data', [0], mutable=True)
        state_holiday_c_data = pm.Data('state_holiday_c_data', [0], mutable=True)
        time_year_data = pm.Data('time_year_data', [0.0], mutable=True)
        
        # Trend (初期値は定数から設定)
        sigma_trend = pm.HalfNormal('sigma_trend', sigma=0.5)
        trend_rw = pm.GaussianRandomWalk('trend_rw', sigma=sigma_trend, dims="time", 
                                       init_dist=pm.Normal.dist(mu=INITIAL_SALES_LOG, sigma=1))
        
        # Seasonality
        seasonality_weekly = pm.Normal('seasonality_weekly', mu=0, sigma=1.0, dims="dayofweek_state")
        yearly_beta = pm.Normal('yearly_beta', mu=0, sigma=1.0, dims="yearly_fourier")
        fourier_features_yearly = pt.concatenate(
            [pt.cos(2 * np.pi * (k + 1) * time_year_data)[:, None] for k in range(N_ORDER_YEARLY)] +
            [pt.sin(2 * np.pi * (k + 1) * time_year_data)[:, None] for k in range(N_ORDER_YEARLY)], axis=1
        )
        seasonality_yearly = pm.math.dot(fourier_features_yearly, yearly_beta)
        
        # Regressors
        beta_promo = pm.Normal('beta_promo', mu=0, sigma=1.0)
        beta_school = pm.Normal('beta_school', mu=0, sigma=1.0)
        beta_state_a = pm.Normal('beta_state_a', mu=0, sigma=1.0)
        beta_state_b = pm.Normal('beta_state_b', mu=0, sigma=1.0)
        beta_state_c = pm.Normal('beta_state_c', mu=0, sigma=1.0)

        mu = (trend_rw + seasonality_weekly[dayofweek_idx] + seasonality_yearly +
              beta_promo * promo_data + beta_school * school_holiday_data +
              beta_state_a * state_holiday_a_data + beta_state_b * state_holiday_b_data +
              beta_state_c * state_holiday_c_data)
        
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.5)
        # 観測データはダミーを設定。予測時には利用されない。
        sales_log_lik = pm.Normal('sales_log_lik', mu=mu, sigma=sigma_obs, observed=[INITIAL_SALES_LOG], dims="time")
    return model

# モデルとトレースをグローバルにロード
pymc_model = build_model()
try:
    trace = az.from_netcdf(MODEL_FILE)
    print(f"学習済みモデル {MODEL_FILE} を正常に読み込みました。")
except FileNotFoundError:
    print(f"警告: モデルファイル {MODEL_FILE} が見つかりません。予測機能は利用できません。")
    trace = None
    pymc_model = None


@app.route('/', methods=['GET', 'POST'])
def index():
    """研究概要ページ 兼 予測実行ページ"""
    research_summary = """
    研究論文は、階層的な販売予測における精度向上と一貫性の確保に焦点を当てています。具体的には、限られた履歴データ、有用な情報の不足、および階層的制約の考慮不足といった課題に対処するため、特徴の分離抽出と再調整に基づく新しい予測手法を提案しています。提案されたフレームワークは、類似製品データを用いた事前学習と微調整、時系列依存および非依存特徴を独立して抽出するモジュール、そして階層的な一貫性を保証する調整モジュールと損失関数を組み合わせています。実世界の小売データを用いた実験により、この手法が既存の主流な予測モデルと比較して優れた性能を示し、供給チェーン管理における意思決定を効果的に支援することが実証されました。
    """
    
    prediction_result = None
    input_data_dict = None

    if request.method == 'POST':
        if pymc_model is None or trace is None:
            flash('予測モデルがロードされていないため、予測を実行できません。', 'danger')
        else:
            try:
                # フォームからデータを取得
                date_str = request.form.get('prediction_date')
                promo = int(request.form.get('promo', 0))
                school_holiday = int(request.form.get('school_holiday', 0))
                state_holiday = request.form.get('state_holiday', '0')
                
                prediction_date = pd.to_datetime(date_str)

                # 入力データを保存
                input_data_dict = {
                    "date": prediction_date.strftime('%Y年%m月%d日'),
                    "promo": "あり" if promo == 1 else "なし",
                    "school_holiday": "あり" if school_holiday == 1 else "なし",
                    "state_holiday": {"0": "なし", "a": "祝日A", "b": "祝日B", "c": "祝日C"}[state_holiday]
                }
                
                # 【変更点】定数を使用して予測用データを作成
                new_time_delta = (prediction_date - LAST_TRAIN_DATE).days
                if new_time_delta <= 0:
                    flash(f'予測日は学習データの最終日 ({LAST_TRAIN_DATE.strftime("%Y-%m-%d")}) より後の日付を選択してください。', 'warning')
                    return render_template('index.html', summary=research_summary, prediction_result=None, input_data=None)

                full_prediction_range = pd.date_range(start=LAST_TRAIN_DATE + pd.Timedelta(days=1), periods=new_time_delta, freq='D')
                
                # 特徴量の作成
                time_coords_pred = np.arange(N_TRAIN, N_TRAIN + new_time_delta)
                time_year_pred = (full_prediction_range.dayofyear / 365.25).values
                dayofweek_pred = (full_prediction_range.dayofweek).values # 0-6
                
                promo_pred = np.zeros(new_time_delta); promo_pred[-1] = promo
                school_holiday_pred = np.zeros(new_time_delta); school_holiday_pred[-1] = school_holiday
                state_holiday_a_pred = np.zeros(new_time_delta); state_holiday_a_pred[-1] = 1 if state_holiday == 'a' else 0
                state_holiday_b_pred = np.zeros(new_time_delta); state_holiday_b_pred[-1] = 1 if state_holiday == 'b' else 0
                state_holiday_c_pred = np.zeros(new_time_delta); state_holiday_c_pred[-1] = 1 if state_holiday == 'c' else 0

                # モデルに新しいデータをセット
                with pymc_model:
                    pm.set_data({
                        'promo_data': promo_pred,
                        'dayofweek_idx': dayofweek_pred,
                        'school_holiday_data': school_holiday_pred,
                        'state_holiday_a_data': state_holiday_a_pred,
                        'state_holiday_b_data': state_holiday_b_pred,
                        'state_holiday_c_data': state_holiday_c_pred,
                        'time_year_data': time_year_pred,
                    })

                    pred = pm.sample_posterior_predictive(
                        trace,
                        var_names=["sales_log_lik"],
                        random_seed=42,
                        extend_inferencedata=False
                    )

                # 最後の時点の予測値のみを取得して結果を計算
                last_day_preds = np.exp(pred.posterior_predictive['sales_log_lik'].sel(time=time_coords_pred[-1]).values.flatten())

                prediction_result = {
                    'median': f"{np.median(last_day_preds):,.0f}",
                    'hdi_3': f"{az.hdi(last_day_preds, hdi_prob=0.94)[0]:,.0f}",
                    'hdi_97': f"{az.hdi(last_day_preds, hdi_prob=0.94)[1]:,.0f}",
                }

            except Exception as e:
                flash(f'予測中にエラーが発生しました: {e}', 'danger')

    return render_template('index.html', summary=research_summary, prediction_result=prediction_result, input_data=input_data_dict)

# --- 以下、アンケートと結果表示のルート (変更なし) ---
@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """アンケートページ"""
    if request.method == 'POST':
        data = { 'q1': request.form.get('q1'), 'q2': request.form.get('q2'), 'q3': request.form.get('q3'), 'q4': request.form.get('q4'), 'q5': request.form.get('q5'), 'comment': request.form.get('comment'), 'timestamp': datetime.now().isoformat() }
        if not all([data['q1'], data['q2'], data['q3'], data['q4'], data['q5']]):
            flash('すべての評価質問に回答してください。', 'danger')
            return render_template('survey.html', questions=QUESTIONS, form_data=data)
        
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
    for filename in sorted(os.listdir(DATA_FOLDER)):
        if filename.endswith('.json'):
            filepath = os.path.join(DATA_FOLDER, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    all_data.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: Skipping corrupted or empty file: {filename}")

    chart_data, comments, total_responses = None, [], len(all_data)
    if all_data:
        df = pd.DataFrame(all_data)
        comments = df['comment'].dropna().tolist()
        chart_labels = [q.split(':')[0] for q in QUESTIONS]
        datasets, ratings = [], [1, 2, 3, 4, 5]
        colors = ['rgba(255, 99, 132, 0.7)', 'rgba(255, 159, 64, 0.7)', 'rgba(255, 205, 86, 0.7)', 'rgba(75, 192, 192, 0.7)', 'rgba(54, 162, 235, 0.7)']
        for i, rating in enumerate(ratings):
            rating_counts = [(df[f'q{q_num}'] == str(rating)).sum() if f'q{q_num}' in df else 0 for q_num in range(1, 6)]
            datasets.append({'label': f'評価 {rating}', 'data': [int(c) for c in rating_counts], 'backgroundColor': colors[i % len(colors)]})
        chart_data = json.dumps({'labels': chart_labels, 'datasets': datasets})
    return render_template('results.html', chart_data=chart_data, comments=comments, total_responses=total_responses)

if __name__ == '__main__':
    app.run(debug=True)