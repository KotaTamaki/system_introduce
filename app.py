import os
import json
from functools import wraps # デコレータ作成のためにインポート
from flask import Flask, render_template, request, redirect, url_for, Response,flash
from datetime import datetime
import pandas as pd
import numpy as np
import csv

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
# 認証情報（自由に変更してください）
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password123'

QUESTIONS = [
    "質問1: この研究の新規性は評価できますか？",
    "質問2: 提案手法の有効性は明確に示されていると感じますか？",
    "質問3: 論文の構成や説明は分かりやすいですか？",
    "質問4: 実世界への応用可能性は高いと感じますか？",
    "質問5: この研究は、関連分野に貢献すると思いますか？"
]

CSV_FILE = './data/result.csv'

# --- 売上予測モデル関連のグローバル設定 ---
MODEL_FILE = "static/sales_model_trace.nc"
N_ORDER_YEARLY = 10  # モデル定義で使用したフーリエ級数の次数

# 学習時の情報
N_TRAIN = 687
LAST_TRAIN_DATE = pd.to_datetime('2015-01-29')
INITIAL_SALES_LOG = 8.8

# ==============================================================================
# 【変更点 1】モデルとトレースをグローバルスコープでNoneとして初期化
# アプリケーション起動時には読み込まず、メモリ上に配置しない。
# ==============================================================================
pymc_model = None
trace = None
# ==============================================================================

def check_auth(username, password):
    """ユーザー名とパスワードが正しいかチェックする"""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def authenticate():
    """認証を要求するレスポンスを返す"""
    return Response(
    '認証が必要です。', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    """認証を要求するデコレータ"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# モデル構築関数 (この関数自体に変更はありません)
def build_model():
    """PyMCモデルの構造を定義する関数。時間の長さを可変にする。"""
    coords = {
        "dayofweek_state": np.arange(7),
        "yearly_fourier": np.arange(2 * N_ORDER_YEARLY)
    }

    with pm.Model(coords=coords) as model:
        time_coords = pm.MutableData("time_coords", [0], dims="time")
        promo_data = pm.MutableData('promo_data', [0], dims="time")
        dayofweek_idx = pm.MutableData('dayofweek_idx', [0], dims="time")
        school_holiday_data = pm.MutableData('school_holiday_data', [0], dims="time")
        state_holiday_a_data = pm.MutableData('state_holiday_a_data', [0], dims="time")
        state_holiday_b_data = pm.MutableData('state_holiday_b_data', [0], dims="time")
        state_holiday_c_data = pm.MutableData('state_holiday_c_data', [0], dims="time")
        time_year_data = pm.MutableData('time_year_data', [0.0], dims="time")

        sigma_trend = pm.HalfNormal('sigma_trend', sigma=0.5)
        trend_rw = pm.GaussianRandomWalk('trend_rw', sigma=sigma_trend, dims="time",
                                       init_dist=pm.Normal.dist(mu=INITIAL_SALES_LOG, sigma=1))

        seasonality_weekly = pm.Normal('seasonality_weekly', mu=0, sigma=1.0, dims="dayofweek_state")
        yearly_beta = pm.Normal('yearly_beta', mu=0, sigma=1.0, dims="yearly_fourier")
        fourier_features_yearly = pt.concatenate(
            [pt.cos(2 * np.pi * (k + 1) * time_year_data)[:, None] for k in range(N_ORDER_YEARLY)] +
            [pt.sin(2 * np.pi * (k + 1) * time_year_data)[:, None] for k in range(N_ORDER_YEARLY)], axis=1
        )
        seasonality_yearly = pm.math.dot(fourier_features_yearly, yearly_beta)

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
        sales_log_lik = pm.Normal('sales_log_lik', mu=mu, sigma=sigma_obs, observed=[INITIAL_SALES_LOG], dims="time")

    return model

# ==============================================================================
# 【変更点 2】モデルを必要に応じて読み込む関数を定義
# ==============================================================================
def load_model_if_needed():
    """
    モデルとトレースがメモリにロードされていなければ、ファイルから読み込む。
    2回目以降の呼び出しでは何もしない。
    """
    global pymc_model, trace
    # 既にロード済みの場合は、即座に処理を終了
    if pymc_model is not None and trace is not None:
        return

    print("--- 初回アクセス or モデル未読込のため、モデルをファイルから読み込みます ---")
    try:
        # グローバル変数にモデルの構造と学習済みトレースを格納する
        pymc_model = build_model()
        trace = az.from_netcdf(MODEL_FILE)
        print(f"--- 学習済みモデル {MODEL_FILE} を正常に読み込みました ---")
    except FileNotFoundError:
        # ファイルが見つからない場合は、変数をNoneのままにしておく
        pymc_model = None
        trace = None
        print(f"--- 警告: モデルファイル {MODEL_FILE} が見つかりません。予測機能は利用できません。 ---")
    except Exception as e:
        pymc_model = None
        trace = None
        print(f"--- モデル読み込み中に予期せぬエラーが発生しました: {e} ---")
# ==============================================================================


@app.route('/', methods=['GET', 'POST'])
def index():
    """研究概要ページ 兼 予測実行ページ"""
    with open("research_summary.txt", "r", encoding="utf-8") as file:
        research_summary = file.read()
    prediction_result = None
    input_data_dict = None

    if request.method == 'POST':
        # ==============================================================================
        # 【変更点 3】予測実行の直前にモデル読み込み関数を呼び出す
        # ==============================================================================
        load_model_if_needed()

        if pymc_model is None or trace is None:
            flash('予測モデルがロードされていない、または読み込みに失敗したため、予測を実行できません。', 'danger')
        else:
            try:
                date_str = request.form.get('prediction_date')
                promo = int(request.form.get('promo', 0))
                school_holiday = int(request.form.get('school_holiday', 0))
                state_holiday = request.form.get('state_holiday', '0')

                prediction_date = pd.to_datetime(date_str)

                input_data_dict = {
                    "date": prediction_date.strftime('%Y年%m月%d日'),
                    "promo": "あり" if promo == 1 else "なし",
                    "school_holiday": "あり" if school_holiday == 1 else "なし",
                    "state_holiday": {"0": "なし", "a": "祝日A", "b": "祝日B", "c": "祝日C"}[state_holiday]
                }

                new_time_delta = (prediction_date - LAST_TRAIN_DATE).days
                if new_time_delta <= 0:
                    flash(f'予測日は学習データの最終日 ({LAST_TRAIN_DATE.strftime("%Y-%m-%d")}) より後の日付を選択してください。', 'warning')
                    return render_template('index.html', summary=research_summary, prediction_result=None, input_data=None)

                full_prediction_range = pd.date_range(start=LAST_TRAIN_DATE + pd.Timedelta(days=1), periods=new_time_delta, freq='D')

                time_coords_pred = np.arange(N_TRAIN, N_TRAIN + new_time_delta)
                time_year_pred = (full_prediction_range.dayofyear / 365.25).values
                dayofweek_pred = (full_prediction_range.dayofweek).values

                promo_pred = np.zeros(new_time_delta, dtype=np.int32); promo_pred[-1] = promo
                school_holiday_pred = np.zeros(new_time_delta, dtype=np.int32); school_holiday_pred[-1] = school_holiday
                state_holiday_a_pred = np.zeros(new_time_delta, dtype=np.int32); state_holiday_a_pred[-1] = 1 if state_holiday == 'a' else 0
                state_holiday_b_pred = np.zeros(new_time_delta, dtype=np.int32); state_holiday_b_pred[-1] = 1 if state_holiday == 'b' else 0
                state_holiday_c_pred = np.zeros(new_time_delta, dtype=np.int32); state_holiday_c_pred[-1] = 1 if state_holiday == 'c' else 0

                with pymc_model:
                    pm.set_data({
                        'time_coords': time_coords_pred,
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

                last_day_preds = np.exp(pred.posterior_predictive['sales_log_lik'].isel(time=-1).values.flatten())

                prediction_result = {
                    'median': f"{np.median(last_day_preds):,.0f}",
                    'hdi_3': f"{az.hdi(last_day_preds, hdi_prob=0.94)[0]:,.0f}",
                    'hdi_97': f"{az.hdi(last_day_preds, hdi_prob=0.94)[1]:,.0f}",
                }

            except Exception as e:
                flash(f'予測中にエラーが発生しました: {e}', 'danger')

    return render_template('index.html', summary=research_summary, prediction_result=prediction_result, input_data=input_data_dict)


# --- 以下、アンケートと結果表示のルート (変更なし) ---

def save_to_csv(data):
    """
    アンケート結果をCSVファイルに保存する関数。
    """
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        if not file_exists:
            # ヘッダーに「コメント」を追加
            header = ['Timestamp'] + [f'質問{i+1}' for i in range(len(QUESTIONS))] + ['コメント']
            writer.writerow(header)
        
        writer.writerow(data)
        
        
@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """
    アンケートの表示（GET）と回答の処理（POST）を行う
    """
    # POSTリクエスト（フォームが送信された）の場合
    if request.method == 'POST':
        # フォームからデータを取得
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        answers = [request.form.get(f'q{i}') for i in range(1,len(QUESTIONS)+1)]
        comment = request.form.get('comment')
        result_data = [timestamp] + answers + [comment]

        # データをCSVに保存
        save_to_csv(result_data)

        # メッセージを表示し、メインページ（この場合は自分自身）にリダイレクト
        flash('ご回答ありがとうございました！')
        return redirect(url_for('index'))

    # GETリクエスト（ページを最初に表示する）の場合
    # survey.html を表示する
    return render_template('survey.html', questions=QUESTIONS)


@app.route('/results')
@requires_auth  # この行で認証を必須にする
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