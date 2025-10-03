# flask_titanic_dashboard.py
# Local Flask app to serve Titanic EDA plots as a simple HTML/CSS dashboard.
# Place this file in the same folder as your train.csv and test.csv files.

from flask import Flask, Response, render_template_string, request, send_file
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

sns.set(style="whitegrid")

app = Flask(__name__)

# Load data once on startup
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"

try:
    df = pd.read_csv(TRAIN_CSV)
except FileNotFoundError:
    df = None

# Basic feature engineering function (same as notebook)
def prepare(df_in):
    df = df_in.copy()
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    # Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    # Simplify Titles
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

if df is not None:
    df = prepare(df)

# Template (inline) - simple responsive layout + CSS
TEMPLATE = """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Titanic EDA Dashboard</title>
  <style>
    :root{--bg:#0f1724;--card:#0b1220;--accent:#7dd3fc;--muted:#94a3b8;--glass: rgba(255,255,255,0.03)}
    body{font-family:Inter,ui-sans-serif,system-ui,Segoe UI,Roboto,"Helvetica Neue",Arial; background:linear-gradient(180deg,#021022 0%, #071426 100%); color:#e6eef8; margin:0;}
    .container{max-width:1100px;margin:28px auto;padding:20px}
    header{display:flex;align-items:center;gap:16px;margin-bottom:18px}
    h1{margin:0;font-size:1.6rem}
    p.lead{margin:0;color:var(--muted)}
    .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin-top:18px}
    .card{background:var(--card);border-radius:12px;padding:12px;box-shadow:0 4px 18px rgba(2,6,23,0.6);border:1px solid rgba(255,255,255,0.03)}
    .card img{width:100%;height:auto;border-radius:8px}
    .full{grid-column:1/ -1}
    footer{color:var(--muted);margin-top:18px;font-size:0.9rem}
    .controls{display:flex;gap:8px;align-items:center}
    .btn{background:var(--glass);color:var(--accent);padding:8px 12px;border-radius:8px;border:1px solid rgba(255,255,255,0.04);cursor:pointer;text-decoration:none}
    @media(max-width:860px){.grid{grid-template-columns:1fr}.container{padding:12px}}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Titanic — EDA Dashboard</h1>
        <p class="lead">Yerel olarak servislenen grafikler. CSV'ler aynı klasörde olmalı (train.csv).</p>
      </div>
      <div style="margin-left:auto" class="controls">
        <a class="btn" href="/download/train">train.csv indir</a>
        <a class="btn" href="/download/test">test.csv indir</a>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <h3>Cinsiyete Göre Hayatta Kalma</h3>
        <img src="/plot/sex" alt="sex-plot">
      </div>

      <div class="card">
        <h3>Sınıfa Göre Hayatta Kalma</h3>
        <img src="/plot/pclass" alt="pclass-plot">
      </div>

      <div class="card">
        <h3>Yaş Dağılımı</h3>
        <img src="/plot/age" alt="age-plot">
      </div>

      <div class="card">
        <h3>Yaş & Hayatta Kalma Yoğunluğu</h3>
        <img src="/plot/age_kde" alt="age-kde-plot">
      </div>

      <div class="card">
        <h3>Aile Büyüklüğüne Göre Hayatta Kalma</h3>
        <img src="/plot/familysize" alt="family-plot">
      </div>

      <div class="card">
        <h3>Korelasyon Matrisi</h3>
        <img src="/plot/corr" alt="corr-plot">
      </div>

      <div class="card full">
        <h3>Feature Importance (RandomForest - basit)</h3>
        <img src="/plot/feat_imp" alt="feat-imp">
      </div>
    </div>

    <footer>
      Not: Eğer train.csv yoksa grafikler boş döner. Daha fazla grafik veya interaktif görselleştirme istersen Plotly/Altair ekleyebilirim.
    </footer>
  </div>
</body>
</html>
"""

# Utility: render matplotlib figure to PNG bytes

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.tight_layout(pad=1.0)
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

# Plot generators

def plot_sex(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='Sex', hue='Survived', data=df, ax=ax)
    ax.set_title('Cinsiyete Göre Hayatta Kalma')
    return fig


def plot_pclass(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax)
    ax.set_title('Yolcu Sınıfına Göre Hayatta Kalma')
    return fig


def plot_age(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df['Age'].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title('Yaş Dağılımı')
    return fig


def plot_age_kde(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.kdeplot(df.loc[df['Survived']==1,'Age'].dropna(), shade=True, label='Hayatta Kaldı', ax=ax)
    sns.kdeplot(df.loc[df['Survived']==0,'Age'].dropna(), shade=True, label='Hayatta Kalmadı', ax=ax)
    ax.legend()
    ax.set_title('Yaşa Göre Hayatta Kalma')
    return fig


def plot_familysize(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='FamilySize', hue='Survived', data=df, ax=ax)
    ax.set_title('Aile Büyüklüğüne Göre Hayatta Kalma')
    return fig


def plot_corr(df):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Korelasyon Matrisi')
    return fig


def plot_feat_imp(df):
    from sklearn.ensemble import RandomForestClassifier
    cols = ['Pclass','Sex','Age','Fare','SibSp','Parch','FamilySize','IsAlone']
    d = df.copy()

    # Kategorik -> sayısal
    d['Sex'] = d['Sex'].map({'male':0,'female':1})

    # Eksikleri doldur
    d['Age'] = d['Age'].fillna(d['Age'].median())
    d['Fare'] = d['Fare'].fillna(d['Fare'].median())

    X = d[cols]
    y = d['Survived']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    imp = clf.feature_importances_

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=imp, y=cols, ax=ax)
    ax.set_title('Feature Importance (RandomForest)')
    return fig



@app.route('/')
def index():
    return render_template_string(TEMPLATE)


@app.route('/plot/<name>')
def serve_plot(name):
    global df
    if df is None:
        return Response('train.csv bulunamadı. Lütfen aynı klasöre train.csv koyun.', status=404)

    # choose plot
    try:
        if name == 'sex':
            fig = plot_sex(df)
        elif name == 'pclass':
            fig = plot_pclass(df)
        elif name == 'age':
            fig = plot_age(df)
        elif name == 'age_kde':
            fig = plot_age_kde(df)
        elif name == 'familysize':
            fig = plot_familysize(df)
        elif name == 'corr':
            fig = plot_corr(df)
        elif name == 'feat_imp':
            fig = plot_feat_imp(df)
        else:
            return Response('Plot bulunamadı', status=404)

        buf = fig_to_png_bytes(fig)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return Response(f'Hata: {e}', status=500)


@app.route('/download/<which>')
def download_csv(which):
    if which == 'train':
        path = TRAIN_CSV
    else:
        path = TEST_CSV
    try:
        return send_file(path, as_attachment=True)
    except Exception:
        return Response('Dosya bulunamadı', status=404)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
