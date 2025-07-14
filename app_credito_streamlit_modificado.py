# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ========================
# CONFIG STREAMLIT
# ========================
st.set_page_config(page_title="Análise de Crédito", layout="wide")
st.title("📊 Projeto de Análise de Crédito")

st.markdown("""
Este aplicativo realiza uma análise exploratória e modelagem de dados de crédito,
utilizando variáveis demográficas e histórico de inadimplência.
""")

# ========================
# FUNÇÕES AUXILIARES
# ========================
def ks_stat(y_true, y_scores):
    df_ks = pd.DataFrame({'y': y_true, 'score': y_scores})
    df_ks = df_ks.sort_values('score', ascending=False)
    df_ks['cum_event'] = (df_ks['y'] == 1).cumsum() / (df_ks['y'] == 1).sum()
    df_ks['cum_nonevent'] = (df_ks['y'] == 0).cumsum() / (df_ks['y'] == 0).sum()
    return max(abs(df_ks['cum_event'] - df_ks['cum_nonevent']))

def gini(y_true, y_scores):
    return 2 * roc_auc_score(y_true, y_scores) - 1

class SubstituirNulos(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_ = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=np.number).columns:
            self.median_[col] = X[col].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col, med in self.median_.items():
            X[col].fillna(med, inplace=True)
        return X

class RemoverOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, k=1.5):
        self.k = k
        self.limits_ = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=np.number).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inf = Q1 - self.k * IQR
            limite_sup = Q3 + self.k * IQR
            self.limits_[col] = (limite_inf, limite_sup)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lim_inf, lim_sup) in self.limits_.items():
            X[col] = np.where(X[col] < lim_inf, lim_inf, X[col])
            X[col] = np.where(X[col] > lim_sup, lim_sup, X[col])
        return X

# ========================
# UPLOAD ARQUIVO
# ========================
st.subheader("📁 Upload do arquivo de dados (.ftr)")

uploaded_file = st.file_uploader("Faça upload do arquivo `.ftr` com os dados de crédito", type=["ftr"])

if uploaded_file is not None:
    try:
        df = pd.read_feather(uploaded_file)
        df = df.sample(n=min(10000, len(df)), random_state=42)
        st.success("Arquivo carregado com sucesso!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        st.stop()
else:
    st.warning("Por favor, carregue um arquivo `.ftr` para continuar.")
    st.stop()

# ========================
# PRE-PROCESSAMENTO
# ========================
categorical_feature = ['posse_de_veiculo']
num_features = ['qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']

numeric_transformer = Pipeline(steps=[
    ('imputer', SubstituirNulos()),
    ('outlier', RemoverOutliers()),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, categorical_feature)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

X = df[num_features + categorical_feature].copy()
y = df['mau']

X_transformed = pipeline.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

# ========================
# AVALIAÇÃO DO MODELO
# ========================
st.subheader("📊 Avaliação do Modelo")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Acurácia", f"{accuracy_score(y_test, y_test_pred):.4f}")
col2.metric("AUC", f"{roc_auc_score(y_test, y_test_proba):.4f}")
col3.metric("KS", f"{ks_stat(y_test, y_test_proba):.4f}")
col4.metric("Gini", f"{gini(y_test, y_test_proba):.4f}")

# ========================
# IMPORTÂNCIA DAS VARIÁVEIS
# ========================
st.subheader("📌 Importância das Variáveis")

numeric_features = num_features
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
categorical_features = ohe.get_feature_names_out(categorical_feature)

feature_names = list(numeric_features) + list(categorical_features)
importances = model.coef_[0]

coef_df = pd.DataFrame({
    'Variável': feature_names,
    'Importância (coef.)': importances
}).sort_values(by='Importância (coef.)', key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=coef_df.head(10), x='Importância (coef.)', y='Variável', palette='viridis', ax=ax)
ax.set_title("Top 10 Variáveis mais importantes")
st.pyplot(fig)

# ========================
# DOWNLOAD MODELO
# ========================
output_buffer = BytesIO()
pickle.dump(model, output_buffer)
output_buffer.seek(0)

st.subheader("📂 Baixar modelo treinado")
st.download_button(
    label="📅 Baixar arquivo `.pkl` do modelo",
    data=output_buffer,
    file_name="modelo_credito.pkl",
    mime="application/octet-stream"
) 
