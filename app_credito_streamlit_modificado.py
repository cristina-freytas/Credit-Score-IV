import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
import matplotlib.pyplot as plt

# Configurações Streamlit
st.set_page_config(page_title="Análise de Crédito", layout="wide")
st.title("📊 Projeto de Análise de Crédito")
st.markdown("""
Este aplicativo realiza uma análise exploratória e modelagem de dados de crédito,
utilizando variáveis demográficas e histórico de inadimplência.
""")

# Funções auxiliares

def ks_stat(y_true, y_scores):
    df_ks = pd.DataFrame({'y': y_true, 'score': y_scores})
    df_ks = df_ks.sort_values('score', ascending=False)
    df_ks['cum_event'] = (df_ks['y'] == 1).cumsum() / (df_ks['y'] == 1).sum()
    df_ks['cum_nonevent'] = (df_ks['y'] == 0).cumsum() / (df_ks['y'] == 0).sum()
    return max(abs(df_ks['cum_event'] - df_ks['cum_nonevent']))

def gini(y_true, y_scores):
    # Implementa gini usando ROC-AUC (sem sklearn)
    from sklearn.metrics import roc_auc_score
    return 2 * roc_auc_score(y_true, y_scores) - 1

# Upload do arquivo
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

# Pré-processamento manual (sem pipeline)

categorical_feature = 'posse_de_veiculo'
num_features = ['qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']

# Substituir nulos nas numéricas pela mediana
for col in num_features:
    med = df[col].median()
    df[col] = df[col].fillna(med)

# Tratar outliers (limitar por IQR)
def limitar_outliers(series, k=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k*IQR
    upper = Q3 + k*IQR
    return series.clip(lower, upper)

for col in num_features:
    df[col] = limitar_outliers(df[col])

# One-hot encode simples para 'posse_de_veiculo' (supondo binária)
df['posse_de_veiculo'] = df[categorical_feature].map({'sim':1, 'não':0}).fillna(0).astype(int)

# Criar X e y
X = df[num_features + ['posse_de_veiculo']]
y = df['mau']

# Escalar manualmente (z-score)
for col in num_features:
    mean = X[col].mean()
    std = X[col].std()
    X[col] = (X[col] - mean) / std

# Dividir treino/teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Treinar modelo de regressão logística simples
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Previsões
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

# Avaliação
from sklearn.metrics import accuracy_score, roc_auc_score

st.subheader("📊 Avaliação do Modelo")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Acurácia", f"{accuracy_score(y_test, y_test_pred):.4f}")
col2.metric("AUC", f"{roc_auc_score(y_test, y_test_proba):.4f}")
col3.metric("KS", f"{ks_stat(y_test, y_test_proba):.4f}")
col4.metric("Gini", f"{gini(y_test, y_test_proba):.4f}")

# Importância das variáveis (coeficientes)
st.subheader("📌 Importância das Variáveis")

feature_names = num_features + ['posse_de_veiculo']
importances = model.coef_[0]

coef_df = pd.DataFrame({
    'Variável': feature_names,
    'Importância (coef.)': importances
}).sort_values(by='Importância (coef.)', key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(coef_df['Variável'], coef_df['Importância (coef.)'], color='skyblue')
ax.invert_yaxis()
ax.set_title("Top Variáveis mais importantes")
st.pyplot(fig)

# Download do modelo
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

