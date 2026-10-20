from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Demonstração de Churn | Random Forest",
    page_icon="🌲",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parent / "churn_dataset.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")


@st.cache_resource
def train_model(data):
    model_data = data.drop(columns=["cliente_id"]).copy()
    categorical = ["tipo_contrato", "plano_internet", "metodo_pagamento"]
    binary = ["tem_suporte_tecnico", "tem_backup_online"]
    model_data[binary] = model_data[binary].apply(lambda col: col.map({"Não": 0, "Sim": 1}))
    model_data = pd.get_dummies(model_data, columns=categorical, dtype=int)
    X = model_data.drop(columns="Churn")
    y = model_data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=6, max_leaf_nodes=5
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "matrix": confusion_matrix(y_test, predictions),
        "y_test": y_test,
        "predictions": predictions,
        "features": X.columns,
    }
    return model, metrics


def prepare_client(values, columns):
    row = pd.DataFrame([values])
    row["tem_suporte_tecnico"] = 1 if row["tem_suporte_tecnico"].iloc[0] == "Sim" else 0
    row["tem_backup_online"] = 1 if row["tem_backup_online"].iloc[0] == "Sim" else 0
    row = pd.get_dummies(
        row,
        columns=["tipo_contrato", "plano_internet", "metodo_pagamento"],
        dtype=int,
    )
    return row.reindex(columns=columns, fill_value=0)


try:
    df = load_data()
except FileNotFoundError:
    st.error("Não encontrei `churn_dataset.csv`. Coloque o arquivo na mesma pasta de `app.py`.")
    st.stop()

model, results = train_model(df)
accuracy = results["accuracy"]
recall = results["recall"]

st.title("🌲 Previsão de cancelamento de clientes")
st.markdown(
    "Demonstração de como um modelo **Random Forest** pode ajudar o time de retenção "
    "a identificar clientes com maior propensão a cancelar. Os resultados abaixo são "
    "calculados em uma amostra de teste que não foi usada no treinamento."
)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Acurácia no teste", f"{accuracy:.0%}")
metric2.metric("Recall de cancelamento", f"{recall:.0%}")
metric3.metric("Clientes no teste", f"{len(results['y_test']):,}".replace(",", "."))

st.caption(
    "Acurácia: proporção de previsões corretas. Recall: proporção dos clientes que "
    "cancelaram e foram identificados pelo modelo. Um recall maior ajuda a reduzir "
    "cancelamentos não detectados, mas pode gerar abordagens a clientes que permaneceriam."
)

overview, insights, simulator = st.tabs(
    ["Desempenho do modelo", "Insights da análise", "Simular cliente"]
)

with overview:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Matriz de confusão")
        cm = results["matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Real: permaneceu", "Real: cancelou"],
            columns=["Previsto: permaneceu", "Previsto: cancelou"],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(
            f"O modelo identificou {cm[1, 1]} dos {cm[1].sum()} cancelamentos no teste "
            f"e deixou passar {cm[1, 0]}."
        )
    with right:
        st.subheader("Importância das variáveis")
        importance = pd.DataFrame(
            {"Variável": results["features"], "Importância": model.feature_importances_}
        ).sort_values("Importância", ascending=False).head(8)
        fig = px.bar(
            importance.sort_values("Importância"),
            x="Importância",
            y="Variável",
            orientation="h",
            color="Importância",
            color_continuous_scale="Teal",
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Importância no modelo")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Importância indica quanto cada variável contribui para as decisões da floresta; não prova causalidade.")

with insights:
    contract_rates = (
        df.groupby("tipo_contrato", as_index=False)["Churn"].mean().assign(Churn=lambda x: x["Churn"] * 100)
    )
    tenure_data = df.assign(
        Faixa=pd.cut(
            df["tempo_contrato_meses"],
            bins=[0, 12, 24, 36, float("inf")],
            labels=["1–12 meses", "13–24 meses", "25–36 meses", "37+ meses"],
            include_lowest=True,
        )
    )
    tenure_rates = tenure_data.groupby("Faixa", observed=False, as_index=False)["Churn"].mean()
    tenure_rates["Churn"] *= 100
    monthly_rate = df.loc[df["tipo_contrato"] == "Mensal", "Churn"].mean()
    non_monthly_rate = df.loc[df["tipo_contrato"] != "Mensal", "Churn"].mean()
    short_rate = df.loc[df["tempo_contrato_meses"] < 24, "Churn"].mean()
    long_rate = df.loc[df["tempo_contrato_meses"] >= 24, "Churn"].mean()

    a, b = st.columns(2)
    with a:
        st.subheader("Cancelamento por tipo de contrato")
        fig = px.bar(contract_rates, x="tipo_contrato", y="Churn", text_auto=".1f", labels={"tipo_contrato": "Contrato", "Churn": "Cancelamento (%)"}, color="tipo_contrato")
        fig.update_layout(showlegend=False, yaxis_range=[0, 70])
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Cancelamento por tempo de permanência")
        fig = px.bar(tenure_rates, x="Faixa", y="Churn", text_auto=".1f", labels={"Faixa": "Tempo de contrato", "Churn": "Cancelamento (%)"}, color="Churn", color_continuous_scale="Oranges")
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 70])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"- **Contrato mensal:** {monthly_rate:.1%} de cancelamento, contra {non_monthly_rate:.1%} "
        "nos contratos anuais e bienais combinados.\n"
        f"- **Relacionamento mais curto:** clientes com menos de 24 meses têm cancelamento de "
        f"{short_rate:.1%}; a partir de 24 meses, a taxa é {long_rate:.1%}.\n"
        "- Esses padrões podem orientar ações de retenção, como contato antecipado e revisão "
        "da oferta para clientes mensais e de menor permanência. São associações observadas "
        "neste conjunto de dados, não evidência de causa."
    )

with simulator:
    st.subheader("Explore uma previsão")
    st.write("Ajuste o perfil e veja a estimativa do modelo para um cliente fictício.")
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tempo de contrato (meses)", 1, 71, 12)
        monthly = c2.slider("Valor mensal (R$)", 30.0, 129.97, 79.0, step=0.5)
        calls = c3.slider("Chamadas ao suporte", 0, 8, 1)
        contract = c1.selectbox("Tipo de contrato", sorted(df["tipo_contrato"].unique()))
        internet = c2.selectbox("Plano de internet", sorted(df["plano_internet"].unique()))
        payment = c3.selectbox("Método de pagamento", sorted(df["metodo_pagamento"].unique()))
        support = c1.selectbox("Suporte técnico", ["Não", "Sim"])
        backup = c2.selectbox("Backup online", ["Não", "Sim"])
        products = c3.slider("Produtos contratados", 1, 5, 3)
        delay = c1.slider("Atraso de pagamento (dias)", 0, 10, 2)
        submitted = st.form_submit_button("Estimar risco")

    if submitted:
        client = {
            "tempo_contrato_meses": tenure,
            "valor_mensal": monthly,
            "num_chamadas_suporte": calls,
            "tem_suporte_tecnico": support,
            "tem_backup_online": backup,
            "num_produtos_contratados": products,
            "atraso_pagamento_dias": delay,
            "tipo_contrato": contract,
            "plano_internet": internet,
            "metodo_pagamento": payment,
        }
        input_row = prepare_client(client, results["features"])
        probability = model.predict_proba(input_row)[0, 1]
        st.metric("Probabilidade estimada de cancelamento", f"{probability:.1%}")
        st.progress(float(probability))
        if probability >= 0.5:
            st.warning("Neste perfil, o modelo sinaliza possível cancelamento. Considere avaliar uma ação de retenção.")
        else:
            st.success("Neste perfil, o modelo estima menor propensão a cancelar.")
        st.caption("A probabilidade é uma estimativa do modelo para demonstração e não garante o comportamento individual.")

st.divider()
st.caption(
    f"Base: {len(df):,} clientes. Modelo treinado com 80% dos dados; avaliação em 20% "
    "com estratificação e random_state=42. Resultados refletem esta base e esta divisão."
    .replace(",", ".")
)
