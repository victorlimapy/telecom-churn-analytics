# Dashboard de Churn de Clientes

Dashboard interativo para análise e previsão de cancelamento de clientes, desenvolvido com Python, Pandas, Scikit-learn, Streamlit e Plotly.

O projeto carrega uma base local de clientes, apresenta métricas e visualizações de churn e permite simular o risco estimado para um perfil fictício. O modelo utilizado é um classificador Random Forest.

---

## Funcionalidades

- Dashboard interativo com Streamlit
- Métricas de acurácia, recall e quantidade de clientes no conjunto de teste
- Matriz de confusão
- Gráfico de importância das variáveis do modelo
- Análise de cancelamento por tipo de contrato e tempo de permanência
- Simulador de probabilidade estimada de cancelamento para um cliente
- Análise exploratória da base no notebook `analise.ipynb`

---

## Tecnologias Utilizadas

### Linguagem e análise

- Python 3.x
- Pandas
- Scikit-learn

### Aplicação e visualização

- Streamlit
- Plotly

---

## Estrutura do Projeto

```text
Treino_Curso_Basico-Kaggle/
│
├── app.py
├── analise.ipynb
├── churn_dataset.csv
├── requirements.txt
└── README.md
```

---

## Instalação

### 1. Criar um ambiente virtual

```bash
python -m venv venv
```

### 2. Ativar o ambiente virtual

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Instalar as dependências

```bash
python -m pip install -r requirements.txt
```

### 4. Executar o dashboard

```bash
streamlit run app.py
```

O arquivo `churn_dataset.csv` deve permanecer na mesma pasta de `app.py`. O endereço local do dashboard será exibido no terminal e aberto no navegador pelo Streamlit.

---

## Dados e Modelo

A base incluída tem 3.500 registros e 12 colunas. A variável `Churn` indica o resultado observado: `1` para cancelamento e `0` para permanência. A coluna `cliente_id` é removida antes do treinamento.

As variáveis categóricas são convertidas em colunas indicadoras e os dados são divididos em treino e teste na proporção de 80% e 20%, com estratificação e `random_state=42`. O modelo Random Forest utiliza 100 árvores.

O resultado serve como demonstração com esta base e configuração. Métricas e importância das variáveis descrevem o comportamento observado no conjunto de dados; não demonstram que uma variável causa cancelamentos nem garantem a previsão de um cliente individual.

---

## Métricas Disponíveis

| Indicador | Descrição |
|------------|------------|
| Acurácia | Proporção de previsões corretas no conjunto de teste |
| Recall de cancelamento | Proporção dos clientes que cancelaram e foram identificados pelo modelo |
| Clientes no teste | Quantidade de registros usados na avaliação |
| Matriz de confusão | Comparação entre os resultados reais e previstos para cada classe |

---

## Visualizações

### Importância das variáveis

Apresenta as oito variáveis com maior importância atribuída pelo Random Forest. Essa medida indica contribuição para as decisões do modelo, sem estabelecer causalidade.

### Cancelamento por contrato

Compara a taxa de churn entre os tipos de contrato presentes na base.

### Cancelamento por tempo de permanência

Agrupa os clientes por faixas de meses de contrato e compara as taxas observadas.

---

## Principais Conceitos Aplicados

### Pandas

Manipulação dos dados, criação de variáveis indicadoras e cálculo de taxas agrupadas:

```python
contract_rates = (
    df.groupby("tipo_contrato", as_index=False)["Churn"]
      .mean()
)
```

### Scikit-learn

Divisão estratificada dos dados, treinamento do classificador e avaliação:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(
    n_estimators=100, random_state=42, max_depth=6, max_leaf_nodes=5
)
```

### Streamlit e Plotly

Construção da interface e dos gráficos interativos:

```python
st.metric("Acurácia no teste", f"{accuracy:.0%}")
st.plotly_chart(fig, use_container_width=True)
```
Demo Online: https://telecom-churn-analytics-01.streamlit.app/
---

## Habilidades Demonstradas

- Python
- Análise e preparação de dados com Pandas
- Classificação com Scikit-learn
- Avaliação de modelos de machine learning
- Construção de dashboards com Streamlit
- Visualização interativa com Plotly
- Análise exploratória de dados

---

## Autor

**Victor Emanoel de Souza Lima**

GitHub: [victorlimapy](https://github.com/victorlimapy)  
LinkedIn: [victorlimapy](https://www.linkedin.com/in/victorlimapy/)
