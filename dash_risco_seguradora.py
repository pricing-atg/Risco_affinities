import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ========== AUTENTICAÇÃO ==========
SENHA_CORRETA = "pricing2025"

if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

def validar_senha():
    if st.session_state["senha_digitada"] == SENHA_CORRETA:
        st.session_state["autenticado"] = True
    else:
        st.session_state["erro_autenticacao"] = True

if not st.session_state["autenticado"]:
    st.title("🔒 Acesso Restrito")
    st.text_input("Digite a senha:", type="password", key="senha_digitada", on_change=validar_senha)
    if st.session_state.get("erro_autenticacao", False):
        st.error("Senha incorreta.")
        st.session_state["erro_autenticacao"] = False
    st.stop()

st.markdown("""
<style>
.stMultiSelect [data-baseweb="tag"] {
    background-color: #1f77b4 !important;
    color: white !important;
}
.stMultiSelect [data-baseweb="tag"] .remove-button {
    color: white !important;
}
.stMultiSelect [role="option"]:hover {
    background-color: #d0e3f3 !important;
}
.stDateInput, .stDateInput input {
    direction: ltr;
    text-align: left;
}
.st-emotion-cache-1wmy9hl {
    font-family: "Segoe UI", sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ========== APP ==========
st.set_page_config(page_title="Dashboard de Similaridade entre Seguradoras", layout="wide")

# 1. Carrega base original
caminho = "C:/Users/gabriel.d/Downloads/apoio_cluster.xlsx"
df = pd.read_excel(caminho)
df["Critério"] = df["Critério"].fillna(method="ffill")

# 2. Transformar em formato longo
df_long = df.melt(id_vars=["Peso", "Critério"], var_name="Seguradora", value_name="Categoria")
df_long.dropna(subset=["Categoria"], inplace=True)
df_long["Critério"] = df_long["Critério"].astype(str)
df_long["Categoria"] = df_long["Categoria"].astype(str)
df_long["Chave"] = df_long["Critério"] + " - " + df_long["Categoria"]
df_long["Peso"] = df_long["Peso"].astype(float)
df_long["Qtd_Respostas_Critério"] = df_long.groupby(["Seguradora", "Critério"])["Categoria"].transform("count")
df_long["Peso_Normalizado"] = df_long["Peso"] / df_long["Qtd_Respostas_Critério"]

# 3. Pivot com pesos normalizados
df_pivot = df_long.pivot_table(
    index="Seguradora",
    columns="Chave",
    values="Peso_Normalizado",
    aggfunc="sum",
    fill_value=0
)
df_pivot.columns = [f"Chave_{col}" for col in df_pivot.columns]
base_modelo = df_pivot.copy()

# 🔢 Cálculo de frequência e CMS
df_numeros = df[df["Critério"].isin(["Itens", "OS", "Custo"])]
df_numeros_long = df_numeros.melt(id_vars=["Critério"], var_name="Seguradora", value_name="Valor")
df_numeros_long = df_numeros_long.dropna(subset=["Valor"])
df_numeros_pivot = df_numeros_long.pivot_table(index="Seguradora", columns="Critério", values="Valor", aggfunc="sum").reset_index()
df_numeros_pivot["Frequência"] = (df_numeros_pivot["OS"] * 12) / df_numeros_pivot["Itens"]
df_numeros_pivot["CMS"] = df_numeros_pivot["Custo"] / df_numeros_pivot["OS"]

# 4. Critérios únicos (sem Itens, OS e Custo)
criterios_opcoes = df_long[~df_long["Critério"].isin(["Custo", "Itens", "OS"])] \
    .groupby("Critério")["Categoria"].unique().to_dict()

# 5. Session state
if "calcular" not in st.session_state:
    st.session_state["calcular"] = False
if "respostas" not in st.session_state:
    st.session_state["respostas"] = {}

# 6. Página 1 - Filtros
if not st.session_state["calcular"]:
    st.title("📋 Preencha o Perfil da Nova Seguradora")

    with st.form("formulario"):
        respostas = {}
        respostas_antigas = st.session_state.get("respostas", {})
        crit_keys = list(criterios_opcoes.keys())

        for i in range(0, len(crit_keys), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(crit_keys):
                    crit = crit_keys[i + j]
                    opcoes = criterios_opcoes[crit]
                    with cols[j]:
                        if crit in ["Produtos Trabalhados", "Canal de venda", "Ramo de mercado"]:
                            default = respostas_antigas.get(crit, [])
                            respostas[crit] = st.multiselect(crit, sorted(opcoes), default=default, key=crit)
                        else:
                            default = respostas_antigas.get(crit, "")
                            opcoes_ordenadas = sorted(opcoes)
                            index = 0
                            if default in opcoes_ordenadas:
                                index = opcoes_ordenadas.index(default) + 1
                            respostas[crit] = st.selectbox(crit, [""] + opcoes_ordenadas, index=index, key=crit)

        submitted = st.form_submit_button("🚀 Calcular Similaridade")

        if submitted:
            st.session_state["respostas"] = respostas
            st.session_state["calcular"] = True
            st.rerun()

# 7. Página 2 - Resultados
else:
    st.title("🔍 Resultados de Similaridade")
    respostas = st.session_state["respostas"]

    st.subheader("🧾 Resumo das Respostas Preenchidas")
    resumo = []
    for crit, val in respostas.items():
        crit_formatado = crit.strip()
        if isinstance(val, list):
            resposta_str = ", ".join(val) if val else "nenhuma"
        else:
            resposta_str = val if val else "não selecionado"
        resumo.append(f"**{crit_formatado}**: {resposta_str}")

    max_por_linha = 4
    for i in range(0, len(resumo), max_por_linha):
        st.markdown(" ; ".join(resumo[i:i + max_por_linha]))

    def construir_vetor(respostas, colunas_modelo):
        total = sum(len(v) if isinstance(v, list) else int(v != "") for v in respostas.values())
        vetor = []
        for col in colunas_modelo:
            peso = 0
            for crit, val in respostas.items():
                if isinstance(val, list):
                    for v in val:
                        if col == f"Chave_{crit} - {v}":
                            peso += 1 / total if total > 0 else 0
                elif val and col == f"Chave_{crit} - {val}":
                    peso = 1 / total if total > 0 else 0
            vetor.append(peso)
        return np.array(vetor).reshape(1, -1)

    vetor_novo = construir_vetor(respostas, base_modelo.columns)
    similaridades = cosine_similarity(vetor_novo, base_modelo.values).flatten()

    resultado = pd.DataFrame({
        "Seguradora": base_modelo.index,
        "Similaridade (%)": (similaridades * 100).round(2)
    }).sort_values(by="Similaridade (%)", ascending=False).reset_index(drop=True)

    top5 = resultado.head(5)

    df_completo = top5.merge(df_numeros_pivot[["Seguradora", "Frequência", "CMS"]], on="Seguradora", how="left")
    df_completo["Frequência"] = pd.to_numeric(df_completo["Frequência"], errors="coerce")
    df_completo["CMS"] = pd.to_numeric(df_completo["CMS"], errors="coerce")
    df_completo["Frequência (%)"] = (df_completo["Frequência"] * 100).round(2).astype(str) + "%"
    df_completo["CMS (R$)"] = df_completo["CMS"].apply(lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    st.subheader("🏆 Top 5 Seguradoras Mais Semelhantes")
    st.dataframe(df_completo[["Seguradora", "Similaridade (%)", "Frequência (%)", "CMS (R$)"]], use_container_width=True)

    st.subheader("📡 Similaridade com Top 5 (Radar)")
    labels = top5["Seguradora"].tolist() + [top5["Seguradora"].iloc[0]]
    valores = top5["Similaridade (%)"].tolist() + [top5["Similaridade (%)"].iloc[0]]

    fig_radar_sim = go.Figure()
    fig_radar_sim.add_trace(go.Scatterpolar(
        r=valores,
        theta=labels,
        fill='toself',
        name='Similaridade (%)',
        line=dict(color='blue')
    ))
    fig_radar_sim.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig_radar_sim, use_container_width=True)

    st.subheader("📌 Estatísticas do Grupo (Médias)")
    freq_media = df_completo["Frequência"].mean() * 100
    cms_media = df_completo["CMS"].mean()
    st.markdown(f"**Frequência média do grupo:** {freq_media:.2f}%")
    st.markdown(f"**Custo médio por OS (CMS):** R$ {cms_media:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    st.subheader("📥 Exportar Ranking Completo e Matriz")

    def gerar_excel(df_resultado, df_matriz):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_resultado.to_excel(writer, index=False, sheet_name="Ranking Similaridade")
            df_matriz.to_excel(writer, index=True, sheet_name="Matriz Similaridade")
        output.seek(0)
        return output

    matriz_similaridade = pd.DataFrame(
        cosine_similarity(base_modelo.values),
        index=base_modelo.index,
        columns=base_modelo.index
    ).round(4)

    excel_data = gerar_excel(resultado, matriz_similaridade)

    st.download_button(
        label="📥 Baixar Excel com Ranking e Matriz",
        data=excel_data,
        file_name="similaridade_completa.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refazer preenchimento", use_container_width=True):
            st.session_state["calcular"] = False
            st.session_state["respostas"] = {}
            st.rerun()

    with col2:
        if st.button("✏️ Corrigir filtros", use_container_width=True):
            st.session_state["calcular"] = False
            st.rerun()
