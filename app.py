
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx

st.set_page_config(page_title="Analisis Pinjol Twitter", layout="wide")
st.title("📊 Analisis Tweet Pinjaman Online (Pinjol)")
st.markdown("---")

# ======= TABS =======
tab1, tab2, tab3, tab4 = st.tabs(["📁 Data", "💬 Sentimen", "🧠 LDA Topik", "🕸️ SNA"])

# ======= TAB 1: DATA =======
with tab1:
    st.header("Dataset Tweet Pinjol")
    try:
        df = pd.read_csv("Pinjol_preprocessed.csv")
        st.write(f"Total tweet: {len(df)}")
        st.dataframe(df.head(100))
    except:
        st.warning("File CSV tidak ditemukan.")

# ======= TAB 2: SENTIMEN =======
with tab2:
    st.header("Analisis Sentimen")
    try:
        df = pd.read_csv("Pinjol_inset_labeled_final.csv")
        if "sentiment" in df.columns or "label" in df.columns:
            col = "sentiment" if "sentiment" in df.columns else "label"
            count = df[col].value_counts()
            st.bar_chart(count)
            st.dataframe(df[[col]].value_counts().reset_index())
        else:
            st.dataframe(df.head(50))
    except:
        st.warning("File sentimen tidak ditemukan.")

# ======= TAB 3: LDA =======
with tab3:
    st.header("Analisis Topik LDA")
    try:
        df_lda = pd.read_csv("hasil_lda_datacrawlmix.csv")
        st.dataframe(df_lda.head(50))
    except:
        pass
    for img_file in ["01_lda_elbow.png","02_lda_topwords.png","03_lda_distribution.png","04_lda_wordcloud.png","05_lda_dashboard.png"]:
        try:
            st.image(img_file, use_column_width=True)
        except:
            pass

# ======= TAB 4: SNA =======
with tab4:
    st.header("Social Network Analysis")
    try:
        st.image("sna_komunitas_louvain.png", use_column_width=True)
    except:
        pass
    try:
        st.image("wordcloud_promosi_vs_korban.png", use_column_width=True)
    except:
        pass
    try:
        df_sna = pd.read_csv("sna_akun_mencurigakan.csv")
        st.dataframe(df_sna)
    except:
        pass
