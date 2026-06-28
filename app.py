
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

st.set_page_config(page_title="Dashboard Pinjol - Kelompok 10", layout="wide")
st.title("🔍 Deteksi Dini & Pola Penyebaran Hoax Pinjol Ilegal")
st.markdown("**Analisis Teks Media Sosial | Universitas Budi Luhur | Kelompok 10**")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv("Pinjol_inset_labeled_final.csv")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df

df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tweet", len(df))
col2.metric("Tweet Negatif", len(df[df["inset_sentiment"]=="Negative"]))
col3.metric("Tweet Positif", len(df[df["inset_sentiment"]=="Positive"]))
col4.metric("Tweet Netral", len(df[df["inset_sentiment"]=="Neutral"]))

def classify_type(text):
    if isinstance(text, str):
        promo_keywords = ["pinjam", "dana cepat", "proses cepat", "hubungi", "wa", "whatsapp", "solusi", "terpercaya", "bunga rendah", "cair"]
        korban_keywords = ["teror", "ancam", "malu", "tagih", "sebar", "data", "galbay", "gagal bayar", "tipu", "modus", "ilegal", "penipuan", "korban"]
        text_lower = text.lower()
        promo = sum(1 for k in promo_keywords if k in text_lower)
        korban = sum(1 for k in korban_keywords if k in text_lower)
        if promo > korban: return "🤖 Promosi/Bot Pinjol"
        elif korban > promo: return "😢 Keluhan Korban"
        else: return "📰 Informasi/Netral"
    return "📰 Informasi/Netral"

df["kategori"] = df["full_text"].apply(classify_type)

st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Data", "🏷️ Klasifikasi", "💬 Sentimen", "🧠 LDA Topik", "🕸️ SNA"])

# ======= TAB 1 =======
with tab1:
    st.header("Dataset Tweet Pinjol")
    st.write(f"Total: **{len(df)} tweet** | Periode: {df['created_at'].min().date()} s/d {df['created_at'].max().date()}")
    search = st.text_input("🔎 Cari kata kunci:")
    if search:
        filtered = df[df["full_text"].str.contains(search, case=False, na=False)]
        st.dataframe(filtered[["created_at","full_text","inset_sentiment","inset_score"]].reset_index(drop=True))
        st.write(f"Ditemukan: {len(filtered)} tweet")
    else:
        st.dataframe(df[["created_at","full_text","inset_sentiment","inset_score","kategori"]].reset_index(drop=True))
    
    st.markdown("---")
    with st.expander("📝 Catatan Presentasi - Tab Data", expanded=True):
        st.info("""
        **[CATATAN PRESENTER]**
        
        Dataset yang kami gunakan berisi **315 tweet** yang dikumpulkan dari Twitter/X 
        menggunakan kata kunci seputar pinjaman online ilegal seperti *pinjol ilegal*, 
        *gagal bayar*, *galbay*, dan *solusi dana cepat*.
        
        Data dikumpulkan pada periode **Juni 2026** dan sudah melalui proses 
        preprocessing seperti cleaning teks, tokenisasi, dan penghapusan stopwords.
        
        Kalian bisa gunakan fitur pencarian di atas untuk melihat tweet berdasarkan 
        kata kunci tertentu secara langsung.
        """)

# ======= TAB 2 =======
with tab2:
    st.header("🏷️ Klasifikasi: Promosi Bot vs Keluhan Korban")
    col1, col2 = st.columns(2)
    with col1:
        kat_count = df["kategori"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(kat_count.values, labels=kat_count.index, autopct="%1.1f%%", colors=["#ff6b6b","#ffd93d","#6bcb77"])
        ax.set_title("Komposisi Kategori Tweet")
        st.pyplot(fig)
    with col2:
        st.bar_chart(kat_count)
    
    st.subheader("Filter berdasarkan Kategori")
    pilihan = st.selectbox("Pilih kategori:", df["kategori"].unique())
    filtered_kat = df[df["kategori"]==pilihan]
    st.write(f"Jumlah: **{len(filtered_kat)} tweet**")
    st.dataframe(filtered_kat[["created_at","full_text","inset_sentiment"]].reset_index(drop=True))
    
    st.markdown("---")
    with st.expander("📝 Catatan Presentasi - Tab Klasifikasi", expanded=True):
        st.info("""
        **[CATATAN PRESENTER]**
        
        Pada tab ini kami melakukan **klasifikasi otomatis** untuk memisahkan dua jenis tweet:
        
        - 🤖 **Promosi/Bot Pinjol** → Tweet yang berisi tawaran pinjaman, kata-kata seperti 
          "dana cepat", "proses mudah", "bunga rendah" — ini diduga berasal dari akun bot 
          atau penyebar iklan pinjol ilegal.
          
        - 😢 **Keluhan Korban** → Tweet yang mengandung kata seperti "teror", "ancam", 
          "sebar data", "gagal bayar" — ini merupakan curahan hati korban pinjol ilegal.
          
        - 📰 **Informasi/Netral** → Tweet berisi berita atau informasi umum tentang pinjol.
        
        Dengan klasifikasi ini, kita bisa membedakan mana konten berbahaya dan mana 
        suara korban yang butuh perhatian.
        """)

# ======= TAB 3 =======
with tab3:
    st.header("💬 Analisis Sentimen (INSET Lexicon)")
    col1, col2 = st.columns(2)
    with col1:
        sent_count = df["inset_sentiment"].value_counts()
        fig, ax = plt.subplots()
        colors = {"Negative":"#ff6b6b","Positive":"#6bcb77","Neutral":"#ffd93d"}
        ax.pie(sent_count.values, labels=sent_count.index, autopct="%1.1f%%",
               colors=[colors.get(s,"#aaa") for s in sent_count.index])
        ax.set_title("Distribusi Sentimen Tweet Pinjol")
        st.pyplot(fig)
    with col2:
        st.subheader("Jumlah per Sentimen")
        st.bar_chart(sent_count)
    
    st.subheader("📈 Tren Sentimen per Hari")
    df["tanggal"] = df["created_at"].dt.date
    tren = df.groupby(["tanggal","inset_sentiment"]).size().unstack(fill_value=0)
    st.line_chart(tren)
    
    st.subheader("📊 Distribusi Skor INSET")
    fig2, ax2 = plt.subplots()
    ax2.hist(df["inset_score"].dropna(), bins=30, color="#4ecdc4", edgecolor="black")
    ax2.set_xlabel("Skor INSET")
    ax2.set_ylabel("Jumlah Tweet")
    ax2.set_title("Histogram Skor Sentimen")
    st.pyplot(fig2)
    
    st.markdown("---")
    with st.expander("📝 Catatan Presentasi - Tab Sentimen", expanded=True):
        st.info("""
        **[CATATAN PRESENTER]**
        
        Analisis sentimen menggunakan metode **INSET Lexicon** — yaitu kamus kata 
        berbahasa Indonesia yang sudah diberi skor positif dan negatif.
        
        Dari **315 tweet** yang dianalisis:
        - Tweet **Negatif** mendominasi → menunjukkan bahwa mayoritas percakapan 
          tentang pinjol di Twitter bernada negatif, baik dari korban maupun kritik publik.
        - Tweet **Positif** kemungkinan berasal dari akun bot yang mempromosikan pinjol.
        - Tweet **Netral** berisi berita atau informasi umum.
        
        Tren sentimen per hari menunjukkan kapan isu pinjol paling ramai dibicarakan.
        Semakin negatif skor INSET, semakin kuat ekspresi negatif dalam tweet tersebut.
        """)

# ======= TAB 4 =======
with tab4:
    st.header("🧠 Analisis Topik LDA")
    try:
        df_lda = pd.read_csv("hasil_lda_datacrawlmix.csv")
        st.write(f"Total data LDA: **{len(df_lda)} baris**")
        if "dominant_topic" in df_lda.columns:
            topic_count = df_lda["dominant_topic"].value_counts().sort_index()
            st.subheader("Distribusi Topik")
            st.bar_chart(topic_count)
        st.dataframe(df_lda.head(50))
    except:
        pass
    
    for nama, file in [("Elbow Perplexity","01_lda_elbow.png"),
                        ("Top Words per Topik","02_lda_topwords.png"),
                        ("Distribusi Topik","03_lda_distribution.png"),
                        ("Word Cloud","04_lda_wordcloud.png"),
                        ("Dashboard LDA","05_lda_dashboard.png")]:
        try:
            st.subheader(nama)
            st.image(file, use_column_width=True)
        except: pass
    
    st.markdown("---")
    with st.expander("📝 Catatan Presentasi - Tab LDA", expanded=True):
        st.info("""
        **[CATATAN PRESENTER]**
        
        **LDA (Latent Dirichlet Allocation)** adalah metode *topic modeling* yang 
        mengelompokkan tweet ke dalam beberapa topik berdasarkan pola kata yang muncul bersama.
        
        Dari hasil LDA pada dataset pinjol ini, ditemukan beberapa klaster topik utama seperti:
        - **Topik modus operandi** → kata-kata seperti "cair", "proses", "bunga", "tenor"
        - **Topik keluhan korban** → kata-kata seperti "teror", "ancam", "sebar", "data pribadi"
        - **Topik regulasi** → kata-kata seperti "OJK", "ilegal", "lapor", "blokir"
        
        Grafik **Elbow** digunakan untuk menentukan jumlah topik optimal.
        **WordCloud** menampilkan kata yang paling sering muncul di setiap topik.
        """)

# ======= TAB 5 =======
with tab5:
    st.header("🕸️ Social Network Analysis")
    st.subheader("Jaringan Mention & Deteksi Bot Sindikat")
    
    try:
        st.image("sna_komunitas_louvain.png", caption="Komunitas Louvain - Deteksi Sindikat Bot", use_column_width=True)
    except: pass
    try:
        st.image("wordcloud_promosi_vs_korban.png", caption="WordCloud: Promosi vs Korban", use_column_width=True)
    except: pass
    
    try:
        df_sna = pd.read_csv("sna_akun_mencurigakan.csv")
        st.subheader("🚨 Akun Mencurigakan (Bot Kandidat)")
        st.write(f"Total akun mencurigakan: **{len(df_sna)}**")
        st.dataframe(df_sna)
    except: pass
    
    st.subheader("📊 Top 10 Akun Paling Aktif (Degree Centrality)")
    mention_list = []
    for text in df["full_text"].dropna():
        mentions = re.findall(r"@(\S+)", text)
        mention_list.extend([m.rstrip(".,;:!?").lower() for m in mentions])
    if mention_list:
        top_mentions = Counter(mention_list).most_common(10)
        df_mentions = pd.DataFrame(top_mentions, columns=["Akun","Jumlah Mention"])
        st.bar_chart(df_mentions.set_index("Akun"))
        st.dataframe(df_mentions)
    
    st.markdown("---")
    with st.expander("📝 Catatan Presentasi - Tab SNA", expanded=True):
        st.info("""
        **[CATATAN PRESENTER]**
        
        **Social Network Analysis (SNA)** digunakan untuk membongkar jaringan akun 
        yang terlibat dalam penyebaran konten pinjol ilegal di Twitter.
        
        Metode yang digunakan:
        - **Degree Centrality** → mengukur seberapa banyak akun di-mention, 
          semakin tinggi = semakin berpengaruh dalam jaringan.
        - **Betweenness Centrality** → menemukan akun yang menjadi "jembatan" 
          penyebaran informasi antar komunitas.
        - **Algoritma Louvain** → mendeteksi komunitas/klaster akun yang saling 
          berinteraksi, berguna untuk membongkar sindikat bot pinjol.
        
        Dari visualisasi graf, terlihat adanya **klaster akun yang saling me-mention** 
        secara masif — ini merupakan indikasi kuat adanya jaringan bot terkoordinasi 
        yang menyebarkan promosi pinjol ilegal.
        """)
