import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
from collections import Counter
import io
import ast

# Load Model & Vectorizer
model = joblib.load('final_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Sentiment Prediction Function
def predict_sentiment(text):
    text_tfidf = vectorizer.transform([text])
    return model.predict(text_tfidf)[0]

# File Analysis Function
def analyze_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)  # Try reading with headers
    except:
        df = pd.read_csv(uploaded_file, header=None, names=["Review"])

    df = df.dropna(subset=["Review"])  # Remove empty rows
    df["Sentiment"] = df["Review"].apply(predict_sentiment)
    
    sentiment_counts = df["Sentiment"].value_counts()
    sentiment_to_stars = {"Positive": 5, "Neutral": 3, "Negative": 1}
    df["Stars"] = df["Sentiment"].map(sentiment_to_stars)
    avg_stars = round(df["Stars"].mean(), 2)

    return df, sentiment_counts, avg_stars

# Load Base Data for Graphs
preprocessed_data = pd.read_csv("preprocessed_data.csv")
freq_itemsets = pd.read_csv("freq_itemsets_final.csv")
assoc_rules = pd.read_csv("assoc_rules_csv_final.csv")
jaccard_df = pd.read_csv("jaccard_similarity_top.csv")
cooccurrence_edges = pd.read_csv("cooccurrence_graph_edges.csv")

# Streamlit UI
st.title("📊 Palate Patterns: Big Data Insights on Yelp Reviews")
st.write("Upload a **CSV file** of customer reviews or enter a review manually.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df, sentiment_counts, avg_stars = analyze_file(uploaded_file)

    # Sentiment Bar Chart
    st.write("### Sentiment Distribution (%)")
    sentiment_percent = (sentiment_counts / sentiment_counts.sum()) * 100
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_percent.index, y=sentiment_percent.values, palette="coolwarm", ax=ax)
    plt.ylim(0, 100)
    for i, v in enumerate(sentiment_percent.values):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=12)
    st.pyplot(fig)

    # Pie Chart
    st.write("### 📊 Sentiment Pie Chart")
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["green", "yellow", "red"])
    st.pyplot(fig)

    # Rating Summary
    st.write(f"### 🌟 Overall Business Rating: **{avg_stars} Stars**")

    # Word Clouds
    positive_text = ' '.join(df[df["Sentiment"] == "Positive"]["Review"])
    negative_text = ' '.join(df[df["Sentiment"] == "Negative"]["Review"])

    st.write("### 🔥 Word Clouds")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive Reviews")
        st.image(WordCloud(width=400, height=200, background_color="white").generate(positive_text).to_array(), use_container_width=True)
    with col2:
        st.subheader("Negative Reviews")
        st.image(WordCloud(width=400, height=200, background_color="black").generate(negative_text).to_array(), use_container_width=True)

    # Table of Reviews
    st.write("### 🔍 Review Sentiment Breakdown")
    st.dataframe(df.sort_values(by=["Sentiment", "Stars"], ascending=False))

    # Recommendations
    st.write("### 📊 Business Recommendations")
    if avg_stars >= 4.5:
        st.success("🌟 Excellent customer satisfaction! Keep up the great work.")
    elif avg_stars >= 4.0:
        st.success("✅ Strong customer satisfaction, but some areas can be improved.")
    elif avg_stars >= 3.5:
        st.warning("⚠️ Generally positive, but consider addressing common concerns.")
    elif avg_stars >= 3.0:
        st.warning("⚠️ Mixed reviews - Focus on improvements to enhance reputation.")
    else:
        st.error("❌ High number of negative reviews - Immediate action required!")

    # Download CSV
    st.write("### 📥 Download Sentiment Analysis Report")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(label="Download CSV", data=csv_buffer.getvalue(), file_name="sentiment_analysis_results.csv", mime="text/csv")

# Manual Text Input
user_review = st.text_area("Or enter a review manually:")
sentiment_to_stars = {"Positive": 5, "Neutral": 3, "Negative": 1}
if st.button("Analyze Text"):
    sentiment = predict_sentiment(user_review)
    star_rating = sentiment_to_stars[sentiment]
    st.success(f"Predicted Sentiment: **{sentiment}**")
    st.write(f"🌟 Expected Star Rating: **{star_rating} Stars**")

st.divider()

# 🔎 Frequent Itemsets
st.header("🔍 Frequent Pattern Mining")
st.dataframe(freq_itemsets.head(10), use_container_width=True)

# 🔗 Association Rules Network
st.subheader("🔗 Association Rules Network")
def parse_set(s):
    try:
        return ', '.join(list(ast.literal_eval(s)))
    except:
        return str(s)

assoc_rules["antecedent"] = assoc_rules["antecedents"].astype(str).apply(parse_set)
assoc_rules["consequent"] = assoc_rules["consequents"].astype(str).apply(parse_set)
assoc_rules = assoc_rules.sort_values(by="confidence", ascending=False).head(30)

G = nx.DiGraph()
for _, row in assoc_rules.iterrows():
    G.add_edge(row["antecedent"], row["consequent"], weight=row["confidence"])

pos = nx.spring_layout(G, k=0.8)
fig, ax = plt.subplots(figsize=(12, 8))
nx.draw_networkx(G, pos, node_color="skyblue", node_size=600, edge_color="gray", font_size=8)
st.pyplot(fig)

st.divider()

# 🔗 Jaccard Similarity
st.header("🔗 Top Word Pairs by Jaccard Similarity")
st.dataframe(jaccard_df.head(10), use_container_width=True)

st.divider()

# 🌐 Co-Occurrence Graph
st.header("🌐 Co-occurrence Graph of Frequent Words")
G2 = nx.from_pandas_edgelist(cooccurrence_edges, "source", "target", edge_attr="weight")
pos2 = nx.spring_layout(G2, k=0.6)
fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx(G2, pos2, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=500, font_size=8, alpha=0.6)
st.pyplot(fig)

# Footer
st.caption("Project: Palate Patterns | Yelp Big Data Analysis | Developed by Utkarsh Gupta")
