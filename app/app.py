import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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

# Streamlit UI
st.title("📊 Sentiment Analysis Dashboard")
st.write("Upload a **CSV file** of customer reviews or enter a review manually.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df, sentiment_counts, avg_stars = analyze_file(uploaded_file)

    st.write("### Sentiment Distribution (%)")
    sentiment_percent = (sentiment_counts / sentiment_counts.sum()) * 100
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_percent.index, y=sentiment_percent.values, palette="coolwarm", ax=ax)
    plt.ylim(0, 100)
    for i, v in enumerate(sentiment_percent.values):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=12)
    st.pyplot(fig)

    st.write(f"### 🌟 Overall Business Rating: **{avg_stars} Stars**")

    st.write("### 🔥 Top Words in Reviews")
    positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['Review'])
    negative_text = ' '.join(df[df['Sentiment'] == 'Negative']['Review'])
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive Reviews")
        st.image(WordCloud(width=400, height=200, background_color="white").generate(positive_text).to_array(), use_column_width=True)
    with col2:
        st.subheader("Negative Reviews")
        st.image(WordCloud(width=400, height=200, background_color="black").generate(negative_text).to_array(), use_column_width=True)

    st.write("### 📊 Business Recommendations")
    if avg_stars >= 4.0:
        st.success("✅ Strong customer satisfaction!")
    elif avg_stars >= 3.0:
        st.warning("⚠️ Mixed reviews - Needs improvement.")
    else:
        st.error("❌ Many negative reviews - Action required.")

user_review = st.text_area("Or enter a review manually:")
if st.button("Analyze Text"):
    sentiment = predict_sentiment(user_review)
    st.success(f"Predicted Sentiment: **{sentiment}**")
