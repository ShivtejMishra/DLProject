import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# App title with emoji
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Movie Review Sentiment Analyzer 🎯</h1>", unsafe_allow_html=True)
st.write("---")

# Input field
review = st.text_input('📝 **Write Your Movie Review Below:**', '')

# Predict button
if st.button('💡 Predict Sentiment'):
    if review.strip() != '':
        with st.spinner('Analyzing your review... 🔍'):
            # Transformation and prediction
            review_scale = scaler.transform([review]).toarray()
            result = model.predict(review_scale)

        st.write("---")
        if result[0] == 0:
            st.error('😞 **Negative Review Detected!**')
            st.snow()  # Snow animation for sad mood
        else:
            st.success('🎉 **Positive Review Detected!**')
            st.balloons()  # Celebration animation for positive feedback
    else:
        st.warning('⚠️ Please enter a valid review text!')

# Footer
st.markdown("<hr><center>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)
