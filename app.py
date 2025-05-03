import streamlit as st
import wikipedia
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_pipeline = load_model()

st.title("🧠 Wikipedia QA Chatbot")
st.write("Ask me anything, and I'll try to answer using Wikipedia!")

question = st.text_input("Enter your question:")
if question:
    try:
    
        search_results = wikipedia.search(question)
        if not search_results:
            st.error("I couldn't find anything on that.")
        else:
            try:
                page = wikipedia.page(search_results[0])
                context = page.content[:1000]  
            
                result = qa_pipeline(question=question, context=context)
                st.success(f"**Answer:** {result['answer']}")
                with st.expander("Show Wikipedia Context"):
                    st.write(context)
            except wikipedia.DisambiguationError as e:
                st.warning(f"Too vague. Try being more specific. Did you mean: {e.options[:5]}?")
            except Exception as e:
                st.error(f"Error retrieving page: {e}")
    except Exception as e:
        st.error(f"Sorry, something went wrong: {e}")
