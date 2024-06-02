import openai
from openai import OpenAI
import streamlit as st
import pinecone
import datetime

openai.api_key = st.secrets["my_secrets"]["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["my_secrets"]["PINECONE_KEY"]
PINECONE_INDEX_NAME = st.secrets["my_secrets"]["INDEX_NAME"]
PINECONE_ENV = st.secrets["my_secrets"]["PINECONE_ENV"]

client = OpenAI(api_key = st.secrets["my_secrets"]["OPENAI_API_KEY"])

class MLChatbot:

    def __init__(self, limit = 3250):
        self.limit = limit

    def display_existing_messages(self):
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def add_user_message_to_session(self, prompt):
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

    @staticmethod
    def get_query_embedding(query):
        # Embed the query using OpenAI's text-embedding-ada-002 engine 
        query_embedding = client.embeddings.create(
            input=[query], engine="text-embedding-ada-002"
        )["data"][0]["embedding"]

        return query_embedding
    
    
    def get_relevant_contexts(self, query_embedding, index): 
        
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        index = pinecone.Index(index_name=index)
        res = index.query(query_embedding, top_k=8, include_metadata=True)
        contexts = []
        for item in res["matches"]:
            metadata = item["metadata"]
            text = metadata.get("text", "")
            url = metadata.get("url", "")
            title = metadata.get("title", "")
            relevance_score = item.get("score", "")
            context = {
                "search_results_text": text, 
                "search_results_url": url, 
                "search_results_title": title, 
                "search_relevance_score": relevance_score
            }
            contexts.append(context)
        
        # Initialize an empty list to hold the final contexts
        final_contexts = []
        
        # Iterate through the contexts and keep track of the total length
        total_length = 0
        for i, context in enumerate(contexts):
            context_str = str(context)
            new_length = total_length + len(context_str)
            if new_length < self.limit:
                final_contexts.append(context)
                total_length = new_length
            else:
                break
        
        return final_contexts
    
    @staticmethod
    def augment_query(contexts, query): 
        augmented_query = (
            f"###Search Results: \n{contexts} #End of Search Results\n\n-----\n\n {query}"
        )
        return augmented_query
    
    def generate_assistant_response(self, augmented_query): 
        primer = """ 
    Your task is to answer user questions based on the information given above each question.It is crucial to cite sources accurately by using the [[number][Title](URL)] notation after the reference. Say "I don't know" if the information is missing and be as detailed as possible. End each sentence with a period. Please begin.
            """ 
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                temperature=0,
                messages=[
                    {"role": "system", "content": primer},
                    {"role": "user", "content": augmented_query},
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            st.session_state["messages"].append(
                {"role": "assistant", "content": full_response}
            )
        return full_response
    
    def print_markdown_from_file(self, file_path):
        with open(file_path, "r") as f:
            markdown_content = f.read()
            st.markdown(markdown_content)

    def main(self):
        st.title("Interactive ML KnowledgeBot: Dive into Machine Learning Queries Instantly!")
        st.write(
        "If the rise of an all-powerful artificial intelligence is inevitable, well it stands to reason that when they take power, our digital overlords will punish those of us who did not help them get there. Ergo, I would like to be a helpful idiot. Like yourself. - Gilfoyle"
        )
        self.display_existing_messages()
        query = st.chat_input("Ask any question related to Machine Learning.")
        if query: 
            self.add_user_message_to_session(query)
            query_embedding = self.get_query_embedding(query)
            contexts = self.get_relevant_contexts(query_embedding, index=PINECONE_INDEX_NAME)
            augmented_query = self.augment_query(contexts, query)
            self.generate_assistant_response(augmented_query)
        with st.sidebar:
            self.print_markdown_from_file("tutorials.md")

if __name__ == "__main__":
    chatbot = MLChatbot()
    chatbot.main()
