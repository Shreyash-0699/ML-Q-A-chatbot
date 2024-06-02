import time
import advertools as adv
import pandas as pd
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4 
from tqdm.auto import tqdm
import tiktoken
import openai
from openai import OpenAI
import pinecone
import toml
import cloudscraper
from bs4 import BeautifulSoup, Comment
import streamlit as st

openai_api_key = st.secrets["my_secrets"]["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["my_secrets"]["PINECONE_KEY"]
pinecone_index = st.secrets["my_secrets"]["INDEX_NAME"]
pinecone_env = st.secrets["my_secrets"]["PINECONE_ENV"]

client = OpenAI(api_key=openai_api_key)

class SitemapScraper:
    def scrape_sitemap(self, sitemaps):
        """ Scrape a list of sitemaps and return a dataframe with the row results """
        df = pd.concat([adv.sitemap_to_df(sitemap) for sitemap in sitemaps])
        df.to_csv("sitemap.csv", index = False)
        return df

    def convert_df_to_list(self, df):
        """ Convert a dataframe to a list """
        return df["loc"].tolist()

    @staticmethod
    def merge_url_lists(url_list1, url_list2):
        """ This static method is used when there are both sitemaps and urls given """
        return list(set(url_list1 + url_list2))

    @staticmethod
    def scrape_pages(url_list):
        scraper = cloudscraper.create_scraper()  # Create a cloudscraper instance
        records = []

        for url in tqdm(url_list):
            response = scraper.get(url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extracting title
                title_tag = soup.title
                title = title_tag.string if title_tag else "Unknown"
                
                # Check if body exists
                if not soup.body:
                    print(f"Skipping {url} due to no body content found.")
                    continue  # Skip this URL and move to the next one
                
                # Extracting body text
                for unwanted_tag in soup(["script", "style", "nav", "footer"]):
                    unwanted_tag.extract()
                
                body_text = soup.body.get_text(separator="\n", strip=True)
                
                records.append({"url": url, "title": title, "body_text": body_text})
            else:
                print(f"Failed to retrieve {url}. Status code: {response.status_code}")

        df = pd.DataFrame(records)
        df.to_json("page_data.json", orient="records", lines=True)
        df.to_csv("page_data.csv", index=False)
        return df
    
class DataLoader: 
    def __init__(self): 

        # Retrieve the value
        self.PINECODE_API_KEY = pinecone_api_key
        self.OPEN_AI_API_KEY = openai_api_key
        self.PINECONE_INDEX_NAME = pinecone_index
        self.PINECONE_ENV = pinecone_env

        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 5
    
    @staticmethod
    def convert_df_to_dict(df): 
        df = pd.read_json("page_data.json", lines=True, orient="records", encoding="utf-8", dtype=str)
        return df.to_dict(orient="records")
    
    @staticmethod
    def tiktoken_len(text): 
        tokenizer = tiktoken.get_encoding("p50k_base")
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)
    
    @staticmethod
    def create_chunks(data): 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500, 
            chunk_overlap = 20, 
            length_function = DataLoader.tiktoken_len, 
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = []
        for idx, record in enumerate(tqdm(data)): 
            texts = text_splitter.split_text(record["body_text"])
            chunks.extend(
                [
                    {
                        "id" : str(uuid4()), 
                        "title" : record["title"],
                        "text" : texts[i], 
                        "chunk" : i, 
                        "url" : record["url"]
                    }
                    for i in range(len(texts))
                ]
            )
        return chunks
    
    def init_pinecone(self): 
        pinecone.init(api_key=self.PINECODE_API_KEY , environment= self.PINECONE_ENV)

    @staticmethod
    def create_index_if_not_exists(index_name, dimension, metric): 
        if index_name not in pinecone.list_indexes(): 
            pinecone.create_index(index_name, dimension=dimension, metric=metric)
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")

    def create_embeddings(self, chunks, embed_model, index, batch_size=100):
        """ 
        Create embeddings for text chunks and insert them into Pinecone index.

        Args: 
            chunks (list): List of text chunks to be embedded.
            embed_model (str): Name of the OpenAI GPT model to use for embedding.
            index (pinecone.Index): Pineconce index object to insert embeddings into.
            batch_size (int): Number of embeddings to create and insert at once.

        Returns: 
            None
        """

        # Set up OpenAI API key
        openai.api_key = self.OPEN_AI_API_KEY
        success = False
        attempts = 0

        for i in tqdm(range(0, len(chunks), batch_size)): 
            # find end of batch
            i_end = min(len(chunks), i + batch_size)
            meta_batch = chunks[i:i_end]
            # get ids
            ids_batch = [x["id"] for x in meta_batch]
            # get texts to encode
            texts = [x["text"] for x in meta_batch]
            # create embeddings (try-except added to avoid RateLimitError)
            while not success and attempts < self.MAX_RETRIES: 
                try: 
                    res = client.embeddings.create(input=texts, engine=embed_model)
                    success = True
                except: 
                    attempts += 1
                    if attempts < self.MAX_RETRIES: 
                        time.sleep(self.RETRY_DELAY)
                    else: 
                        print("Max retries reached. Failed to create embeddings.")
            embeds = [record["embedding"] for record in res["data"]]
            # cleanup metadata
            meta_batch = [
                {
                    "title": x["title"], 
                    "text": x["text"], 
                    "chunk": x["chunk"], 
                    "url": x["url"]
                }
                for x in meta_batch
            ]
            to_upsert = list(zip(ids_batch, embeds, meta_batch))
            # upsert to Pinecone
            index.upsert(vectors = to_upsert)

sitemaps = [
    "https://machinelearningmastery.com/post-sitemap.xml", 
    "https://machinelearningmastery.com/post-sitemap2.xml", 
    "https://vinija.ai/sitemap.xml", 
    "https://aman.ai/sitemap.xml"
]

# Step 1
# scraper = SitemapScraper()
# df = scraper.scrape_sitemap(sitemaps)
# urls = scraper.convert_df_to_list(df)
# scraper.scrape_pages(urls)


# Step 2 Embedding and indexing

data = pd.read_json(
    "page_data.json", lines=True, orient="records", encoding="utf-8", dtype=str
)

# convert to a list of dicts
data = data.to_dict(orient="records")

embed_data = DataLoader()

chunks = embed_data.create_chunks(data)

# save chunks to json
if not os.path.exists("chunks.json"):
    pd.DataFrame(chunks).to_json("chunks.json", orient="records", lines=True)

# Step 3 --> Uploading to pinecone

print("Initializing Pinecone...")
print(f"Index Name: {pinecone_index}")

embed_data.init_pinecone()
embed_data.create_index_if_not_exists(
    index_name=pinecone_index, dimension=1536, metric="cosine"
)

index = pinecone.GRPCIndex(pinecone_index)

embed_data.create_embeddings(
    chunks=chunks, 
    embed_model="text-embedding-ada-002", 
    index=index,
    batch_size=100,
)
