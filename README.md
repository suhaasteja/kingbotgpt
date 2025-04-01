#KingbotGPT
[KingbotGPT](https://libapps.sjsu.edu/kingbot/) is SJSU Library's chatbot. The chatbot is capable of answering questions about the library and its services. It uses a retrieval-augemented generation (RAG) model, and relies on information from the library's website and a local dataset of questions and answers.

##Architecture

In production, KingbotGPT includes the following components:
- User Interface:Streamlit
- Chatbot Control: LlamaIndex
- Vector Database: ChromaDB
- Web Scraper: Scrapy
- Usage Data: MySQL
- LLM: ChatGPT API
![kingbot 2025](https://github.com/user-attachments/assets/de0c0d5c-3eb6-478a-a495-6b79bbf8b6b1)

##Limitations of this distribution

In packaging this shareable code, we are currently choosing to prioritize the ability to run the shared code immediately on [Streamlit Community Cloud](https://streamlit.io/cloud) or any other Streamlit environment. For this reason, the shared code includes only the user interface, chatbot control, and connection to the LLM. Because both the process for creating the local vector database and the usage statistics are hosted outside of the Streamlit environment, we have omitted them from this distribution of the code. 

For an example of a RAG chatbot where the vector database is generated and stored within the Streamlit environment, consider our ["workshop chatbot"](https://github.com/sjsu-library/chatbot-workshop) which we use when teaching RAG chatbot development within the library.

If you are interested in the other components of KingbotGPT, please reach out! We are interested in sharing this code with other libraries in the format that would be most useful to you!

##Quick Start
To make your own copy of KingbotGPT, you will need:
- A GitHub account
- A [Streamlit Community Cloud](https://streamlit.io/cloud) account
- An OpenAI API key
  
###Quick Start Steps
1. Fork this repository on GitHub
3. [Deploy your app on Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
4. Before running your app, add your OpenAI API key to the app's [secrets file](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management). The 
