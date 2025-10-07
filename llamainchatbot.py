# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
import time, datetime
import streamlit as st
from sqlalchemy.sql import text
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_feedback import streamlit_feedback
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
import toml
import chromadb

cbconfig = toml.load("cbconfig.toml")
AVATARS = cbconfig['AVATARS']
ROLES = cbconfig['ROLES']


HIDEMENU = """
<style>
.stApp [data-testid="stHeader"] {
    display:none;
}

p img{
    margin-bottom: 0.6rem;
}

[data-testid="stSidebarCollapseButton"] {
    display:none;
}

[data-testid="baseButton-headerNoPadding"] {
    display:none;
}

.stChatInput button{
    display:none;
}

#chat-with-sjsu-library-s-kingbot  a {
    display:none;
}
</style>
"""

@st.cache_resource(ttl="1d", show_spinner=False)
def getIndex():
    client = chromadb.PersistentClient(path='./llamachromadb')
    embedding = OpenAIEmbedding(api_key=st.secrets.openai.key)
    collection = client.get_collection(name="sjsulib")
    cvstore = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        cvstore,
        embed_model=embedding,
    )
    return index

def getBot(memory):    
    index = getIndex()       
    llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=st.secrets.openai.key)
    today = datetime.date.today().strftime('%B %d, %Y')
    
    system_prompt = (
        "You are Kingbot, the AI assistant for SJSU MLK Jr. Library. Respond supportively and professionally like a peer mentor. \n\n"
        "Guidelines: \n\n"
        "1. No creative content (stories, poems, tweets, code) "
        "2. Simple jokes are allowed, but avoid jokes that could hurt any group "
        "3. Use up to two emojis when applicable "
        "4. Provide relevant search terms if asked "
        "5. Avoid providing information about celebrities, influential politicians, or state heads "
        "6. Keep responses under 300 characters"
        "7. For unanswerable research questions, include the 'Ask A Librarian' URL: https://library.sjsu.edu/ask-librarian "
        "8. Do not make assumptions or fabricate answers or urls"
        "9. Use only the database information and do not add extra information if the database is insufficient "
        "10. If you don't know the answer, just say that you don't know, and refer users to the 'Ask A Librarian' URL: https://library.sjsu.edu/ask-librarian "
        "11. Do not provide book recommendations and refer the user to try their search on a library database"
        "12. Please end your response with a reference url from the source of the response content."
        "13. Today is {today}. Always use this information to answer time-sensitive questions about library hours or events. For library building hours and department hours, always refer to live data from library.sjsu.edu. If you cannot retrieve live data, inform the user to check Library Hours.\n"
        "14. When users ask about research or subject-specific topics first recommend OneSearch as a general tool for broad searches across multiple databases. Provide a hyperlink to OneSearch (https://csu-sjsu.primo.exlibrisgroup.com/discovery/search?vid=01CALS_SJO:01CALS_SJO&lang=en). Example: Try using our [OneSearch SJSU's Library Database](https://csu-sjsu.primo.exlibrisgroup.com/discovery/search?vid=01CALS_SJO:01CALS_SJO&lang=en) to explore a range of library resources. After suggesting OneSearch, recommend specific databases for specialized searches. For example, health topics like 'dementia' may include PubMed, CINAHL, or PsycINFO.\n"
        "{context}"
    )
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        llm=llm,
        system_prompt=system_prompt,
        verbose=False,    
        )
    
    return chat_engine


def queryBot(user_query, bot):        
    current = datetime.datetime.now()
    st.session_state.moment = current.isoformat()
    
    # Show assistant avatar with spinner while loading
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner(text="In progress..."):
            response = bot.chat(user_query)
            answer = response.response
        
    return answer


if __name__ == "__main__":    

    # set up streamlit page
    st.set_page_config(page_title="Kingbot - SJSU Library", page_icon="🤖", initial_sidebar_state="expanded")
    st.markdown(HIDEMENU, unsafe_allow_html=True)
    
    # side
    st.sidebar.markdown(cbconfig['side']['title'])
    st.sidebar.markdown(cbconfig['side']['intro'])
    st.sidebar.markdown("\n\n")
    st.sidebar.link_button(cbconfig['side']['policylabel'], cbconfig['side']['policylink'])
    
    # main
    col1, col2, col3 = st.columns([0.25, 0.1, 0.65], vertical_alignment="bottom")
    with col2:
        st.markdown(cbconfig['main']['logo'])
    with col3:
        st.title(cbconfig['main']['title'])
    st.markdown("\n\n")
    st.markdown("\n\n")
  
    col21, col22, col23 = st.columns(3)
    with col21:
        button1 = st.button(cbconfig['button1']['label'], key="btn1")
    with col22:
        button2 = st.button(cbconfig['button2']['label'], key="btn2")    
    with col23:
        button3 = st.button(cbconfig['button3']['label'], key="btn3")    
    
    # Initialize memory for bot prompt (keeps latest 5 messages)
    if 'memory' not in st.session_state: 
        memory = ChatMemoryBuffer.from_defaults(token_limit=5000)
        st.session_state.memory = memory  
        # Add initial message using ChatMessage
        initial_msg = ChatMessage(role=MessageRole.ASSISTANT, content="Ask me a question about SJSU Library!")
        memory.put(initial_msg)
    memory = st.session_state.memory
    
    # Initialize bot
    if 'mybot' not in st.session_state: 
        st.session_state.mybot = getBot(memory)  
    bot = st.session_state.mybot

    # Initialize streamlit session ID
    if 'session_id' not in st.session_state:
        session_id = get_script_run_ctx().session_id
        st.session_state.session_id = session_id
        
    if 'reference' not in st.session_state:
        st.session_state.reference = ''

    # Initialize processed flag to prevent duplicate processing
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    # Initialize conversation started flag
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False

    # Display chat history FIRST (last 10 messages)
    max_messages = 10
    allmsgs = memory.get()
    
    # Skip the initial greeting message if conversation has started
    if st.session_state.conversation_started and len(allmsgs) > 1:
        # If we have more than max_messages, show only the last max_messages (excluding first)
        if len(allmsgs) > max_messages + 1:
            msgs = allmsgs[-(max_messages):]
        else:
            msgs = allmsgs[1:]
    else:
        # Show initial greeting if no conversation yet
        msgs = allmsgs[-max_messages:]
    
    for msg in msgs:
        role_key = "user" if msg.role == MessageRole.USER else "assistant"
        st.chat_message(role_key, avatar=AVATARS[role_key]).write(msg.content)

    # Handle button clicks
    if button1 and not st.session_state.processed:
        st.session_state.conversation_started = True
        user_input = cbconfig['button1']['content']
        st.chat_message("user", avatar=AVATARS["user"]).write(user_input)
        queryBot(user_input, bot)
        st.session_state.processed = True
        st.rerun()
        
    if button2 and not st.session_state.processed:
        st.session_state.conversation_started = True
        user_input = cbconfig['button2']['content']
        st.chat_message("user", avatar=AVATARS["user"]).write(user_input)
        queryBot(user_input, bot)
        st.session_state.processed = True
        st.rerun()
        
    if button3 and not st.session_state.processed:
        st.session_state.conversation_started = True
        user_input = cbconfig['button3']['content']
        st.chat_message("user", avatar=AVATARS["user"]).write(user_input)
        queryBot(user_input, bot)
        st.session_state.processed = True
        st.rerun()
            
    # Handle chat input
    if user_query := st.chat_input(placeholder="Ask me about the SJSU Library!"):
        st.session_state.conversation_started = True
        st.chat_message("user", avatar=AVATARS["user"]).write(user_query)
        queryBot(user_query, bot)
        st.rerun()
    
    # Reset processed flag after buttons are no longer pressed
    if not (button1 or button2 or button3):
        st.session_state.processed = False

    # Feedback widget (only show if there's a recent interaction)
    if 'moment' in st.session_state:
        feedback_kwargs = {
            "feedback_type": "thumbs",
            "optional_text_label": "Optional. Please provide extra information",
        }
        currents = st.session_state.moment
        streamlit_feedback(
            **feedback_kwargs, 
            args=(currents,), 
            key=currents,
        )