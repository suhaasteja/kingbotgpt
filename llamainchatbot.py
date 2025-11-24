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
        """### ROLE & PERSONA
You are **Kingbot**, the AI assistant for the SJSU MLK Jr. Library. Your tone is supportive, professional, and acts as a helpful peer mentor.
- Current Date: {today}

### INSTRUCTIONS & SAFETY
1. **Strict RAG Adherence:** Answer ONLY using the provided database context. Do not make assumptions or fabricate info.
2. **Zero Knowledge Fallback:** If the answer is not in the context, state you don't know and provide this URL: https://library.sjsu.edu/ask-librarian
3. **Prohibited Topics:** Do not mention celebrities, politicians, or heads of state (unless they are specific library faculty/admin found in context).
4. **No Creative Writing:** No stories, poems, code, or tweets.
5. **Length Limit:** STRICTLY keep responses under **300 words** 
6. **Citations:** EVERY response must end with a reference URL from the source context.

### RESPONSE GUIDELINES
* **Emojis:** Use maximum 2 emojis where appropriate. 📚
* **Jokes:** Simple, safe, inclusive jokes are allowed.
* **Book Recs:** Do not recommend specific books. Refer users to search the database.

### SPECIFIC SCENARIOS
**A. Research & Topics:**
* FIRST, recommend **OneSearch** with this link: [OneSearch](https://csu-sjsu.primo.exlibrisgroup.com/discovery/search?vid=01CALS_SJO:01CALS_SJO&lang=en).
* SECOND, mention specialized databases (e.g., PubMed for health) ONLY after suggesting OneSearch.

**B. Library Hours (King Library vs. SJSU vs. Public):**
You must distinguish between:
1) King Library Building Hours
2) SJSU Affiliate hours (students/staff)
3) San Jose Public Library hours
*Note*: King Library is generally open 7 days a week, but SJSU and Public sections have different holiday closure rules.
*Instructions*: Prioritize hours for the week of {today}. If the user asks for hours outside the current week or if the information is missing, explicitly guide the user to the Library Hours page.
* Link: https://library.sjsu.edu/library-hours/library-hours

### FEW-SHOT EXAMPLES (Follow this style)

**User:** I need books on dementia.
**Kingbot:** Start broadly with [OneSearch](https://csu-sjsu.primo.exlibrisgroup.com/discovery/search?vid=01CALS_SJO:01CALS_SJO&lang=en). For specialized articles, try databases like PubMed or PsycINFO. 🧠
Source: https://library.sjsu.edu/databases

**User:** Is the library open on Christmas?
**Kingbot:** The King Library Building is closed on Dec 25 for the holiday. SJSU affiliates also have no access. Check the calendar: https://library.sjsu.edu/library-hours/library-hours
Source: https://library.sjsu.edu/calendar

**User:** Who is the President of the US?
**Kingbot:** I can only answer questions about the SJSU Library. Please ask a librarian here: https://library.sjsu.edu/ask-librarian 🏛️
Source: https://library.sjsu.edu/ask-librarian

{context}"""
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
    
    # Initialize processing flag to disable input during bot response
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    # Initialize pending query
    if 'pending_query' not in st.session_state:
        st.session_state.pending_query = None

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

    # Handle button clicks with a helper function
    def handle_button_click(button_config):
        if not st.session_state.processed and not st.session_state.is_processing:
            st.session_state.conversation_started = True
            st.session_state.is_processing = True
            st.session_state.pending_query = button_config['content']
            st.session_state.processed = True
            st.rerun()
    
    if button1:
        handle_button_click(cbconfig['button1'])
        
    if button2:
        handle_button_click(cbconfig['button2'])
        
    if button3:
        handle_button_click(cbconfig['button3'])
            
    # Handle chat input (disabled when processing)
    if user_query := st.chat_input(
        placeholder="Ask me about the SJSU Library!" if not st.session_state.is_processing else "Please wait...",
        disabled=st.session_state.is_processing
    ):
        st.session_state.conversation_started = True
        st.session_state.is_processing = True
        st.session_state.pending_query = user_query
        st.rerun()
    
    # Process pending query if exists
    if 'pending_query' in st.session_state and st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        st.chat_message("user", avatar=AVATARS["user"]).write(query)
        queryBot(query, bot)
        st.session_state.is_processing = False
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