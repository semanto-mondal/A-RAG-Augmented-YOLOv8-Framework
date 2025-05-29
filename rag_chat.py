from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import streamlit as st

def create_custom_prompt():
    """Create a custom prompt template for better RAG responses"""
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer or if the context doesn't contain relevant information, just say that you don't have specific information about this topic in your knowledge base.
    Don't try to make up an answer or provide information that's not in the context.

    Context: {context}

    Question: {question}

    Answer: """
    
    return PromptTemplate(template=template, input_variables=["context", "question"])

@st.cache_resource(show_spinner=False)
def prepare_rag_llm(api_key, vector_store_path, temperature=0.3, max_length=2048):
    """Initialize RAG system with LLM, vector store, and memory"""
    try:
        llm = ChatGroq(
            api_key=api_key,
            model="llama3-8b-8192",
            temperature=temperature,
            max_tokens=max_length
        )

        memory = ConversationBufferWindowMemory(
            k=5,  # Keep last 5 conversation turns
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Load vector store
        db = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Create conversational retrieval chain with custom retriever
        retriever = db.as_retriever(search_kwargs={"k": 3, "fetch_k": 10})
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            memory=memory,
            verbose=False,
            combine_docs_chain_kwargs={
                "prompt": create_custom_prompt()
            }
        )

        return qa_chain, llm, memory
    
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None, None

def generate_answer(question, conversation, llm=None):
    """Generate answer using RAG or fallback to LLM"""
    if not conversation:
        return "RAG system not initialized.", []

    # First, check if question is related to coffee/agriculture
    coffee_keywords = ['coffee', 'leaf', 'disease', 'plant', 'crop', 'fungus', 'pest', 'treatment', 'remedy', 'cultivation', 'agriculture', 'farming']
    is_coffee_related = any(keyword.lower() in question.lower() for keyword in coffee_keywords)
    
    # For general greetings or non-coffee questions, use LLM directly
    general_greetings = ['hi', 'hello', 'good morning', 'good afternoon', 'good evening', 'how are you', 'thanks', 'thank you']
    is_greeting = any(greeting.lower() in question.lower() for greeting in general_greetings)
    
    if is_greeting or not is_coffee_related:
        if llm:
            try:
                if is_greeting:
                    greeting_prompt = f"""
                    You are a helpful coffee agriculture assistant. Respond to this greeting in a friendly, professional manner and offer to help with coffee-related questions:
                    
                    User: {question}
                    
                    Keep your response brief and friendly.
                    """
                else:
                    greeting_prompt = f"""
                    You are a helpful assistant specializing in coffee agriculture and plant diseases. The user asked: {question}
                    
                    If this question is not related to coffee, plants, or agriculture, politely redirect them to coffee-related topics while still being helpful.
                    """
                
                answer = llm.predict(greeting_prompt)
                return answer.strip(), ["Generated from general knowledge"]
            except Exception as e:
                return "Hello! I'm here to help you with coffee leaf diseases and cultivation questions. How can I assist you today?", []

    try:
        # Get response from conversational retrieval chain for coffee-related questions
        response = conversation({"question": question})
        
        if isinstance(response, dict) and "answer" in response:
            answer = response["answer"]
            source_docs = response.get("source_documents", [])
            sources = [doc.page_content for doc in source_docs]
            
            # Clean up the answer
            if "Helpful Answer:" in answer:
                answer = answer.split("Helpful Answer:")[-1].strip()
            
            # Check if the answer seems to be a generic/irrelevant response
            # Look for signs that the RAG system returned irrelevant content
            irrelevant_indicators = [
                "according to the provided coffee leaf disease guide" in answer.lower(),
                "leaf miner" in answer.lower() and "leaf miner" not in question.lower(),
                len(answer.strip()) < 20,
                answer.strip().startswith("*") and question.lower() not in answer.lower()
            ]
            
            # Check similarity between question and answer
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            common_words = question_words.intersection(answer_words)
            similarity_score = len(common_words) / max(len(question_words), 1)
            
            # If answer seems irrelevant or has low similarity, use LLM fallback
            if any(irrelevant_indicators) or similarity_score < 0.1:
                if llm:
                    try:
                        fallback_prompt = f"""
                        You are an expert in coffee cultivation and plant pathology. Please provide a detailed and helpful answer to the following question:
                        
                        Question: {question}
                        
                        If you don't have specific information about this topic, please say so and provide general guidance where appropriate.
                        Be honest about the limitations of your knowledge while still being helpful.
                        """
                        
                        llm_response = llm.predict(fallback_prompt)
                        answer = llm_response.strip()
                        sources = ["Generated from general agricultural knowledge"]
                    except Exception as e:
                        print(f"Fallback LLM failed: {e}")
                        # Keep original answer if fallback fails
            
            # Clean up sources
            if not sources or all(len(src.strip()) < 20 for src in sources):
                sources = ["Retrieved from knowledge base"]
                
        else:
            answer = str(response)
            sources = ["No sources available"]
            
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(error_msg)
        
        # Try fallback LLM if main chain fails
        if llm:
            try:
                fallback_prompt = f"""
                You are a helpful coffee agriculture expert. Please answer this question:
                {question}
                
                Be friendly and helpful, and if you don't know something specific, admit it while providing what general guidance you can.
                """
                answer = llm.predict(fallback_prompt)
                sources = ["Generated from general knowledge (system temporarily unavailable)"]
            except Exception as fallback_error:
                answer = "I apologize, but I'm currently experiencing technical difficulties. Please try asking your question again."
                sources = []
        else:
            answer = "I'm sorry, I'm unable to process your question right now. Please try again later."
            sources = []

    return answer, sources

def reset_conversation_memory(memory):
    """Reset the conversation memory"""
    if memory:
        memory.clear()
        return True
    return False