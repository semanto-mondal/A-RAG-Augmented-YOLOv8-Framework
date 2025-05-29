import streamlit as st
from PIL import Image
from config import GROQ_API_KEY, VECTOR_STORE_PATH, YOLO_WEIGHTS_PATH
from yolo_model import load_yolo_model, detect_diseases
from rag_chat import prepare_rag_llm, generate_answer

def main():
    st.title("Coffee Leaf Disease Detector & RAG Assistant ☕🌿")

    # Load YOLO model
    yolo_model = load_yolo_model(YOLO_WEIGHTS_PATH)

    # Initialize RAG system in session state once
    if "conversation" not in st.session_state:
        if GROQ_API_KEY:
            qa_chain, llm, memory = prepare_rag_llm(GROQ_API_KEY, VECTOR_STORE_PATH)
            st.session_state.conversation = qa_chain
            st.session_state.llm = llm
            st.session_state.memory = memory
            st.session_state.chat_history = []
            st.session_state.detection_completed = False
        else:
            st.warning("Please set your GROQ_API_KEY in the environment to use the RAG assistant.")
            return

    # Image Detection Section
    st.markdown("## 📸 Upload Coffee Leaf Image")
    uploaded_file = st.file_uploader("Upload coffee leaf image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Detect diseases
        with st.spinner("Analyzing image for diseases..."):
            detected_classes, annotated_img = detect_diseases(img, yolo_model)

        if detected_classes:
            st.image(annotated_img, caption="Detection Results", use_container_width=True)
            unique_diseases = set(detected_classes)
            st.success(f"🦠 **Detected disease(s):** {', '.join(unique_diseases)}")

            # Generate remedy automatically
            with st.spinner("Generating remedy suggestions..."):
                remedy_question = f"What is the remedy for {', '.join(unique_diseases)} in coffee leaves? Provide detailed treatment and prevention methods."
                remedy, sources = generate_answer(remedy_question, st.session_state.conversation, st.session_state.llm)
            
            st.markdown("### 💊 Suggested Remedy")
            st.markdown(remedy)
            
            # Show sources if available
            if sources and sources[0] != "Fallback to LLM (no retrieved docs)":
                with st.expander("📚 Source Documents"):
                    for i, src in enumerate(sources, 1):
                        st.write(f"**Source {i}:** {src[:300]}...")
            
            st.session_state.detection_completed = True
            
        else:
            st.info("✅ No disease detected on the leaf. The leaf appears healthy!")
            st.session_state.detection_completed = True

    # Follow-up Chat Section (only show after detection is completed)
    if st.session_state.detection_completed:
        st.markdown("---")
        st.markdown("## 💬 Ask Follow-up Questions")
        st.markdown("Ask me anything about coffee leaf diseases, treatments, prevention, or general coffee cultivation!")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (speaker, message) in enumerate(st.session_state.chat_history):
                if speaker == "You":
                    st.markdown(f"**🙋 You:** {message}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message}")
                    if i < len(st.session_state.chat_history) - 1:  # Add separator except for last message
                        st.markdown("---")

        # Chat input section
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Your question:", 
                placeholder="e.g., How can I prevent coffee leaf rust? What are the symptoms of coffee berry disease?",
                key="chat_input"
            )
            submit_button = st.form_submit_button("Send 📤")

        # Process user input
        if submit_button and user_question.strip():
            # Add user question to chat history
            st.session_state.chat_history.append(("You", user_question))
            
            # Generate answer
            with st.spinner("Thinking..."):
                answer, sources = generate_answer(user_question, st.session_state.conversation, st.session_state.llm)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append(("Assistant", answer))
            
            # Store sources for the latest response
            st.session_state.latest_sources = sources
            
            # Rerun to update the display
            st.rerun()

        # Show sources for the latest response (if available)
        if hasattr(st.session_state, 'latest_sources') and st.session_state.latest_sources:
            if st.session_state.latest_sources[0] != "Fallback to LLM (no retrieved docs)":
                with st.expander("📚 Sources for Latest Response"):
                    for i, src in enumerate(st.session_state.latest_sources, 1):
                        st.write(f"**Source {i}:** {src[:400]}...")
            else:
                st.info("💡 Answer generated from general knowledge (no specific documents found in knowledge base)")

        # Clear chat history button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                if hasattr(st.session_state, 'latest_sources'):
                    del st.session_state.latest_sources
                # Clear memory as well
                st.session_state.memory.clear()
                st.rerun()

    # Sidebar with instructions
    with st.sidebar:
        st.markdown("## 📋 How to Use")
        st.markdown("""
        1. **Upload Image**: Upload a coffee leaf image to detect diseases
        2. **View Results**: See detection results and automatic remedy suggestions
        3. **Ask Questions**: Use the follow-up chat to ask additional questions
        
        ### 💡 Example Questions:
        - How do I prevent coffee leaf rust?
        - What are the early symptoms of coffee berry disease?
        - What organic treatments are available?
        - How often should I spray fungicides?
        - What environmental conditions promote diseases?
        """)
        
        if st.session_state.detection_completed:
            st.success("✅ Detection completed!👍 You can now ask follow-up questions.")

if __name__ == "__main__":
    main()