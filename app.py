# import streamlit as st
# import PyPDF2
# from groq import Groq

# # Set page title
# st.title("📚 Document Q&A Assistant")

# # Sidebar for API key
# with st.sidebar:
#     groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key (starts with 'gsk_')")
#     st.markdown("Using llama-3.3-70b model for document analysis")
    
#     if groq_api_key:
#         client = Groq(api_key=groq_api_key)
    
# # File uploader that accepts PDF and text files
# uploaded_files = st.file_uploader(
#     "Upload documents (PDF, TXT)", 
#     type=["pdf", "txt"], 
#     accept_multiple_files=True
# )

# def read_file(uploaded_file):
#     if uploaded_file.type == "application/pdf":
#         pdf_reader = PyPDF2.PdfReader(uploaded_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         return text
#     else:  # txt file
#         return uploaded_file.getvalue().decode("utf-8")

# # Process uploaded files
# if uploaded_files:
#     combined_text = ""
#     total_size = 0
    
#     # Calculate total size and combine text
#     for file in uploaded_files:
#         total_size += file.size
#         if total_size > 200 * 1024 * 1024:  # 200MB limit
#             st.error("Total file size exceeds 200MB limit!")
#             break
#         combined_text += read_file(file) + "\n\n"

#     # Question input
#     question = st.text_input(
#         "Ask a question about the documents",
#         placeholder="What is the main topic discussed in these documents?",
#         disabled=not combined_text
#     )

#     if question and not groq_api_key:
#         st.info("Please add your Groq API key to continue.")
        
#     elif question and groq_api_key:
#         try:
#             with st.spinner("Analyzing documents using llama-3.3-70b..."):
#                 # Create the prompt
#                 prompt = f"""Context: {combined_text}\n\nQuestion: {question}\n\nPlease answer the question based on the provided context."""
                
#                 # Call Groq API
#                 completion = client.chat.completions.create(
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": prompt
#                         }
#                     ],
#                     model="deepseek-r1-distill-qwen-32b",
#                     temperature=0.1,
#                     max_tokens=1024,
#                 )
                
#                 st.write("### Answer")
#                 st.write(completion.choices[0].message.content)
                
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
# else:
#     st.info("Please upload your documents to begin.")

# # Add requirements
# if st.sidebar.checkbox("Show Requirements"):
#     st.sidebar.code("""
#     streamlit
#     PyPDF2
#     groq
#     """)




















# import streamlit as st
# import PyPDF2
# from groq import Groq
# import tiktoken
# from typing import List

# # Set page title
# st.title("📚 Document Q&A Assistant")

# # Initialize tokenizer
# def num_tokens_from_string(string: str) -> int:
#     encoding = tiktoken.get_encoding("cl100k_base")
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

# def chunk_text(text: str, max_tokens: int = 4000) -> List[str]:
#     """Split text into chunks of approximately max_tokens."""
#     chunks = []
#     current_chunk = []
#     current_size = 0
    
#     # Split by sentences (rough approximation)
#     sentences = text.replace('\n', ' ').split('.')
    
#     for sentence in sentences:
#         sentence = sentence.strip() + '.'
#         sentence_tokens = num_tokens_from_string(sentence)
        
#         if current_size + sentence_tokens > max_tokens:
#             # Join the current chunk and add it to chunks
#             chunks.append(' '.join(current_chunk))
#             current_chunk = [sentence]
#             current_size = sentence_tokens
#         else:
#             current_chunk.append(sentence)
#             current_size += sentence_tokens
    
#     # Add the last chunk if it exists
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
    
#     return chunks

# # Sidebar for API key
# with st.sidebar:
#     groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key (starts with 'gsk_')")
#     model_name = st.selectbox(
#         "Select Model",
#         ["llama-3.3-70b", "mixtral-8x7b", "gemma-7b", "deepseek-r1-distill-qwen-32b"],
#         help="Choose the model to use"
#     )
#     st.markdown(f"Using {model_name} for document analysis")
    
#     if groq_api_key:
#         client = Groq(api_key=groq_api_key)

# def read_file(uploaded_file):
#     if uploaded_file.type == "application/pdf":
#         pdf_reader = PyPDF2.PdfReader(uploaded_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         return text
#     else:  # txt file
#         return uploaded_file.getvalue().decode("utf-8")

# # File uploader
# uploaded_files = st.file_uploader(
#     "Upload documents (PDF, TXT)", 
#     type=["pdf", "txt"], 
#     accept_multiple_files=True
# )

# if uploaded_files:
#     combined_text = ""
#     total_size = 0
    
#     for file in uploaded_files:
#         total_size += file.size
#         if total_size > 200 * 1024 * 1024:  # 200MB limit
#             st.error("Total file size exceeds 200MB limit!")
#             break
#         combined_text += read_file(file) + "\n\n"

#     # Question input
#     question = st.text_input(
#         "Ask a question about the documents",
#         placeholder="What is the main topic discussed in these documents?",
#         disabled=not combined_text
#     )

#     if question and not groq_api_key:
#         st.info("Please add your Groq API key to continue.")
        
#     elif question and groq_api_key:
#         try:
#             with st.spinner(f"Analyzing documents using {model_name}..."):
#                 # Split text into chunks
#                 chunks = chunk_text(combined_text)
                
#                 # Process each chunk and combine results
#                 all_responses = []
                
#                 for i, chunk in enumerate(chunks):
#                     progress_text = f"Processing chunk {i+1} of {len(chunks)}"
#                     st.text(progress_text)
                    
#                     prompt = f"""Context (part {i+1} of {len(chunks)}): {chunk}\n\nQuestion: {question}\n\nPlease answer the question based on this part of the context. If you don't find relevant information in this part, just respond with 'No relevant information found in this part.'"""
                    
#                     completion = client.chat.completions.create(
#                         messages=[{"role": "user", "content": prompt}],
#                         model=model_name,
#                         temperature=0.1,
#                         max_tokens=1000,
#                     )
                    
#                     response = completion.choices[0].message.content
#                     if "No relevant information found in this part" not in response:
#                         all_responses.append(response)
                
#                 st.write("### Answer")
#                 if all_responses:
#                     # Final summary prompt
#                     summary_prompt = f"Question: {question}\n\nHere are the relevant pieces of information found:\n\n" + "\n\n".join(all_responses) + "\n\nPlease provide a coherent summary answering the question based on all these pieces of information."
                    
#                     final_completion = client.chat.completions.create(
#                         messages=[{"role": "user", "content": summary_prompt}],
#                         model=model_name,
#                         temperature=0.1,
#                         max_tokens=1000,
#                     )
                    
#                     st.write(final_completion.choices[0].message.content)
#                 else:
#                     st.write("No relevant information found in the documents.")
                
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
# else:
#     st.info("Please upload your documents to begin.")

# # Add requirements
# if st.sidebar.checkbox("Show Requirements"):
#     st.sidebar.code("""
#     streamlit
#     PyPDF2
#     groq
#     tiktoken
#     """)

















import streamlit as st
import PyPDF2
from groq import Groq
import tiktoken
from typing import List

# Set page title
st.title("📚 Document Q&A Assistant")

# Initialize tokenizer
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text(text: str, max_tokens: int = 4000) -> List[str]:
    """Split text into chunks of approximately max_tokens."""
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Split by sentences (rough approximation)
    sentences = text.replace('\n', ' ').split('.')
    
    for sentence in sentences:
        sentence = sentence.strip() + '.'
        sentence_tokens = num_tokens_from_string(sentence)
        
        if current_size + sentence_tokens > max_tokens:
            # Join the current chunk and add it to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_size += sentence_tokens
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key (starts with 'gsk_')")
    st.markdown("Using DeepSeek R1 Distill Qwen 32B for document analysis")
    
    if groq_api_key:
        client = Groq(api_key=groq_api_key)

def read_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:  # txt file
        return uploaded_file.getvalue().decode("utf-8")

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents (PDF, TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    combined_text = ""
    total_size = 0
    
    for file in uploaded_files:
        total_size += file.size
        if total_size > 200 * 1024 * 1024:  # 200MB limit
            st.error("Total file size exceeds 200MB limit!")
            break
        combined_text += read_file(file) + "\n\n"

    # Question input
    question = st.text_input(
        "Ask a question about the documents",
        placeholder="What is the main topic discussed in these documents?",
        disabled=not combined_text
    )

    if question and not groq_api_key:
        st.info("Please add your Groq API key to continue.")
        
    elif question and groq_api_key:
        try:
            with st.spinner("Analyzing documents using DeepSeek R1 Distill Qwen 32B..."):
                # Split text into chunks
                chunks = chunk_text(combined_text)
                
                # Process each chunk and combine results
                all_responses = []
                
                for i, chunk in enumerate(chunks):
                    progress_text = f"Processing chunk {i+1} of {len(chunks)}"
                    st.text(progress_text)
                    
                    prompt = f"""Context (part {i+1} of {len(chunks)}): {chunk}\n\nQuestion: {question}\n\nPlease answer the question based on this part of the context. If you don't find relevant information in this part, just respond with 'No relevant information found in this part.'"""
                    
                    completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="deepseek-r1-distill-qwen-32b",
                        temperature=0.1,
                        max_tokens=1000,
                    )
                    
                    response = completion.choices[0].message.content
                    if "No relevant information found in this part" not in response:
                        all_responses.append(response)
                
                st.write("### Answer")
                if all_responses:
                    # Final summary prompt
                    summary_prompt = f"Question: {question}\n\nHere are the relevant pieces of information found:\n\n" + "\n\n".join(all_responses) + "\n\nPlease provide a coherent summary answering the question based on all these pieces of information."
                    
                    final_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": summary_prompt}],
                        model="deepseek-r1-distill-qwen-32b",
                        temperature=0.1,
                        max_tokens=1000,
                    )
                    
                    st.write(final_completion.choices[0].message.content)
                else:
                    st.write("No relevant information found in the documents.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload your documents to begin.")

# Add requirements
if st.sidebar.checkbox("Show Requirements"):
    st.sidebar.code("""
    streamlit
    PyPDF2
    groq
    tiktoken
    """)