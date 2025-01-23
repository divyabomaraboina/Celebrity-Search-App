#integrate code OpenAI API

import os 
from constants import openai_key
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
from PIL import Image

import streamlit as st
st.set_page_config(page_title="Celebrity Search App", page_icon=":star:")

st.markdown(
        """
        <style>
        .stApp {
            background-color: #D4EEEA;  /* Walmart uses a clean white background */
            color: #0E185F;             /* A deep blue for excellent readability */
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; /* Clean, professional font */
        }
        h1, h2, h3 {
            color: #0071DC;            /* Walmart's signature blue for headings */
        }
        .stTextInput > label, .stButton > button {
            background-color: #0071DC;  /* Signature blue background for buttons and input labels */
            color: white;               /* White text for contrast */
            border: none;
        }
        .stButton > button:hover {
            background-color: #0056b3;  /* Darker blue on hover for buttons */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Streamlit page configuration


    # Header Section
st.write("## Hi!!👋")
st.write("Here is where you can search🔍 about your favourite celebrities.")
logo = Image.open("C:\\Users\\bomar\\Downloads\\logo.jpeg.webp")
st.sidebar.image(logo, use_container_width=True)


os.environ["OPENAI_API_KEY"]= openai_key
st.title("Celebrity Search Application")
input_text= st.text_input("Search the topic u want")

#memory
# person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
# dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
# descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

#For Custom serach you need to create your own prompt
#Prompt Templates:
first_input_prompt = PromptTemplate(
   input_variables= ['name'],
    template =" Tell me about celebrity{name}"
)

#for every template i will have a llm chain
#openAi LLM models
llm = OpenAI(temperature=0.6, max_tokens=500)
chain=LLMChain(llm=llm , 
               prompt= first_input_prompt,
               verbose=True,
               output_key='person',)

second_input_prompt = PromptTemplate(
    input_variables =['person'],
    template=" when was {person} born"
)

chain2=LLMChain(llm=llm , 
               prompt= second_input_prompt,
               verbose=True,
               output_key='dob')

third_input_prompt = PromptTemplate(
    input_variables =['dob'],
    template=" Mention 3 major events happened around {dob} in the world"
)

chain3=LLMChain(llm=llm , 
               prompt= third_input_prompt,
               verbose=True,
               output_key='description')

parent_chain =SequentialChain(
    chains=[chain,chain2,chain3],
    input_variables=['name'],
    output_variables=['person','dob','description'],
    verbose= True)


if input_text :
    st.write(parent_chain({'name':input_text}))


