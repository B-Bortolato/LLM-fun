#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 23:26:27 2025

@author: B-Bortolato




This Streamlit app creates a support chatbot interface. It initializes the UI, 
manages conversation history using session state, and processes user queries 
by calling backend functions for GPT-based intent detection and data analysis. 
Responses and metadata are displayed in a chat format.



"""

import streamlit as st
import backend as b
import json

from openai import OpenAI
from api_key import api_key
client = OpenAI(api_key=api_key)



#__________________________Streamlit gui initialization
st.set_page_config(page_title="Support Chatbot", layout="centered")
st.title("🤖 Support Analytics Chatbot - Test")

if 'agent' not in st.session_state:
    st.session_state.agent = b.storage()
agent = st.session_state.agent






#__________________________Display conversation history
if len(agent.display_conversation_history) > 0:

        
    for i in range(len(agent.display_conversation_history)):
        msg = agent.display_conversation_history[i]
        st.chat_message(msg["role"]).markdown(msg["content"])

    with st.expander("### 📊 Parsed Metadata"):
        try:
            st.json(agent.last_response)
        except json.JSONDecodeError:
            st.error("❌ Could not parse response as JSON.")
        





#__________________________Logic
user_input = st.chat_input("Ask a support-related analytics question:")
if user_input:

    with st.spinner("Analyzing your query..."):
        try:
            query = user_input
            agent.display_conversation_history += [{'role': 'user', 'content': query}]
            response, request_json, query = b.get_query_intent_with_gpt(agent, query, client)
            final_response  = b.request_analysis(request_json)
            
            agent.conversation_history += [{'role': 'assistant', 'content': final_response}]
            agent.display_conversation_history +=  [{'role': 'assistant', 'content': final_response}]


        except Exception as e:
            st.error(f"❌ API Error: {e}")
        st.rerun()


