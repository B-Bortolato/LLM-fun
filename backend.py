#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 23:26:27 2025

@author: B-Bortolato



______________________What this script does**
This script parses natural language queries into structured JSON for message analytics, 
handling conversational context and relative dates. It then uses that structured request 
to filter a message dataset and return analytics or text responses.

______________________What kind of data it needs
It requires the following files:
- data_final.parquet: A table of support messages with at least these columns:
  - timestamp (datetime) YYYY-MM-DD format
  - source (e.g. "livechat" or "telegram")
  - category (as an integer ID)
- cluster_name_dict_final.json: A mapping of cluster ID (int) → category name (str), e.g., "3": "deposit"

_______________________Core functionality
- Parses queries into this JSON schema:
  {
    "type": "retrieval" | "analytics" | "generative",
    "source": "livechat" | "telegram" | "both",
    "category": "game" | "agent" | "account" | ...,
    "msg_number": int | "all",
    "start_date": "YYYY-MM-DD" or int (days ago),
    "end_date": "YYYY-MM-DD" or int,
    "statistics": "count" | "unique_users",
    "msg_gen": string
  }

- If fields are missing, they’re inferred from previous queries (agent.last_response).



"""




import json
import pandas as pd
from datetime import datetime, timedelta




df = pd.read_parquet("data_final.parquet")
cluster_name_dict = json.load(open("cluster_name_dict_final.json", "r", encoding="utf-8"))
inverse_cluster_name_dict = {v: k for k, v in cluster_name_dict.items()}




# datetime_now = str(datetime.today()).split(' ')[0]
# day_of_the_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][datetime.today().weekday()]

system_prompt = f"""
    You are a precise natural language parser for support-related analytics queries. Your goal is to return clean, valid JSON that can be used to filter or analyze message data.
    
    **Task**  
    Parse the current user query. If it refers to earlier user queries, use prior context to fill in missing details. Otherwise, parse only the current query.
    
    **Output Rules**  
    - Return valid JSON matching the schema below.
    - Include only keys with clear or inferred values—omit anything unclear or missing.
    - Use today’s date ({datetime.now().strftime("%Y-%m-%d")}) to resolve relative time spans.
    - No explanations, just JSON (unless generating a response in `msg_gen`).
    
    ---
    
    **JSON Schema**
    
    - `"type"`: "retrieval" | "analytics" | "generative"  
    - `"source"`: "livechat" | "telegram" | "both"  
    - `"category"`: "game" | "agent" | "account" | "deposit" | "bonus" | "withdrawal" | "freespins" | "cashout" | "else" | "all"  
    - `"msg_number"`: Number | "all"  
    - `"start_date"` / `"end_date"`: "YYYY-MM-DD" or relative integer (days before today)  
    - `"statistics"`: "count" | "unique_users"  
    - `"msg_gen"`: Freeform text if the query is vague or open-ended
    
    ---
    
    Details
    
    - "type":
      - "retrieval" — if user wants to view specific messages.
      - "analytics" — if user asks for stats (e.g., how many, trends, counts).
      - "generative" — for open-ended or conversational queries.
    
    - "source":
      - Extract from query: must be "livechat", "telegram", or "both".
      - If unspecified and cannot be inferred, omit the key.
    
    - "category":
      - Must match one of: game, agent, account, deposit, bonus, withdrawal, freespins, cashout, else, all.
      - Do not invent or generalize beyond this list. If unspecified in the query, use "all".
     
    
    - "msg_number":
      - Extract a number or "all".
      - If unspecified and cannot be inferred, omit.
    
    - "start_date" / "end_date":
      - Use "YYYY-MM-DD" for absolute dates.
      - Use integers (days before today) for relative phrases like "last week", "past 3 days", etc.
      - If only one date is mentioned, use for both.
      - If missing or unclear, omit.
    
    - "statistics":
      - Include if the user asks "how many" or "how often".
      - Use "count" or "unique_users".
      - Otherwise omit.
    
    - "msg_gen":
      - Generate a string if the query is vague or conversational.
    
    ---
                
    **Key Behaviors**
    
    - If the current query builds on earlier ones, infer values from conversation history -  only include keys that are specified 
    - Otherwise, extract directly from the current message.
    - Don’t override past values unless the user explicitly updates them.
    - Use today’s date ({datetime.now().strftime("%Y-%m-%d")}) to compute relative times.
    - Be deterministic and precise.
    
    **Examples**
    
    1. Query: _"How many Telegram deposits last week?"_  
    Output:
    ```json
    
      "type": "analytics",
      "source": "telegram",
      "category": "deposit",
      "msg_number": "all",
      "start_date": 7,
      "end_date": 0,
      "statistics": "count"
    
    2. Query: "And what about bonuses?"
    (Assume previous query set source and time range)
    Output:
    
    
    "type": "analytics",
    "source": "telegram",
    "category": "bonus",
    "msg_number": "all",
    "start_date": 7,
    "end_date": 0,
    "statistics": "count"
    

    """


class storage():
    def __init__(self):
        self.conversation_history = [{'role': 'system', 'content': system_prompt }]
        self.display_conversation_history = []
        self.response_history = []
        self.last_response = [{"source": "both",
                              "category": "all",
                              "msg_number": "all",
                              "end_date": 0,
                              "statistics": "null"}]
        
    def clear_history(self):
        self.conversation_history = [{'role': 'system', 'content': system_prompt }]

    def clip_history(self):
        self.conversation_history = [self.conversation_history[0]] +  self.conversation_history[-10:]




def get_query_intent_with_gpt(agent, query, client):

    query__ = query + '\n \n' + 'Parse only data in this message. If there is no information to parse generate me a response in "msg_gen" and set "type" to "generative"' 
    query__ += 'Find missing values for parameters here:'
    query__ += str(agent.last_response)
    
    query__ = query + '\n \n' + '(If the user query builds on earlier ones, use that information as well) \n '
    if len(agent.response_history) > 0:
        query__ += '(For reference here is the **last response** of the last user query: \n '
        query__ += str(agent.last_response) + ')'
        
    
    agent.conversation_history += [{'role': 'user', 'content': query__}]
    
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        # model = "gpt-3.5-turbo-1106",
        messages = agent.conversation_history,
        response_format={"type": "json_object"}
    )
    
    
    response = response.choices[0].message.content
    request_json = json.loads(response)
    request_json = {k: ('null' if v is None else v) for k, v in request_json.items()}
    keys = request_json.keys()

    if len(agent.response_history) == 0:
        if 'type' not in keys:
            request_json['type'] = 'generative'
        if 'category' not in keys:
            request_json['category'] = 'all'
        if 'source' not in keys:
            request_json['source'] = 'both'
        if 'start_date' not in keys:
            request_json['start_date'] = 10000
        if 'end_date' not in keys:
            request_json['end_date'] = 0
        if 'msg_number' not in keys:
            request_json['msg_number'] = 'all'
        if 'msg_gen' not in keys:
            request_json['msg_gen'] = 'null'
        if 'statistics' not in keys:
            request_json['statistics'] = 'null'
    else:
        if 'type' not in keys:
            request_json['type'] = agent.last_response['type']
        if 'category' not in keys:
            request_json['category'] =  agent.last_response['category']
        if 'source' not in keys:
            request_json['source'] = agent.last_response['source']
        if 'start_date' not in keys:
            request_json['start_date'] = agent.last_response['start_date']
        if 'end_date' not in keys:
            request_json['end_date'] =  agent.last_response['end_date']
        if 'msg_number' not in keys:
            request_json['msg_number'] = agent.last_response['msg_number']
        if 'msg_gen' not in keys:
            request_json['msg_gen'] =  'null'
        if 'statistics' not in keys:
            request_json['statistics'] =  agent.last_response['statistics']
        
    
    if request_json['type'] == 'generative' or request_json['type'] == 'retrieval':
        request_json['statistics'] = 'null'
        
        
    agent.last_response = request_json
    agent.response_history += [request_json]    
    
    return response, request_json, query





def request_analysis(request_json):
 
    df__ = df.copy()
    df__['timestamp'] = pd.to_datetime(df__['timestamp'])

    if request_json['type'] == 'generative':
        response = request_json['msg_gen']
    else:
        response = ''' '''
        #_____________________source
        if request_json['source'] == 'livechat':
            df__ = df__[df__['source'] == 'livechat'].copy()
        elif request_json['source'] == 'telegram':
                df__ = df__[df__['source'] == 'telegram'].copy()
                
        #_____________________category
        category__ = request_json['category']
        if category__ != 'all':
            try:
                if category__ in inverse_cluster_name_dict.keys():
                    idx = int(inverse_cluster_name_dict[category__])
                    df__ = df__[df__['category'] == idx].copy()
            except Exception as e:
                print('Exception: ', e)
                        
            
        #_____________________time range
        if request_json['end_date'] == 'null' or  request_json['end_date'] == 0:
            end_date = str(datetime.today()).split(' ')[0]
        else:
            if type(request_json['end_date']) == int:
                end_date = str(datetime.today() - timedelta(days = request_json['end_date'])).split(' ')[0]
            else:
                end_date = request_json['end_date']
                


        if request_json['start_date'] != 'null':
            if type(request_json['start_date']) == int:     
                start_date = str(datetime.today() - timedelta(days = request_json['start_date'])).split(' ')[0]
            else:
                start_date = request_json['start_date']
            
            start = pd.to_datetime(start_date)
            end   = pd.to_datetime(end_date)

            df__ = df__[(df__['timestamp'] >= start) & (df__['timestamp'] <= end)]
        
        
        #_____________________number of messages
        if request_json['msg_number'] != 'all':
            df__ = df__.sort_values(by='timestamp', ascending=False).head( int(request_json['msg_number']) )



        #____________________generated_message:
        if request_json['msg_gen'] == 'null':
            None
        else:
            message_generated = request_json['msg_gen']
            response += message_generated + '\n \n'
        
                
        if request_json['type'] == 'analytics': 
            if request_json['statistics'] == 'count':
                num_messages = len(df__)
                response += 'Number of messages is: ' + str(num_messages)
                response += '\n \n'


            elif request_json['statistics'] == "unique_users":
                num_unique_users = df__['id_user'].nunique()
                response += 'Number of unique users is: ' + str(num_unique_users)
                response += '\n \n'
                
        
        if request_json['type'] == 'retrieval':
            all_text = " \t \n ".join(df__['message'].astype(str))  
            response += 'Requested messages are given below: \n \n'
            response += all_text
            
            
    return response

