#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 23:26:27 2025

@author: B-Bortolato




This script analyzes customer service messages by embedding text, reducing dimensionality, clustering, and categorizing clusters.

1. **Text Embedding**: Converts messages into numerical embeddings using OpenAI's embedding model (`text-embedding-3-small`).
2. **Data Handling**: Reads customer message data (CSV/Parquet) and adds embeddings for messages if not already present.
3. **Dimensionality Reduction**: Reduces embeddings to 2D using UMAP, with a pre-trained reducer saved for future use.
4. **Clustering**: Groups messages into clusters using HDBSCAN. Labels are assigned to clusters and saved.
5. **Visualization**: Plots 2D embeddings before and after clustering and visualizes message distribution over time.
6. **Adding New Data**: Allows the addition of new messages by embedding them, reducing dimensions, and predicting their cluster.
7. **Cluster Prompts**: Generates prompts for OpenAI's GPT to summarize and categorize clusters, saving responses in a dictionary.
8. **Categorization**: Assigns a two-word description to each cluster using GPT, storing results in a JSON file.
9. **UMAP Train/Test Split**: Splits data into train/test sets, applies UMAP, and visualizes the results.

This script can be used for clustering and analyzing customer support messages to improve response strategies.


Input:
    file = 'LLM-DataScientist-Task_Data.csv'
With Columns: 'id_user', 'timestamp', 'source', 'message'

"""






import umap
import json
import hdbscan
import joblib
import numpy as np
import pandas as pd
from dateutil import parser

import matplotlib as mpl
import matplotlib.pyplot as plt


import openai
from api_key import api_key
openai.api_key = api_key







#____________________embedding
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model, dimensions=256)
    return response.data[0].embedding




#____________________read data
try:
    df = pd.read_csv("data.csv", index=False)
except:
    try:
        df = pd.read_parquet("data.parquet")
    except:
        file = 'LLM-DataScientist-Task_Data.csv'
        df = pd.read_csv(file)
        df['message'] = df['message'].str.strip('"')
        df["embedding"] = df["message"].apply(lambda msg: get_embedding(msg))
    






#_________________Dimensional reduction
embeddings = np.vstack(df['embedding'])

try:
    reducer = joblib.load('reducer.joblib')
    emb = np.vstack(df['emb2d'])

except:
    reducer = umap.UMAP(n_neighbors = 30, n_components=2, random_state=50)
    emb = reducer.fit_transform(embeddings)
    joblib.dump(reducer, filename = 'reducer.joblib')
    df["emb2d"] = list(emb)
    #
    df.to_csv("data.csv", index=False)
    df.to_parquet("data.parquet")
    
    

#______________plot points 2D
plt.figure()
plt.scatter(emb[:,0], emb[:,1], c = 'black')
plt.title('Dimensional reduction')
plt.tight_layout()
plt.show()



# #__________________Clustering
try:
    clusterer = joblib.load('clusterer.joblib')
    labels = np.array(df["category"])
except:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5, 
                                  cluster_selection_method='eom', 
                                  prediction_data=True)
    labels = clusterer.fit_predict(emb)
    joblib.dump(clusterer, filename = 'clusterer.joblib')
    df["category"] = labels
    #
    df.to_csv("data.csv", index=False)
    df.to_parquet("data.parquet")




#______________plot clusters
''' Compare this plot with the one without categorized clusters. Are the points well clustered?'''
cmap = mpl.colormaps['gist_rainbow']
colors = cmap(np.linspace(0, 1, labels.max()+1))

plt.figure()
plt.scatter(emb[:,0], emb[:,1], s = 5, c = 'black')
for j in range(labels.max()):
    c = labels == j
    plt.scatter(emb[c,0], emb[c,1], s = 5, c = colors[j])
plt.title('Clusters in 2D')
plt.legend()
plt.tight_layout()
plt.show()


#_____________________How add new points
''' 
Steps to add new messages:
1) Get the embedding for the new message.
   [1]: embeddings_new = get_embedding(text='New text message regarding game issues...').reshape(1, -1)
   
2) Transform the new embeddings using the reducer.
   [2]: emb_new = reducer.transform(embeddings_new)
   
3) Predict the cluster label for the new message using the clusterer.
   [3]: abels_new, _ = hdbscan.approximate_predict(clusterer, emb_new)
   
4) Add the new message and its label to the dataframe 'df'.
'''




df.to_csv("data.csv", index=False)
df.to_parquet("data.parquet")



#_______________________plot time dependence for each cluster
'''Plot distirbution of points along time-axis '''
for jj in range(labels.max()+1):
    tt__ = np.array(df['timestamp'])[labels == jj]
    tt = np.array([parser.parse(tt__[j]) for j in range(len(tt__))])
    plt.figure()
    plt.hist(tt, bins = 100)
    plt.title('cluster index: ' + str(jj))
    plt.ylabel('count')
    plt.tight_layout()
    plt.show()





def get_prompt(sampled_messages):
    prompt = f"""
            I have a group of customer service messages (from a live chat system). Here are a few examples:
            
            {sampled_messages}
            
            Can you:
            1. Summarize what this cluster of messages is about.
            2. Describe the customer's intent.
            3. Suggest a general response or strategy an agent could use when replying.
            """
    return prompt





#________________________define dictionary of prompts
'''
To categorize identified clusters, ask the LLM what the messages belonging to the same cluster have in common.
For each identified cluster, generate a prompt and save it in a dictionary.
'''

try:
    prompts_dict = json.load(open("prompts_dict.json", "r", encoding="utf-8"))
    cluster_info_dict = json.load(open("cluster_info_dict.json", "r", encoding="utf-8"))
    
except Exception as e:
    print(e)
    

    prompts_dict = {}
    num_clusters = labels.max()
    for cluster_id in range(num_clusters):
        msgs = df['message'][labels == cluster_id]
        sampled_messages = "\n- " + "\n- ".join(msgs[:20])
        prompt = get_prompt(sampled_messages)
        prompts_dict[cluster_id] = prompt
        
        
    

    client = openai.OpenAI(api_key=api_key)
    cluster_info_dict = {}
    for cluster_id in range(num_clusters):
        prompt = prompts_dict[cluster_id]
        response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert customer service assistant analyzer."},
                        {"role": "user", "content": prompt}
                        ])
        cluster_info_dict[cluster_id] = response.choices[0].message.content
        
    
    #______________________save cluster describtion dictionary
    json.dump(cluster_info_dict, 
              open("cluster_info_dict.json", "w", encoding="utf-8"), 
              ensure_ascii=False, indent=4)
    
    
    #______________________save prompt dictionary
    json.dump(prompts_dict, 
              open("prompts_dict.json", "w", encoding="utf-8"), 
              ensure_ascii=False, indent=4)
    
    
    







#__________________________UMAP train/test
reducerx = umap.UMAP(n_neighbors = 30, n_components=2, random_state=50)
emb_train = reducerx.fit_transform(embeddings[:3000])
emb_test  = reducerx.transform(embeddings[3000:])


plt.figure()
plt.scatter(emb_train[:,0], emb_train[:,1], s = 5, c = 'black')
plt.title('Dimensional reduction: Test')
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(emb_test[:,0], emb_test[:,1], s = 5, c = 'red')
plt.title('Dimensional reduction: Train')
plt.tight_layout()
plt.show()




# ________________________________________Categorization
'''
Ask the LLM to assign two words that best describe each cluster.
'''

try:
    cluster_name_dict = json.load(open("cluster_name_dict.json", "r", encoding="utf-8"))

except Exception as e:
    print(e)
    num_clusters = labels.max() + 1
    cluster_name_dict = {}
    for cluster_id in range(num_clusters):
        msgs = df['message'][labels == cluster_id]
        sampled_messages = "\n- " + "\n- ".join(msgs[:20])
        
        prompt = f'Return only 2 words that best describe messages: {sampled_messages}'
        response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert at categorizing messages with two words only. You only return 2 word."},
                        {"role": "user", "content": prompt}
                        ])
        cluster_name_dict[cluster_id] = response.choices[0].message.content
    


    json.dump(cluster_name_dict, 
              open("cluster_name_dict.json", "w", encoding="utf-8"), 
              ensure_ascii=False, indent=4)



#___________________________________________Manual tweeking
# cluster_name_dict[0] = 'Game Issues'
# cluster_name_dict[1] = 'Agent Transfer'
# cluster_name_dict[2] = 'Expressing emotions'
# cluster_name_dict[3] = 'Verifications'
# cluster_name_dict[4] = 'Account Issues'
# cluster_name_dict[5] = 'Deposit Issues'
# cluster_name_dict[6] = 'Deposit Bonus Issues'
# cluster_name_dict[7] = 'Withdrawal Issues'
# cluster_name_dict[8] = 'Bonus Inquiry'
# cluster_name_dict[9] = 'Gaming Frustrations'
# cluster_name_dict[10] = 'Freespins Issues'
# cluster_name_dict[11] = 'Cashout Issues'







#__________________FAISS: Not used; not needed for this example
# import faiss
# embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")
# index = faiss.IndexFlatL2(embeddings.shape[1])  
# index.add(embeddings)




############################################################
############################################################
############################################################
# Combine identified clusters, simplify names

category_map = {
                0: 0,
                1: 1,
                2:-1,
                3: 2,
                4: 2,
                5: 3,
                6: 4,
                7: 5,
                8: 4,
                9: 0,
                10: 6,
                11: 7,
                12: -1,
                }


cluster_name_dict_final = {'0': 'game',
                            '1': 'agent',
                            '2': 'account',
                            '3': 'deposit',
                            '4': 'bonus',
                            '5': 'withdrawal',
                            '6': 'freespins',
                            '7': 'cashout',
                            '-1': 'uncategorized'}


df["category"] = df["category"].replace(category_map)


#__________________________save it to another file
df.to_parquet("data_final.parquet")
json.dump(cluster_name_dict_final, 
          open("cluster_name_dict_final.json", "w", encoding="utf-8"), 
          ensure_ascii=False, indent=4)



