# Pytorch LightGCN Working

https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.LightGCN.html

To get the embeddings, use model.get_embedding(train_data.edge_index.to(device))

# Overview

Takes the likes interactions and builds a bipartite interaction graph. Then trains a LightGCN model on the graph. This gets us our user and post embeddings. Also can perform recommendations for a user. Data is split 80/20 train/test temporally (you can change this).

# pytorch_lightgcn.ipynb

Main file for running everything.

# load_interactions.py

This is the file for loading the interactions. Right now only doing likes. Change the dates in the DuckDB query to get more or less data.

Example likes dataframe:

interaction_uri	                                        post_uri	                                        user_uri	                        timestamp
0	at://did:plc:42kmtf65uqs765coei7bimwx/app.bsky...	at://did:plc:5tgxxpsiv36w3e37im6kd2se/app.bsky...	did:plc:42kmtf65uqs765coei7bimwx	2023-04-25 20:53:03.947
1	at://did:plc:42kmtf65uqs765coei7bimwx/app.bsky...	at://did:plc:p2cp5gopk7mgjegy6wadk3ep/app.bsky...	did:plc:42kmtf65uqs765coei7bimwx	2023-04-30 23:34:44.283
2	at://did:plc:7bo3bipb4qeg43bm5v5oawlu/app.bsky...	at://did:plc:kfmu6v6wzkvr7ez3w2glx4oy/app.bsky...	did:plc:7bo3bipb4qeg43bm5v5oawlu	2023-04-08 05:30:05.770

# build_user_post_graph.py

user2id:

{'did:plc:itkqyh43eswavdpwyxke6yox': 0,
 'did:plc:7kwhdqk7sjv6sjk3af7m3z3c': 1,
 'did:plc:fz6djvsf6xpwrswz6knwkxa2': 2,
 'did:plc:hv4tewpmpx273or7z6nzrpaj': 3,
 ...
}

post2id:

{'at://did:plc:7kwhdqk7sjv6sjk3af7m3z3c/app.bsky.feed.post/3lbni253z322v': 30470,
 'at://did:plc:7kwhdqk7sjv6sjk3af7m3z3c/app.bsky.feed.post/3lbpk4pom6c2o': 30471,
 'at://did:plc:ro23zl3eumgaad34ohfi3t7y/app.bsky.feed.post/3knjljt2ue723': 30472,
 'at://did:plc:ij2egb4apneazwgg6eqenfiz/app.bsky.feed.post/3l6y2tculrm2t': 30473,
 ...
}

The result is a PyG Data object with the following attributes:

- edge_index: Tensor of shape [2, num_edges] containing all edges, where the first row is the user_ids and the second row is the post_ids
- num_nodes: Total number of nodes (users + posts)
- num_users: Number of user nodes
- num_items: Number of post nodes
- user2id: Dictionary mapping user URIs to integer IDs
- post2id: Dictionary mapping post URIs to integer IDs

# temporal_split.py

This is the file for creating the train/test temporal split. It uses build_user_post_graph.py to build the graph. 

It returns the train_data, test_data, and test_interactions.

test_interactions:

{user_id: [post_id, post_id, post_id, ...],
 user_id: [post_id, post_id, post_id, ...],
 ...
}

{1213: [65333, 65337, 44105, 62935, 62923, 64457, 18021],
 823: [65192,
  65187,
  65173,
  65061,
  65036,
...
 2080: [61155],
 774: [33738],
 1374: [23288],
 1767: [35649],
 1652: [17970, 14148]}

# evaluate.py

This is the file for evaluating the LightGCN model. Calculates the average recall@k for a given set of k values.

# recommend_for_user.py

This file contains the logic for recommending posts for a user.

# train_lightgcn.py

This is the actual training file. It runs through all the files above, training the LightGCN model, then evaluates the recommendations, then also outputs recommendations for a selected user. 
