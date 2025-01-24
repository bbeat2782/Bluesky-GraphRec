import torch
import duckdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def recommend_for_user(model, user_idx, data, top_k=5):
    """
    Given a trained LightGCN model and a user index, return the top_k recommended posts
    in both AT URI and web URL formats, along with the user's profile information and recent likes.
    """
    # Get the user's AT URI and convert to DID and profile URL
    inv_user2id = {v: k for k, v in data.user2id.items()}
    user_uri = inv_user2id[user_idx]
    user_did = user_uri  # user_uri is already in DID format for users
    user_profile_url = f"https://bsky.app/profile/{user_did}"
    
    # Get recommendations using built-in method
    with torch.no_grad():
        src_index = torch.tensor([user_idx]).to(device)         
        start_idx = len(data.user2id)
        dst_index = torch.arange(start_idx, data.num_nodes).to(device)
        
        # Gets the top_k recommendations for the user from 
        recommendations = model.recommend(
            edge_index=data.edge_index.to(device),
            src_index=src_index,
            dst_index=dst_index,
            k=top_k
        )
        
        # Connect to DuckDB to get post content
        con = duckdb.connect('../random_tests/scan_results.duckdb')
        
        # Map recommendations back to post URIs and get content
        inv_post2id = {v: k for k, v in data.post2id.items()}
        formatted_recs = []
        rec_uris = []
        for idx in recommendations[0]:
            at_uri = inv_post2id[idx.item()]
            rec_uris.append(at_uri)
            # Convert AT URI to web URL
            parts = at_uri.split('/')
            did = parts[2]
            post_id = parts[-1]
            web_url = f"https://bsky.app/profile/{did}/post/{post_id}"
            formatted_recs.append((at_uri, web_url))
            
        # Fetch post content for recommendations
        rec_content = con.execute("""
            SELECT 
                'at://' || repo || '/app.bsky.feed.post/' || rkey as post_uri,
                json_extract_string(record, '$.text') as text,
                repo as author,
                createdAt
            FROM records 
            WHERE collection = 'app.bsky.feed.post'
                AND 'at://' || repo || '/app.bsky.feed.post/' || rkey IN (SELECT UNNEST(?))
        """, [rec_uris]).fetchdf()
        
        # Create recommendation lookup
        rec_lookup = {
            row['post_uri']: {
                'text': row['text'],
                'author': row['author'],
                'created_at': row['createdAt']
            }
            for _, row in rec_content.iterrows()
        }
        
        # Get user's recent likes
        recent_likes = []
        # TODO
        
        con.close()
        
        return user_did, user_profile_url, formatted_recs, rec_lookup