import torch
from typing import List, Dict, Tuple

def get_recommendations(model: torch.nn.Module,
                       user_id: str,
                       user_mapping: Dict,
                       post_mapping: Dict,
                       adj_matrix: torch.sparse.FloatTensor,
                       top_k: int = 10,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[Tuple[str, str]]:
    """Get recommendations for a user with both AT protocol URIs and web URLs."""
    model = model.to(device)
    adj_matrix = adj_matrix.to(device)
    
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(adj_matrix)
        
        user_idx = user_mapping[user_id]
        user_emb = user_embeddings[user_idx]
        scores = torch.matmul(user_emb, item_embeddings.t())
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        # Convert indices to recommendations with both URI and web URL formats
        reverse_mapping = {v: k for k, v in post_mapping.items()}
        recommendations = []
        for idx in top_indices.cpu():
            at_uri = reverse_mapping[idx.item()]
            # Convert AT URI to web URL
            # Format: at://did:plc:xyz/app.bsky.feed.post/123 
            # To: https://bsky.app/profile/[handle]/post/123
            parts = at_uri.split('/')
            did = parts[2]
            post_id = parts[-1]
            web_url = f"https://bsky.app/profile/{did}/post/{post_id}"
            recommendations.append((at_uri, web_url))
        
        return recommendations