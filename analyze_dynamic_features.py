import pickle
import numpy as np
from datetime import datetime

def quick_check():
    with open('./DG_data/bluesky/user_dynamic_features.pkl', "rb") as f:
        data = pickle.load(f)
    
    # Get 2 consecutive days
    sorted_dates = sorted(data.keys())
    mid_idx = len(sorted_dates) // 2
    day1, day2 = sorted_dates[mid_idx:mid_idx+2]  # Two consecutive days from middle of dataset
    
    # Convert to readable dates
    date1_str = datetime.utcfromtimestamp(day1).strftime('%Y-%m-%d')
    date2_str = datetime.utcfromtimestamp(day2).strftime('%Y-%m-%d')
    
    print(f"Checking consecutive days: {date1_str} → {date2_str}")
    
    # Find common users
    common_users = set(data[day1].keys()) & set(data[day2].keys())
    print(f"Found {len(common_users)} common users")
    
    # Test first 5 common users
    for user_id in list(common_users)[:5]:
        emb1 = data[day1][user_id]
        emb2 = data[day2][user_id]
        
        print(f"\nUser {user_id}:")
        print(f"Day 1 norm: {np.linalg.norm(emb1):.4f}")
        print(f"Day 2 norm: {np.linalg.norm(emb2):.4f}")
        
        # Check for zero vectors
        if np.allclose(emb1, 0) or np.allclose(emb2, 0):
            print("⚠️ Zero vector detected!")
            continue
            
        # Raw dot product
        dot = np.dot(emb1, emb2)
        print(f"Raw dot: {dot:.4f}")
        
        # Cosine similarity
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm_product == 0:
            print("💥 Zero norm product!")
        else:
            cos_sim = dot / norm_product
            print(f"Cosine similarity: {cos_sim:.4f}")
        
        # Direct comparison
        print(f"Vectors identical: {np.allclose(emb1, emb2)}")

if __name__ == "__main__":
    quick_check()