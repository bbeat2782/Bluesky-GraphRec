import numpy as np
from scipy.linalg import orthogonal_procrustes
from preprocess_data.factorize import align_embeddings
import pickle
import time

def test_alignment():
    # Test 1: Simple rotation check
    print("=== Test 1: Simple Rotation ===")
    # Original vectors (2D for simplicity)
    source = np.array([[1, 0], [0, 1]], dtype=np.float32)
    target = np.array([[0, 1], [-1, 0]], dtype=np.float32)  # 90-degree rotation
    
    # Dummy producer embeddings (should rotate same way)
    source_prod = np.array([[1, 0], [0, 1]], dtype=np.float32)
    
    aligned_cons, aligned_prod = align_embeddings(source, source_prod, target)
    
    print("Consumer alignment:")
    print("Expected:\n", target)
    print("Actual:\n", aligned_cons)
    
    print("\nProducer alignment:")
    print("Expected:\n", target)
    print("Actual:\n", aligned_prod)
    
    assert np.allclose(aligned_cons, target, atol=1e-6), "Consumer rotation failed"
    assert np.allclose(aligned_prod, target, atol=1e-6), "Producer rotation failed"

def test_real_data():
    print("\n=== Test 2: Real Data Consistency ===")
    with open('./DG_data/bluesky/user_dynamic_features.pkl', 'rb') as f:
        all_embeddings = pickle.load(f)
    
    dates = sorted(all_embeddings.keys())
    num_pairs = 10
    test_pairs = []
    
    # Collect 10 random consecutive pairs
    for _ in range(num_pairs):
        idx = np.random.randint(0, len(dates)-1)
        test_pairs.append((dates[idx], dates[idx+1]))
    
    results = []
    for day1, day2 in test_pairs:
        # Get common users
        common_users = set(all_embeddings[day1].keys()) & set(all_embeddings[day2].keys())
        assert len(common_users) > 0, "No common users between first two days"
        
        # Get embeddings matrices (sorted by user ID)
        users = sorted(common_users)
        day1_embs = np.array([all_embeddings[day1][u] for u in users])
        day2_embs = np.array([all_embeddings[day2][u] for u in users])
        
        # Before alignment stats
        raw_dot = np.diag(day1_embs @ day2_embs.T)  # Raw cosine similarities
        print(f"\nAnalyzing dates: {time.strftime('%Y-%m-%d', time.localtime(day1))} and {time.strftime('%Y-%m-%d', time.localtime(day2))}")
        print(f"Mean similarity before alignment: {np.mean(raw_dot):.4f}")
        
        # Align day2 to day1
        aligned_day2, _ = align_embeddings(day2_embs, day2_embs, day1_embs)
        
        # After alignment stats
        aligned_dot = np.diag(day1_embs @ aligned_day2.T)
        print(f"Mean similarity after alignment: {np.mean(aligned_dot):.4f}")
        
        # Calculate metrics
        R, _ = orthogonal_procrustes(day2_embs, day1_embs)
        ortho_error = np.linalg.norm(R.T @ R - np.eye(R.shape[1]))
        
        results.append({
            'days': (day1, day2),
            'date1': time.strftime('%Y-%m-%d', time.localtime(day1)),
            'date2': time.strftime('%Y-%m-%d', time.localtime(day2)),
            'common_users': len(common_users),
            'pre_alignment_sim': np.mean(raw_dot),
            'post_alignment_sim': np.mean(aligned_dot),
            'ortho_error': ortho_error
        })
    
    # Print summary
    print("\n=== Test Results Summary ===")
    print(f"Tested {num_pairs} consecutive day pairs")
    print(f"Average common users per pair: {np.mean([r['common_users'] for r in results]):.1f}")
    print(f"Average pre-alignment similarity: {np.mean([r['pre_alignment_sim'] for r in results]):.4f}")
    print(f"Average post-alignment similarity: {np.mean([r['post_alignment_sim'] for r in results]):.4f}")
    print(f"Max orthogonality error: {max([r['ortho_error'] for r in results]):.2e}")
    
    # Assertions
    assert all(r['post_alignment_sim'] > 0.8 for r in results), "Some pairs failed alignment"
    assert max([r['ortho_error'] for r in results]) < 1e-5, "Rotation matrices not orthogonal enough"

def analyze_temporal_drift():
    print("\n=== Temporal Drift Analysis ===")
    with open('./DG_data/bluesky/user_dynamic_features.pkl', 'rb') as f:
        all_embeddings = pickle.load(f)
    
    dates = sorted(all_embeddings.keys())
    base_date = dates[0]  # First day in dataset
    
    # Analyze drift over increasing intervals
    intervals = [1, 7, 14, 30, 60]  # Days between comparisons
    results = []
    
    for delta in intervals:
        if len(dates) <= delta:
            continue
            
        target_date = dates[delta]
        common_users = set(all_embeddings[base_date].keys()) & set(all_embeddings[target_date].keys())
        if not common_users:
            continue
            
        # Get embeddings
        users = sorted(common_users)
        base_embs = np.array([all_embeddings[base_date][u] for u in users])
        target_embs = np.array([all_embeddings[target_date][u] for u in users])
        
        # Calculate similarity
        similarities = np.diag(base_embs @ target_embs.T)
        mean_sim = np.mean(similarities)
        
        results.append((delta, mean_sim))
    
    # Print results
    print("\nDays\tAvg Similarity")
    for delta, sim in results:
        print(f"{delta}\t{sim:.4f}")
    
    # Plotting (optional)
    try:
        import matplotlib.pyplot as plt
        plt.plot([d for d, _ in results], [s for _, s in results], 'bo-')
        plt.xlabel('Days Between Embeddings')
        plt.ylabel('Average Cosine Similarity')
        plt.title('Temporal Drift of User Embeddings')
        plt.show()
    except ImportError:
        pass

def check_embedding_norms():
    print("\n=== Embedding Norm Analysis ===")
    with open('./DG_data/bluesky/user_dynamic_features.pkl', 'rb') as f:
        all_embeddings = pickle.load(f)
    
    dates = sorted(all_embeddings.keys())
    
    # Track statistics
    all_norms = []
    issues = []
    
    for date in dates:
        embeddings = np.array(list(all_embeddings[date].values()))
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Collect statistics
        all_norms.extend(norms)
        
        # Check for anomalies
        non_unit = np.abs(norms - 1.0) > 1e-6
        if np.any(non_unit):
            date_str = time.strftime('%Y-%m-%d', time.localtime(date))
            issues.append({
                'date': date_str,
                'min_norm': np.min(norms),
                'max_norm': np.max(norms),
                'non_unit': np.sum(non_unit)
            })
    
    all_norms = np.array(all_norms)
    
    print(f"\nNorm Statistics:")
    print(f"Mean: {np.mean(all_norms):.6f}")
    print(f"Std:  {np.std(all_norms):.6f}")
    print(f"Min:  {np.min(all_norms):.6f}")
    print(f"Max:  {np.max(all_norms):.6f}")
    
    if issues:
        print("\nDates with non-unit norms:")
        for issue in issues:
            print(f"{issue['date']}: {issue['non_unit']} vectors with norms between "
                  f"{issue['min_norm']:.6f} and {issue['max_norm']:.6f}")
    else:
        print("\nAll embeddings are properly normalized (unit vectors)")

if __name__ == "__main__":
    # test_alignment()
    # test_real_data()
    analyze_temporal_drift()
    check_embedding_norms() 