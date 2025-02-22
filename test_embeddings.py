import numpy as np
import scipy.sparse as sp
from preprocess_data.factorize import factorize, align_embeddings
import pickle
import time
from datetime import datetime
import os

def test_basic_factorization():
    """Test basic matrix factorization without alignment"""
    print("\n=== Test 1: Basic Factorization ===")
    
    # Create a simple test matrix
    n_consumers, n_producers = 100, 150
    density = 0.1
    rand_matrix = sp.random(n_consumers, n_producers, density=density, format='coo')
    
    # Run factorization with fewer clusters for testing
    _, _, consumer_emb, producer_emb, _ = factorize(rand_matrix, n_components=64, n_clusters=50)
    
    # Verify shapes
    assert consumer_emb.shape == (n_consumers, 64), f"Consumer embedding shape wrong: {consumer_emb.shape}"
    assert producer_emb.shape == (n_producers, 64), f"Producer embedding shape wrong: {producer_emb.shape}"
    
    # Verify normalization
    consumer_norms = np.linalg.norm(consumer_emb, axis=1)
    producer_norms = np.linalg.norm(producer_emb, axis=1)
    assert np.allclose(consumer_norms, 1.0, atol=1e-6), "Consumer embeddings not normalized"
    assert np.allclose(producer_norms, 1.0, atol=1e-6), "Producer embeddings not normalized"
    
    print("✓ Basic factorization test passed")

def test_alignment():
    """Test Procrustes alignment with known rotation"""
    print("\n=== Test 2: Alignment ===")
    
    # Create simple embeddings
    n_users = 10
    dim = 64
    
    # Create source embeddings
    source_consumer = np.random.randn(n_users, dim)
    source_consumer = source_consumer / np.linalg.norm(source_consumer, axis=1, keepdims=True)
    source_producer = np.random.randn(n_users, dim)
    source_producer = source_producer / np.linalg.norm(source_producer, axis=1, keepdims=True)
    
    # Create target by rotating source
    theta = np.pi/4  # 45-degree rotation
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    # Pad rotation matrix to full dimensionality
    full_rotation = np.eye(dim)
    full_rotation[:2, :2] = rotation
    
    target_consumer = source_consumer @ full_rotation
    
    # Test alignment
    aligned_consumer, aligned_producer = align_embeddings(
        source_consumer, 
        source_producer, 
        target_consumer,
        consumer_ids=list(range(n_users)),
        prev_consumer_ids=list(range(n_users))
    )
    
    # Verify alignment quality
    similarity = np.diag(aligned_consumer @ target_consumer.T)
    assert np.all(similarity > 0.99), f"Poor alignment: min similarity = {np.min(similarity):.4f}"
    
    # Verify normalization preserved
    assert np.allclose(np.linalg.norm(aligned_consumer, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(aligned_producer, axis=1), 1.0, atol=1e-6)
    
    print("✓ Alignment test passed")

def test_temporal_consistency():
    """Test alignment across multiple days with changing user sets"""
    print("\n=== Test 3: Temporal Consistency ===")
    
    # Create sequence of sparse matrices with overlapping but different users
    n_days = 3
    base_consumers = 100
    base_producers = 150
    matrices = []
    consumer_sets = []
    producer_sets = []
    
    for day in range(n_days):
        # Add some new users each day
        n_consumers = base_consumers + day * 10
        n_producers = base_producers + day * 5
        
        matrix = sp.random(n_consumers, n_producers, density=0.1, format='coo')
        matrices.append(matrix)
        consumer_sets.append(set(range(n_consumers)))
        producer_sets.append(set(range(n_producers)))
    
    # Run temporal factorization with fewer clusters
    previous_embeddings = None
    prev_consumer_ids = None
    all_consumer_emb = []
    
    for day in range(n_days):
        _, _, consumer_emb, producer_emb, _ = factorize(
            matrices[day],
            n_components=64,
            n_clusters=50,
            previous_embeddings=previous_embeddings,
            consumer_ids=list(consumer_sets[day]),
            prev_consumer_ids=prev_consumer_ids
        )
        
        all_consumer_emb.append(consumer_emb)
        previous_embeddings = (producer_emb, consumer_emb)
        prev_consumer_ids = list(consumer_sets[day])
    
    # Check temporal consistency
    for day in range(1, n_days):
        # Get common users
        common_users = consumer_sets[day] & consumer_sets[day-1]
        common_indices_curr = [i for i in range(len(consumer_sets[day])) if i in common_users]
        common_indices_prev = [i for i in range(len(consumer_sets[day-1])) if i in common_users]
        
        # Compare embeddings for common users
        curr_emb = all_consumer_emb[day][common_indices_curr]
        prev_emb = all_consumer_emb[day-1][common_indices_prev]
        
        similarity = np.mean(np.sum(curr_emb * prev_emb, axis=1))
        print(f"Average similarity between day {day-1} and {day}: {similarity:.4f}")
        assert similarity > 0.8, f"Poor temporal consistency: {similarity:.4f}"
    
    print("✓ Temporal consistency test passed")

def test_real_data():
    """Test with actual data from your system"""
    print("\n=== Test 4: Real Data ===")
    
    # Load a small sample of your real data
    # This is just an example - adjust paths and data loading as needed
    if os.path.exists('../DG_data/bluesky/user_dynamic_features.pkl'):
        with open('../DG_data/bluesky/user_dynamic_features.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Get two consecutive days
        dates = sorted(data.keys())
        day1, day2 = dates[:2]
        
        # Basic checks
        print(f"Testing dates: {datetime.fromtimestamp(day1)} -> {datetime.fromtimestamp(day2)}")
        print(f"Number of users: {len(data[day1])} -> {len(data[day2])}")
        
        # Check embedding properties
        for day in [day1, day2]:
            embeddings = np.array(list(data[day].values()))
            norms = np.linalg.norm(embeddings, axis=1)
            zero_vectors = np.sum(np.all(embeddings == 0, axis=1))
            
            print(f"\nDate: {datetime.fromtimestamp(day)}")
            print(f"Mean norm: {np.mean(norms):.6f}")
            print(f"Zero vectors: {zero_vectors}")
            print(f"Shape: {embeddings.shape}")
    else:
        print("Skipping real data test - no data file found")

if __name__ == "__main__":
    tests = [
        test_basic_factorization,
        test_alignment,
        test_temporal_consistency,
        test_real_data
    ]
    
    results = []
    for test in tests:
        try:
            test()
            results.append((test.__name__, "PASSED", None))
        except Exception as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
            results.append((test.__name__, "FAILED", str(e)))
            continue
    
    # Print summary
    print("\n=== Test Summary ===")
    for test_name, status, error in results:
        status_symbol = "✓" if status == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    # Final stats
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    total = len(results)
    print(f"\nPassed {passed}/{total} tests")