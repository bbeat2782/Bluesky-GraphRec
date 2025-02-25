import numpy as np

def dummy_candidate_generator(batch_src_node_ids, batch_node_interact_times):
    """
    A placeholder candidate generator that returns dummy candidates.
    Returns candidates in the same format as CandidateEdgeSampler.sample()
    """
    # Create a base set of candidate posts as np.int64
    dummy_candidate_posts = {np.int64(i) for i in range(5628585, 5628585 + 2000)}
    
    # Create dictionary with timestamps as keys and sets of np.int64 as values
    candidates_dict = {}
    for interact_time in batch_node_interact_times:
        # Convert to np.float64 to match the key type
        time_key = np.float64(interact_time)
        # Store as a set of np.int64 to match the format
        candidates_dict[time_key] = {np.int64(post_id) for post_id in dummy_candidate_posts}

    return candidates_dict