def dummy_candidate_generator(batch_src_node_ids, batch_node_interact_times):
    """
    A placeholder candidate generator that returns dummy candidates.
    For now, it ignores user and time and just returns a fixed set of post IDs.
    """
    candidate_posts = [101, 102, 103, 104, 105]  # Example post IDs - replace with actual post IDs from your data if you know some

    candidates_dict = {}
    for interact_time in batch_node_interact_times: # Use each time in batch, even if dummy
        candidates_dict[interact_time] = candidate_posts

    return candidates_dict