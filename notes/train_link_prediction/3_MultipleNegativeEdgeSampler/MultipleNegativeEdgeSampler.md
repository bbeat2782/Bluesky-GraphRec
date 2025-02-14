class MultipleNegativeEdgeSampler:
    def __init__(self, src_node_ids, dst_node_ids, interact_times, size_factor=4, seed=None):
        """
        Stores historical interactions and samples multiple negative edges
        
        Example data:
        src_node_ids = [1, 2, 1, 3]  # Users
        dst_node_ids = [4, 5, 6, 4]  # Posts
        interact_times = [100, 200, 300, 400]  # Timestamps
        """

    Timeline of Interactions:
    Time:  100    200    300    400
        │      │      │      │
        v      v      v      v
    User1: 4      -      6      -
    User2: -      5      -      -
    User3: -      -      -      4

    Internal Storage:
    sources_dict = {
        100: {1: [4]},    # At t=100, user1 interacted with post4
        200: {2: [5]},    # At t=200, user2 interacted with post5
        300: {1: [6]},    # At t=300, user1 interacted with post6
        400: {3: [4]}     # At t=400, user3 interacted with post4
    }

---------------------------------------------------------------------------

def sample(self, size=2, current_batch_start_time=[350, 450]):
    """
    Input:
    - size: number of positive edges we're sampling negatives for
    - current_batch_start_time: start and end times of the current batch
    """

---------------------------------------------------------------------------

Input:
------
sample_source_indices = [13700579, 13700580, 13700581, ...]  # Batch of 512 users

Output:
-------
batch_neg_dst_node_ids.shape = (512, 4)  # Each user gets 4 negative posts

Example for first 3 users:
User 13700579's negatives: [4145387, 303234, 550967, 5043429]
User 13700580's negatives: [626064, 2179717, 980280, 106981]
User 13700581's negatives: [5488843, 4270608, 3336469, 1635801]

Visual Format:
User      Neg1     Neg2    Neg3    Neg4
13700579  4145387  303234  550967  5043429
13700580  626064   2179717 980280  106981
13700581  5488843  4270608 3336469 1635801
...       ...      ...     ...     ...