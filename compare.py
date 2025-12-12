import heapq
import itertools
import random
import timeit
from enum import IntEnum


# --- 1. Toplogical Definitions (From our discussion) ---

class Eps(IntEnum):
    OPEN_END = -2
    NEG_ZERO = -1
    CLOSED = 0
    OPEN_START = 2


# We represent intervals as tuples: ((start_val, start_eps), (end_val, end_eps))
# Merge rule: diff <= 3

def fast_sweep(sorted_iterable):
    """
    Consumes a sorted iterable (list or generator) and merges intervals
    based on the {-2, -1, 0, 2} topology with diff <= 3.
    """
    iterator = iter(sorted_iterable)
    try:
        current = next(iterator)
    except StopIteration:
        return []

    merged = []

    # Unpack current for speed (avoiding attribute lookups inside loop)
    # Structure: ((s_val, s_eps), (e_val, e_eps))
    curr_start, curr_end = current
    c_end_val, c_end_eps = curr_end

    for next_iv in iterator:
        # Unpack next
        n_start, n_end = next_iv
        n_start_val, n_start_eps = n_start

        # --- MERGE LOGIC ---

        # 1. Check strict overlap
        is_overlapping = n_start_val < c_end_val

        # 2. Check Adjacency (if values equal)
        is_touching = False
        if not is_overlapping and n_start_val == c_end_val:
            # Universal Rule: diff <= 3
            if (n_start_eps - c_end_eps) <= 3:
                is_touching = True

        if is_overlapping or is_touching:
            # Merge: Extend current end if next goes further
            # Max logic: Compare Value, then Epsilon
            n_end_val, n_end_eps = n_end
            if (n_end_val > c_end_val) or \
                    (n_end_val == c_end_val and n_end_eps > c_end_eps):
                c_end_val, c_end_eps = n_end_val, n_end_eps
                curr_end = n_end  # Update the tuple reference
        else:
            # Gap detected: Commit and Reset
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_iv
            c_end_val, c_end_eps = curr_end

    # Commit last
    merged.append((curr_start, curr_end))
    return merged


# --- 2. Data Generation ---

def generate_sorted_chunk(size, start_offset=0):
    """Generates a list of sorted, disjoint intervals."""
    data = []
    current_val = start_offset
    for _ in range(size):
        # Move forward a bit (gap)
        gap = random.uniform(0.1, 2.0)
        start_val = current_val + gap

        # Determine width
        width = random.uniform(0.1, 2.0)
        end_val = start_val + width

        # Randomize epsilons (using our set)
        s_eps = random.choice([-2, -1, 0, 2])
        e_eps = random.choice([-2, -1, 0, 2])

        data.append(((start_val, s_eps), (end_val, e_eps)))
        current_val = end_val

    return data


# --- 3. Strategies to Test ---

def run_timsort_merge(list_of_lists):
    """
    Strategy A: Concatenate all lists, Sort them, then Sweep.
    """
    # itertools.chain is faster than sum(lists, [])
    flat_list = list(itertools.chain.from_iterable(list_of_lists))
    flat_list.sort()  # Timsort (C-optimized)
    return fast_sweep(flat_list)


def run_heapq_merge(list_of_lists):
    """
    Strategy B: Use heapq.merge to stream sorted data, then Sweep.
    """
    # heapq.merge assumes inputs are already sorted (which they are)
    # It returns an iterator, so memory usage is O(1) for the stream
    stream = heapq.merge(*list_of_lists)
    return fast_sweep(stream)


def run_unsorted_process(raw_list):
    """
    Strategy C: Standard processing of raw user input.
    """
    # We must sort in place or copy-sort
    raw_list.sort()
    return fast_sweep(raw_list)


# --- 4. The Benchmark Runner ---

def benchmark():
    N_ITEMS = 50000
    K_LISTS = 2  # Case 1: Binary Operation (A | B)
    K_MANY = 10  # Case 2: Multi-set Union

    print(f"--- Generating Data ({N_ITEMS} items per list) ---")

    # Dataset A: 2 sorted lists (for A | B)
    list_a = generate_sorted_chunk(N_ITEMS, 0)
    list_b = generate_sorted_chunk(N_ITEMS, 0)  # Overlapping range
    data_2 = [list_a, list_b]

    # Dataset B: 10 sorted lists (for Union(A, B, C...))
    data_10 = [generate_sorted_chunk(N_ITEMS // 5, 0) for _ in range(K_MANY)]

    # Dataset C: One giant unsorted list (User Input)
    data_unsorted = generate_sorted_chunk(N_ITEMS, 0)
    random.shuffle(data_unsorted)

    print(f"--- Benchmark Results ---")

    # 1. Timsort vs Heapq (2 Lists)
    t_tim_2 = timeit.timeit(lambda: run_timsort_merge(data_2), number=50)
    t_heap_2 = timeit.timeit(lambda: run_heapq_merge(data_2), number=50)

    print(f"\n1. Merging 2 Sorted Lists (Total items: {2 * N_ITEMS})")
    print(f"   Timsort + Sweep: {t_tim_2:.4f}s")
    print(f"   Heapq   + Sweep: {t_heap_2:.4f}s")
    if t_tim_2 < t_heap_2:
        print(f"   >> Timsort is {t_heap_2 / t_tim_2:.2f}x faster")
    else:
        print(f"   >> Heapq is {t_tim_2 / t_heap_2:.2f}x faster")

    # 2. Timsort vs Heapq (10 Lists)
    t_tim_10 = timeit.timeit(lambda: run_timsort_merge(data_10), number=50)
    t_heap_10 = timeit.timeit(lambda: run_heapq_merge(data_10), number=50)

    print(f"\n2. Merging 10 Sorted Lists (Total items: {len(list(itertools.chain(*data_10)))})")
    print(f"   Timsort + Sweep: {t_tim_10:.4f}s")
    print(f"   Heapq   + Sweep: {t_heap_10:.4f}s")

    # 3. Unsorted Input
    # We copy the list inside the lambda so the sort doesn't affect the next run
    t_unsorted = timeit.timeit(lambda: run_unsorted_process(data_unsorted[:]), number=50)

    print(f"\n3. Processing Unsorted User Input (Items: {N_ITEMS})")
    print(f"   Sort + Sweep:    {t_unsorted:.4f}s")


if __name__ == "__main__":
    benchmark()

r"""
C:\Python313\python.exe C:\Users\avery\PycharmProjects\intervals\compare.py 
--- Generating Data (50000 items per list) ---
--- Benchmark Results ---

1. Merging 2 Sorted Lists (Total items: 100000)
   Timsort + Sweep: 1.6277s
   Heapq   + Sweep: 1.7871s
   >> Timsort is 1.10x faster

2. Merging 10 Sorted Lists (Total items: 100000)
   Timsort + Sweep: 2.4272s
   Heapq   + Sweep: 2.8719s

3. Processing Unsorted User Input (Items: 50000)
   Sort + Sweep:    4.0854s

Process finished with exit code 0

"""

import timeit
import random
import heapq
import itertools
import bisect
from enum import IntEnum


# --- 1. Topology & optimized Sweep ---

class Eps(IntEnum):
    OPEN_END = -2
    NEG_ZERO = -1
    CLOSED = 0
    OPEN_START = 2


def optimized_sweep(sorted_iterable):
    """
    Highly optimized sweep.
    Assumes input is ((s_val, s_eps), (e_val, e_eps)).
    """
    iterator = iter(sorted_iterable)
    try:
        # Pre-fetch first item to init logic
        # Unpack immediately into locals (C-stack speed)
        curr_rec = next(iterator)
        (c_s_val, c_s_eps), (c_e_val, c_e_eps) = curr_rec
    except StopIteration:
        return []

    merged = []

    # Cache append method to avoid dot lookup in loop
    merged_append = merged.append

    for next_rec in iterator:
        # Unpack next
        (n_s_val, n_s_eps), (n_e_val, n_e_eps) = next_rec

        # LOGIC:
        # 1. Overlap: next_start < curr_end
        # 2. Touch:   next_start == curr_end AND (next_eps - curr_eps <= 3)

        # Note: We use nested ifs because it's faster than constructing boolean objects
        if n_s_val < c_e_val:
            # Overlap -> Merge
            # Logic: curr_end = max(curr_end, next_end)
            if (n_e_val > c_e_val) or (n_e_val == c_e_val and n_e_eps > c_e_eps):
                c_e_val, c_e_eps = n_e_val, n_e_eps
        elif n_s_val == c_e_val:
            if (n_s_eps - c_e_eps) <= 3:
                # Touch -> Merge
                if (n_e_val > c_e_val) or (n_e_val == c_e_val and n_e_eps > c_e_eps):
                    c_e_val, c_e_eps = n_e_val, n_e_eps
            else:
                # Disjoint -> Commit Current
                merged_append(((c_s_val, c_s_eps), (c_e_val, c_e_eps)))
                c_s_val, c_s_eps = n_s_val, n_s_eps
                c_e_val, c_e_eps = n_e_val, n_e_eps
        else:
            # Disjoint -> Commit Current
            merged_append(((c_s_val, c_s_eps), (c_e_val, c_e_eps)))
            c_s_val, c_s_eps = n_s_val, n_s_eps
            c_e_val, c_e_eps = n_e_val, n_e_eps

    # Commit last
    merged_append(((c_s_val, c_s_eps), (c_e_val, c_e_eps)))
    return merged


# --- 2. Data Gen ---

def generate_sorted_chunk(size, start_offset=0):
    data = []
    current_val = start_offset
    for _ in range(size):
        start_val = current_val + random.uniform(0.1, 2.0)
        end_val = start_val + random.uniform(0.1, 2.0)
        s_eps = random.choice([-2, -1, 0, 2])
        e_eps = random.choice([-2, -1, 0, 2])
        data.append(((start_val, s_eps), (end_val, e_eps)))
        current_val = end_val
    return data


# --- 3. Strategies ---

def run_timsort_no_key(list_of_lists):
    """Concatenate and Sort without a key lambda (uses C-level tuple compare)."""
    flat_list = list(itertools.chain.from_iterable(list_of_lists))
    flat_list.sort()  # Native Tuple Comparison
    return optimized_sweep(flat_list)


def run_heapq_merge(list_of_lists):
    """Stream merge."""
    stream = heapq.merge(*list_of_lists)
    return optimized_sweep(stream)


def run_incremental_bisect(base_list, new_items):
    """
    Simulates adding 'new_items' one by one to 'base_list' using bisect.
    This mimics a user building an interval set iteratively.
    """
    # Work on a copy
    current_list = list(base_list)

    for item in new_items:
        # 1. Bisect to find spot
        # We search by the whole tuple, which effectively sorts by start
        idx = bisect.bisect_left(current_list, item)

        # 2. Insert
        current_list.insert(idx, item)

        # 3. Local Merge (Simplified for benchmark)
        # In reality, we would check idx-1 and idx+1.
        # Here we just want to measure the cost of insert vs sort.
        # We will do a full sweep at the end to be fair?
        # No, the point is to avoid the full sweep.
        # We will assume a 'Local Merge Cost' is negligible (O(1))
        # and just benchmark the insertion cost O(N).

    return current_list


def run_incremental_sort(base_list, new_items):
    """
    Simulates adding 'new_items' by appending and re-sorting every time.
    """
    current_list = list(base_list)
    for item in new_items:
        current_list.append(item)
        current_list.sort()  # Resort the whole thing every time
    return current_list


# --- 4. Runner ---

def benchmark():
    N_ITEMS = 50000
    N_NEW = 100  # Number of items to insert incrementally

    print(f"--- Generating Data ---")
    list_a = generate_sorted_chunk(N_ITEMS, 0)
    list_b = generate_sorted_chunk(N_ITEMS, 0)
    data_2 = [list_a, list_b]

    # For incremental test: one big list and a few random new items
    data_large = generate_sorted_chunk(N_ITEMS, 0)
    new_items = generate_sorted_chunk(N_NEW, 0)
    # Scatter new items so they aren't all at the end
    new_items = [((x[0][0] * 0.5, x[0][1]), x[1]) for x in new_items]

    print(f"--- 1. Bulk Merge (A | B) ---")
    t_tim = timeit.timeit(lambda: run_timsort_no_key(data_2), number=50)
    t_heap = timeit.timeit(lambda: run_heapq_merge(data_2), number=50)

    print(f"   Timsort: {t_tim:.4f}s")
    print(f"   Heapq:   {t_heap:.4f}s")

    print(f"\n--- 2. Incremental Updates (Adding {N_NEW} items to {N_ITEMS}) ---")
    # Bisect strategy
    t_bisect = timeit.timeit(lambda: run_incremental_bisect(data_large, new_items), number=20)
    # Naive Append+Sort strategy
    t_re_sort = timeit.timeit(lambda: run_incremental_sort(data_large, new_items), number=20)

    print(f"   Bisect Insert:   {t_bisect:.4f}s")
    print(f"   Append + Sort:   {t_re_sort:.4f}s")
    print(f"   >> Bisect is {t_re_sort / t_bisect:.1f}x faster")


if __name__ == "__main__":
    benchmark()

r"""
C:\Python313\python.exe C:\Users\avery\PycharmProjects\intervals\compare.py 
--- Generating Data ---
--- 1. Bulk Merge (A | B) ---
   Timsort: 2.0613s
   Heapq:   2.1715s

--- 2. Incremental Updates (Adding 100 items to 50000) ---
   Bisect Insert:   0.0818s
   Append + Sort:   6.7077s
   >> Bisect is 82.0x faster

Process finished with exit code 0

"""