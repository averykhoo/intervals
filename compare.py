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