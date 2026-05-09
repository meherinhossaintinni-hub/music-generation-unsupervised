import numpy as np

# =========================
# 1. PITCH HISTOGRAM SIMILARITY
# =========================

def pitch_histogram_similarity(p, q):
    """
    L1 distance between pitch distributions
    """
    return np.sum(np.abs(p - q))


# =========================
# 2. RHYTHM DIVERSITY SCORE
# =========================

def rhythm_diversity(sequence):
    """
    #unique durations / total notes
    """
    flat = sequence.flatten()
    return len(np.unique(flat)) / len(flat)


# =========================
# 3. REPETITION RATIO
# =========================

def repetition_ratio(sequence):
    """
    repeated patterns / total patterns
    """
    flat = sequence.flatten()

    unique_patterns = len(np.unique(flat))
    total_patterns = len(flat)

    return 1 - (unique_patterns / total_patterns)


# =========================
# 4. FULL MODEL COMPARISON
# =========================

def compare_models(task1, vae):
    print("\n===== BASELINE COMPARISON =====")

    print("Task 1 Rhythm Diversity:", np.mean([rhythm_diversity(s) for s in task1]))
    print("VAE Rhythm Diversity:", np.mean([rhythm_diversity(s) for s in vae]))

    print("\nTask 1 Repetition:", np.mean([repetition_ratio(s) for s in task1]))
    print("VAE Repetition:", np.mean([repetition_ratio(s) for s in vae]))


# =========================
# 5. HUMAN LISTENING SCORE (PLACEHOLDER)
# =========================

def human_score():
    """
    This is NOT computed automatically.
    Fill after survey (Google Form / classmates)
    """
    return {
        "Task 1": 3.8,
        "Task 2": 4.2,
        "Task 3": 4.0
    }