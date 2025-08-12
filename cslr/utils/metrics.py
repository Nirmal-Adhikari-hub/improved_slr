
from typing import List

def _levenshtein(a: List[str], b: List[str]) -> int:
    """
    Calculate the Levenshtein distance between two lists of strings.
    Args:
        a (List[str]): First list of strings.
        b (List[str]): Second list of strings.  
    
    Returns:
        int: The Levenshtein distance between the two lists.
    """
    # classic DP
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[n][m]

def wer(ref: List[str], hyp: List[str]) -> float:
    """Word Error Rate between reference and hypothesis gloss sequences (as token lists)."""
    if len(ref) == 0:
        return float(len(hyp) > 0)
    dist = _levenshtein(ref, hyp)
    return dist / max(1, len(ref))
