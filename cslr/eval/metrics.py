from typing import List, Tuple

def _levenshtein(ref: List[str], hyp: List[str]) -> Tuple[int,int,int]:
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0] = i
    for j in range(1, m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    # backtrack counts
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i-1][j] + 1: D += 1; i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1: I += 1; j -= 1
        else:
            if i > 0 and j > 0 and ref[i-1] != hyp[j-1]: S += 1
            i -= 1; j -= 1
    return S, D, I

def wer_percent(ref: List[str], hyp: List[str]):
    N = len(ref)
    if N == 0: return 0.0, (0,0,0,0)
    S, D, I = _levenshtein(ref, hyp)
    return 100.0 * (S + D + I) / N, (S, D, I, N)
