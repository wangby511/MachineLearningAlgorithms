from __future__ import print_function
import json
import numpy as np
import sys


def forward(pi, A, B, O):
    """
    Forward algorithm

    Inputs:
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})
    """
    S = len(pi)
    N = len(O)
    alpha = np.zeros([S, N])
    ###################################################
    # Q3.1 Edit here
    ###################################################
    for i in range(N):
        for j in range(S):
            if(i == 0):
                alpha[j,0] = pi[j] * B[j,O[0]]
            else:
                for k in range(S):
                    alpha[j,i] += alpha[k, i - 1] * A[k,j]
                alpha[j,i] = alpha[j,i] * B[j,O[i]]
    return alpha


def backward(pi, A, B, O):
    """
    Backward algorithm

    Inputs:
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
    """
    S = len(pi)
    N = len(O)
    beta = np.zeros([S, N])
    ###################################################
    # Q3.1 Edit here
    ###################################################
    beta[:,N - 1] = 1
    for i in range(N - 2,-1,-1):
        for j in range(S):
            for s in range(S):
                beta[j,i] += beta[s, i + 1] * A[j,s] * B[s,O[i + 1]]
    return beta


def seqprob_forward(alpha):
    """
    Total probability of observing the whole sequence using the forward messages

    Inputs:
    - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})

    Returns:
    - prob: A float number of P(X_{1:N}=O)
    """
    prob = 0
    ###################################################
    # Q3.2 Edit here
    ###################################################
    N = alpha.shape[1]
    prob = np.sum(alpha[:,N - 1])

    return prob


def seqprob_backward(beta, pi, B, O):
    """
    Total probability of observing the whole sequence using the backward messages

    Inputs:
    - beta: A numpy array beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence
        (in terms of the observation index, not the actual symbol)

    Returns:
    - prob: A float number of P(X_{1:N}=O)
    """
    prob = 0
    ###################################################
    # Q3.2 Edit here
    ###################################################
    S = beta.shape[0]
    for j in range(S):
        prob += beta[j,0] * pi[j] * B[j,O[0]]

    return prob


def viterbi(pi, A, B, O):
    """
    Viterbi algorithm

    Inputs:
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - path: A list of the most likely hidden state path (in terms of the state index)
    """
    path = []
    ###################################################
    # Q3.3 Edit here
    ###################################################
    S = len(pi)
    N = len(O)
    Q = np.zeros((S,N))
    best = np.zeros((S,N))
    for i in range(S):
        Q[i,0] = pi[i] * B[i,O[0]]
    for i in range(1,N):
        for j in range(S):
            Q[j,i] = 0
            best[j,i] = -1
            maxposs = -np.inf
            for k in range(S):
                r = A[k,j] * B[j,O[i]] * Q[k,i - 1]
                if r > maxposs:
                    maxposs = r
                    best[j,i] = k
                    Q[j,i] = r
                # print("i =",i,",j =",j,",maxposs = ",maxposs)
    finalPoss = -1.0
    finalBestO = -1
    for i in range(S):
        if Q[i,N - 1] > finalPoss:
            finalPoss = Q[i,N - 1]
            finalBestO = i

    current = finalBestO
    # print("current = ",current)
    path.append(current)
    for i in range(N - 2,-1,-1):
        # print("current =", current)
        current = int(best[current, i + 1])
        path.append(current)
    path.reverse()
    return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
    model_file = sys.argv[1]
    Osymbols = sys.argv[2]

    #### load data ####
    with open(model_file, 'r') as f:
        data = json.load(f)
    A = np.array(data['A'])
    B = np.array(data['B'])
    pi = np.array(data['pi'])
    #### observation symbols #####
    obs_symbols = data['observations']
    #### state symbols #####
    states_symbols = data['states']

    N = len(Osymbols)
    O = [obs_symbols[j] for j in Osymbols]

    alpha = forward(pi, A, B, O)
    beta = backward(pi, A, B, O)

    prob1 = seqprob_forward(alpha)
    prob2 = seqprob_backward(beta, pi, B, O)
    print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

    viterbi_path = viterbi(pi, A, B, O)

    print('Viterbi best path is ')
    for j in viterbi_path:
        print(states_symbols[j], end=' ')


if __name__ == "__main__":
    main()
