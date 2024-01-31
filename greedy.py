'''
description of data structures:
G.node[v]['embedding']   : input embedding of node v
G.node[v]['delta'] = H_k : the k-hop neighborhood of v
G.node[v]['H_alphak']    : the neighborhood of v within k+1 and alpha*k hop. it is different from my draft to save memory.
G.node[v]['DU']          : distribution uniformity of v
G.node[v]['nf']          : the normalization factor of v
G.node[v]['SIM']         : SIM(v, delta(S))
'''

import argparse
import sys
import math
import networkx as nx
import random
import heapq
import sys
import numpy as np
import time
from numpy.linalg import norm
import os
import json
from utils import load_data, get_cora_training_set


maxDistance = 0
minDistance = sys.maxsize
disRange = 0
PSIM = {}


def parse_args():
    parser = argparse.ArgumentParser(description='profit divergence minimization')
    parser.add_argument('--dataset', type=str, help='input dataset')
    parser.add_argument('--B', type=int, default=None, help='the labelling budget')
    parser.add_argument('--sim_metric', type=str, default='COSINE', help='the embedding similarity metric: ED or COSINE')
    parser.add_argument('--t', type=float, default=0.9999, help='the target similarity')
    parser.add_argument('--k', type=int, default=1, help='the number of aggregation iteration')
    parser.add_argument('--sample_size', type=int, default=0, help='preprocessing sample size, 0 means no sampling')
    return parser.parse_args()


def read_graph(adj, features):
    G = nx.from_scipy_sparse_matrix(adj)
    features = np.array(features)
    for v in G.nodes():
        G.node[v]['embedding'] = features[v]
        G.node[v]['SIM'] = 0
    return G


def SIM(embedding1, embedding2, sim_metric):
    global disRange
    if (sim_metric == 'COSINE'):
        return max(1e-10, np.dot(embedding1, embedding2)/(norm(embedding1)*norm(embedding2)))
    
    if (sim_metric == 'ED'):
        norm_distance = (norm(embedding1-embedding2)-minDistance)/disRange
        return 1/(1+norm_distance)


# compute G.node[v]['delta']=H_k, G.node[v]['H_alphak'], G.node[v]['DU'] and G.node[v]['nf'].
def preprocess(G, k, sim_metric, t, sample_size):
    global maxDistance, minDistance, disRange

    for v in G.nodes():
        G.node[v]['H_alphak'] = G.neighbors(v)

        if (k > 1):
            hopmarking = {v: 1}
            H_k = set()
            visited = set()
            visited.add(v)
            queue = [v]
            hop = 1
            
            while (queue and hop <= k):
                u = queue.pop(0)
                marking = -1
                for w in G.neighbors(u):
                    if (w not in visited):
                        # print(w)
                        visited.add(w)
                        queue.append(w)
                        
                        if (u in hopmarking):
                            marking = w

                        H_k.add(w)

                        if (sim_metric == 'ED'):
                            distance = norm(G.node[v]['embedding'] - G.node[w]['embedding'])
                            maxDistance = max(maxDistance, distance)
                            minDistance = min(distance, minDistance)

                if (u in hopmarking):
                    hop = hop + 1
                    if (marking > -1):
                        hopmarking[marking] = hop

            G.node[v]['delta'] = H_k
        elif (k == 1):
            G.node[v]['delta'] = set(G.neighbors(v))

    if (sample_size > 0):
        for v in G.nodes():
            G.node[v]['H_alphak'] = random.sample(G.node[v]['H_alphak'], min(sample_size, len(G.node[v]['H_alphak'])))

    if (sim_metric == 'ED'):
        disRange = maxDistance - minDistance
    
    # compute DU and nf

    for v in G.nodes():
        distances = []
        
        for w in G.node[v]['H_alphak']:
            small, large = min(w, v), max(w, v)
            if ((small, large) not in PSIM):
                PSIM[(small, large)] = SIM(G.node[v]['embedding'], G.node[w]['embedding'], sim_metric)
            distances.append(PSIM[(small, large)])

        distances.sort()

        if (len(distances) == 0):
            continue

        quartersum = 0
        totalsum = 0
        counter = 0
        limit = max(1, int(len(distances)/4))
        
        for dis in distances:
            if (counter < limit):
                counter += 1
                quartersum += dis
            totalsum += dis

        #print (quartersum,limit,totalsum,len(distances))
        #print ((quartersum/limit),totalsum/len(distances))        
        G.node[v]['DU']=(quartersum/limit)/(totalsum/len(distances))
        
        # nf        
        e=2.71828
        eta=-math.log(2/(t+1)-1,e)*G.node[v]['DU']
        G.node[v]['nf']=eta/totalsum
        #print ("quartersum:",quartersum," totalsum:",totalsum," length:",len(distances))
        #print ("DU:",G.node[v]['DU'], " eta:",eta," nf:",G.node[v]['nf']," equation:",2/(1+math.exp(-eta/G.node[v]['DU']))-1-t)
        #-math.log(2/(t+1)-1,e)/totalsum
        #2 / (1 + math.exp(simscore*math.log(2/(t+1)-1,e)/totalsum)) - 1


# compute the representative score
def Rscore(G, v, simscore):
    return 2/(1+math.exp(-simscore/G.node[v]['DU']*G.node[v]['nf']))-1


def computeGain(G, v, S, delta_S, sim_metric):
    I = set()
    triangle_R = 0
    triangle_delta = set()
    tempSIM = {}
    for u in G.node[v]['delta']:
        if (u not in delta_S):
            triangle_delta.add(u)

    for r in triangle_delta:
        for u in G.node[r]['H_alphak']:
            if (u in delta_S or u in G.node[v]['delta']):
                continue

            if (u not in I):
                I.add(u)
                tempSIM[u] = G.node[u]['SIM']
            small, large = min(u, r), max(u, r)
            tempSIM[u] += PSIM[(small,large)]
    for u in I:
        R1 = Rscore(G, u, tempSIM[u])
        triangle_R += Rscore(G, u, tempSIM[u]) - Rscore(G, u, G.node[u]['SIM'])

    return len(triangle_delta) + triangle_R


# greedy with early termination
def greedyET(G, B, sim_metric, training_set):
    S = list()
    delta_S = set()
    PQ = []
    count = 0
    t_start = time.time()

    while(len(S) < B):
        count += 1
        if (len(S) == 0):
            start = time.time()
            for v in training_set:
                ub = computeGain(G, v, S, delta_S, sim_metric)
                heapq.heappush(PQ, (-ub, v))
            minus_ub, bestnode = heapq.heappop(PQ)        
        else:
            maxgain = -1
            visited = []
            while(PQ):
                minus_ub, v = heapq.heappop(PQ)
                ub = computeGain(G, v, S, delta_S, sim_metric)
                visited.append((ub, v))
                if (ub > maxgain):
                    maxgain = ub
                    bestnode = v
                if (PQ and maxgain > -PQ[0][0]):
                    break
            if (len(S) < B-1):
                for ub, v in visited:
                    if (v != bestnode):
                        heapq.heappush(PQ, (-ub, v))

        # update G.node[u]['SIM'] based on bestnode, since S will become S.add(bestnode).
        I = set()
        triangle_delta = set()
        for u in G.node[bestnode]['delta']:
            if (u not in delta_S):
                triangle_delta.add(u)

        for r in triangle_delta:
            for u in G.node[r]['H_alphak']:
                if (u in delta_S or u in G.node[bestnode]['delta']):
                    continue
                # G.node[u]['SIM'] += SIM(G.node[u]['embedding'], G.node[r]['embedding'], sim_metric)
                small, large = min(u, r), max(u, r)            
                G.node[u]['SIM'] += PSIM[(small, large)]

        S.append(bestnode)
        t_cur = time.time() - t_start
        print("| #{} | seed selection time elapsed: {:.4f} s | seed selected: {}".format(count, t_cur, bestnode))

        for u in G.node[bestnode]['delta']:
            if (u not in delta_S):
                delta_S.add(u)
    return S


def save_seeds(S, dataset, sim_metric, B, t, k, ss):
    output_dir = "output_seeds"
    os.makedirs(output_dir, exist_ok=True)

    method = "greedyET_{}_t{}_k{}_ss{}_b{}".format(sim_metric, t, k, ss, B)    
    with open("{}/{}_{}.json".format(output_dir, dataset, method), 'w') as f:
        json.dump(S, f)


def main(args):
    adj, features, _labels, idx_val, idx_test = load_data(dataset=args.dataset)
    G = read_graph(adj, features)
    
    t_pre_start = time.time()
    preprocess(G, args.k, args.sim_metric, args.t, args.sample_size)
    t_pre_end = time.time()
    print("| pre-processing time elapsed: {:.4f} s".format(t_pre_end - t_pre_start))
    
    training_set = get_cora_training_set(idx_val, idx_test)
    S = greedyET(G, args.B, args.sim_metric, training_set)
    save_seeds(S, args.dataset, args.sim_metric, args.B, args.t, args.k, args.sample_size)

        
if __name__ == '__main__':
    args = parse_args()
    main(args)
