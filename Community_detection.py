from numpy.random.mtrand import choice
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools 

def community_detection(nodes, edges, population=200, generation=30, r=1.5):
    
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    Adj = nx.adjacency_matrix(graph)
    nodes_length = len(graph.nodes())

    pos = nx.spring_layout(graph, seed=42)  # Pozicioniranje čvorova
    nx.draw(graph, pos) 

    node_labels = {node: str(node) for node in nodes}
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_color='black')
    plt.show()
    
    chromosomes = [generate_chrom1(nodes_length, Adj, i) for i in range(population)]
    subsets = [find_subsets(chrom, Adj) for chrom in chromosomes]
    community_scores = [community_score(chrom, subs, r, Adj) for chrom, subs in zip(chromosomes, subsets)]
    
    gen = 0
    while gen < generation:
        new_chromosomes = []
        new_community_scores = []
        total_cs = np.sum(community_scores)
        normalized_cs = [cs / total_cs for cs in community_scores]
        
        elite_cs = []
        elite_chrom = []
        if gen % 5 == 0:
            elite_indices = np.argsort(community_scores)[-int(np.floor(population / 10)):]
            for i in elite_indices:
                elite_cs.append(community_scores[i])
                elite_chrom.append(chromosomes[i])
            
         
        for _ in range(int(np.floor(population / 10))):
            
            p1, p2 = random.choices(chromosomes, weights = normalized_cs, k = 2) 
            
            child = uniform_crossover(p1, p2)
            if len(child) == 0:
                continue
            
            child = mutation1(child, Adj, 0.2)
            new_chromosomes.append(child)
            new_subs = find_subsets(child, Adj)
            new_cs = community_score(child, new_subs, r, Adj)
            new_community_scores.append(new_cs)
            
        chromosomes.extend(new_chromosomes)
        community_scores.extend(new_community_scores)
        chromosomes.extend(elite_chrom)
        community_scores.extend(elite_cs)
        chromosomes_scores = list(zip(chromosomes, community_scores))
        chromosomes_scores.sort(key=lambda x: x[1], reverse=True)
        chromosomes = [chromosome for chromosome, _ in chromosomes_scores[:population]]
        community_scores = [score for _, score in chromosomes_scores[:population]]
        chromosomes = list(chromosomes)
        community_scores = list(community_scores)

        gen += 1
    
    best_chrom = max(chromosomes, key=lambda chrom: community_score(chrom, find_subsets(chrom, Adj), r, Adj))
    best_subsets = find_subsets(best_chrom, Adj)
    nodes_list = list(graph.nodes())
    print("fitnes")
    print(community_score(best_chrom, best_subsets, r, Adj) )
    
    result = []
    for subs in best_subsets:
        subset = [nodes_list[n] for n in subs]
        result.append(subset)
    return result


def generate_chrom1(nodes_length, Adj, i):
  
    chrom = np.zeros(nodes_length, dtype=int)
    for x in range(nodes_length):
        
        non_zero_indices = Adj.indices[Adj.indptr[x]:Adj.indptr[x+1]]
        
        np.random.shuffle(non_zero_indices)
        chrom[x] = non_zero_indices[0]
    return chrom

def merge_subsets(sub):
    
    arr =[]
    to_skip=[]
    for s in range(len(sub)):
        if sub[s] not in to_skip:
            new = sub[s]
            for x in sub:
                if sub[s] & x:
                    new = new | x
                    to_skip.append(x)
            arr.append(new)
    
    return arr
   
def find_subsets(chrom, Adj):
    
    sub = [{x, chrom[x]} for x in range(len(chrom))] 
    i = 0
    while i < len(sub):
        candidate = merge_subsets(sub)
        if candidate == sub:
            break
        sub = candidate
        i += 1
    return sub


def community_score(chrom,subsets,r,Adj):
    matrix = Adj.toarray()
    CS=0
    for s in subsets:
        submatrix = np.zeros((len(chrom),len(chrom)),dtype=int)
        for i in s:
            for j in s:
                submatrix[i][j]=matrix[i][j]
        M=0
        v=0
        for row in list(s):
            row_mean = np.sum(submatrix[row])/len(s) 
            v+=np.sum(submatrix[row])
            M+=(row_mean**r)/len(s)
        CS+=M*v
    return CS




def uniform_crossover(parent_1, parent_2):
    
    length = len(parent_1)
    mask = np.random.randint(2, size=length)
    child = np.zeros(length,dtype=int)
    for i in range(len(mask)):
        if mask[i] == 1:
            child[i]=parent_1[i]
        else:
            child[i]=parent_2[i]
    return child


def mutation1(chrom, Adj, mutation_rate):
    if np.random.random_sample() < mutation_rate:
        neighbor = []
        mutant = np.random.randint(1, len(chrom)) #generisanje na kom mestu u jedinci cemo da mutiramo (ovo ja dalje nazivam mutant znaci nije mi cela jedinka mutant nego na tom mestu)
        midstep = Adj[:, [mutant]].toarray() #izdvajanje komsija od mutanta
        row = midstep[:, 0]  # Convert 1D slice to 2D slice using explicit indices 
        neighbor = [i for i in range(len(row)) if row[i] == 1] #izdvajanje indeksa komsija 
        if len(neighbor) > 1: #ovaj if je za proveru da l postoji uopste komsija
            neighbor.remove(chrom[mutant]) #da se ne pogodi da bude isti komsija ko pre na mestu koje mutiramo
            to_change = int(np.floor(np.random.random_sample() * len(neighbor))) #random generisanje kojeg cemo komsiju od preostalih sad da stavimo na mesto mutanta
            chrom[mutant] = neighbor[to_change] #stavljanje novog komsije na mesto mutanta
    return chrom




nodes = [0,1,2,3,4,5,6,7,8,9,10]
edges = [(0,1),(0,4),(1,2),(2,3),(1,3),(3,0),(0,2),(4,5),(5,6),(6,7),(10,8),(10,9),(8,9),(8,7),(9,7),(7,10)]




res = community_detection(nodes,edges)
print("GA")
print(res)




# BRUTE FORCE

nodes = [0,1,2,3,4,5,6,7,8,9,10]
edges = [(0,1),(0,4),(1,2),(2,3),(1,3),(3,0),(0,2),(4,5),(5,6),(6,7),(10,8),(10,9),(8,9),(8,7),(9,7),(7,10)]

  
def communities_using_brute(gfg):
  nodes = gfg.nodes()
  n = gfg.number_of_nodes()
  first_community = []
    
  for i in range(1, n//2 + 1):
    c = [list(a) for a in itertools.combinations(nodes, i)]
    first_community.extend(c)
  
  second_community = []
  
  for i in range(len(first_community)):
    b = list(set(nodes)-set(first_community[i]))
    second_community.append(b)
  
  intra_edges1 = []
  intra_edges2 = []
  inter_edges = []
      
  ratio = []  
  
  for i in range(len(first_community)):
    intra_edges1.append(gfg.subgraph(first_community[i]).number_of_edges())
  
  for i in range(len(second_community)):
    intra_edges2.append(gfg.subgraph(second_community[i]).number_of_edges())
  
  e = gfg.number_of_edges()
  
  for i in range(len(first_community)):
    inter_edges.append(e-intra_edges1[i]-intra_edges2[i])
  
  for i in range(len(first_community)):
    ratio.append((float(intra_edges1[i]+intra_edges2[i]))/inter_edges[i])
  
  maxV=max(ratio)
  mindex=ratio.index(maxV)
  
  print('[ ', first_community[mindex], ' ] , [ ', second_community[mindex], ' ]')


graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)
communities_using_brute(graph)





def generate_graph():
    for i in range(2) :
    
      nodes = [0,1,2,3,4,5,6,7,8,9,10]
      edges = [(0,1),(0,4),(1,2),(2,3),(1,3),(3,0),(0,2),(4,5),(5,6),(6,7),(10,8),(10,9),(8,9),(8,7),(9,7),(7,10)]
    
      num_nodes = 31 
    
      nodes = list(range(num_nodes))
    
     
      edges = []
      for node in nodes:
        neighbor = random.choice([n for n in nodes if n != node])
        edges.append((node, neighbor))
    
    
    
      res = community_detection(nodes,edges)
      print(res)
      #brute force zbog slozenosti ne moze da se zavrsi uspesno u predvidjenom vremenskom roku   
generate_graph()
