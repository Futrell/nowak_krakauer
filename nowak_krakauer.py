""" Implementation of basic language evolution model from Nowak & Krakauer (1999) """
import math
import random
import itertools
import torch

def random_stochastic_matrix(shape, temperature): # higher temperature -> more uniform initialization
    energies = 1/temperature * torch.randn(shape)
    return torch.softmax(energies, dim=-1)

def random_langs(num_langs, num_meanings, num_signals, temperature=1):
    P = random_stochastic_matrix((num_langs, num_meanings, num_signals), temperature)
    Q = random_stochastic_matrix((num_langs, num_signals, num_meanings), temperature)
    return P, Q

def basic_population_fitness(Ps, Qs, lam=1/2):
    # form tensor of shape B x B x M x S x M -- given agent A and agent B, for A's meaning M and signal S, probability that B receives M'
    R = Ps[:, None, :, :, None] * Qs[None, :, None, :, :]
    
    # an agent doesn't interact with itself, so blank out the entries R[i,i,...]
    agents = range(R.shape[0])
    R[agents, agents] = 0
    
    # now get fitness for each agent
    return lam * torch.einsum('abiji -> a', R) + (1-lam) * torch.einsum('abiji -> b', R)

def mutate(p, num_samples):
    eye = torch.eye(p.shape[-1])
    sample_indices = torch.stack([torch.multinomial(sub_p, num_samples, replacement=True) for sub_p in p])
    samples = eye[sample_indices]
    return samples.mean(axis=-2)

def evolution_step(fitness_fn, Ps, Qs, num_samples=10):
    num_agents = Ps.shape[0]
    fitness = fitness_fn(Ps, Qs)
    relative_fitness = fitness / fitness.sum()
    new_population = torch.multinomial(relative_fitness, num_agents, replacement=True)
    new_Ps = mutate(Ps[new_population], num_samples)
    new_Qs = mutate(Qs[new_population], num_samples)
    return new_Ps, new_Qs

def evolve(fitness_fn=basic_population_fitness, num_agents=100, num_meanings=5, num_signals=5, init_temperature=5, num_samples=10):
    Ps, Qs = random_langs(num_agents, num_meanings, num_signals, init_temperature)
    yield Ps, Qs
    for t in itertools.count(1):
        Ps, Qs = evolution_step(fitness_fn, Ps, Qs, num_samples)
        yield Ps, Qs

def main():
    evolution = evolve()
    try:
        for t, (Ps, Qs) in enumerate(evolution):
            print("step %d" % t, "population fitness: ", basic_population_fitness(Ps, Qs).mean().item())
    except KeyboardInterrupt:
        print(Ps.mean(dim=0))

if __name__ == '__main__':
    main()
        
    
    
    
    
    
    
    








    
