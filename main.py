import copy
import random
import math
from read_file import read_file
from time import perf_counter

def random_sum(n, total):
    rand_floats = [random.random() for _ in range(n)]
    rand_nums = [math.floor(i * total / sum(rand_floats)) for i in rand_floats]
    for _ in range(total - sum(rand_nums)):
        rand_nums[random.randint(0, n - 1)] += 1
    return rand_nums


def random_chromosome(demands):
    flows = {str(i): [] for i in range(1, len(demands) + 1)}
    for index, demand in enumerate(demands):
        h_d = demand.get('h_d')
        n = len(demand.get('paths'))
        flows[str(index + 1)] = random_sum(n, h_d)

    return flows


def calculate_links_loads(flows, demands, links):
    loads = {str(i): None for i in range(1, len(links) + 1)}
    flows_arr = []
    demands_arr = []
    for i in flows.values():  # normal cast to list not possible, because we want a single list not a list of lists
        flows_arr.extend(i)
    for i in demands:
        demands_arr.extend(i.get('paths'))

    for i in range(1, len(links) + 1):
        indexes = [demands_arr.index(j) for j in demands_arr if i in j]
        load = sum(flows_arr[k] for k in indexes)
        loads[str(i)] = load

    return loads


def calculate_links_overloads(loads, links, module_cap):
    overloads = {str(i): None for i in range(1, len(links) + 1)}
    capacities = [module_cap * int(link.get('module_count')) for link in links]

    for index, load, capacity in zip(overloads.keys(), loads.values(), capacities):
        overloads[index] = int(load) - capacity

    return overloads


def calculate_links_size(loads, links, module_cap):
    links_size = {str(i): None for i in range(1, len(links) + 1)}

    for index, load in zip(links_size.keys(), loads.values()):
        links_size[index] = math.ceil(load / module_cap)

    return links_size


def crossover(parent1, parent2, crossover_probability):
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)

    for i in offspring1:
        if random.random() <= crossover_probability:
            offspring1[i] = parent2[i]
            offspring2[i] = parent1[i]

    return offspring1, offspring2


def mutate(chromosomes, mutation_occur_probability, mutation_probability):
    for chromosome in chromosomes:
        if random.random() <= mutation_occur_probability:
            gene_to_mutate = random.randint(1, len(chromosome))
            for gene in chromosome:
                if random.random() <= mutation_probability or gene_to_mutate == int(gene):
                    decreased_index = random.randint(0, len(chromosome[gene]) - 1)
                    while chromosome[gene][decreased_index] == 0:  # We can't have negative values
                        decreased_index = random.randint(0, len(chromosome[gene]) - 1)

                    increased_index = random.randint(0, len(chromosome[gene]) - 1)
                    while increased_index == decreased_index:  # To shift one unit from one path to another indexes must be different
                        increased_index = random.randint(0, len(chromosome[gene]) - 1)

                    chromosome[gene][decreased_index] -= 1
                    chromosome[gene][increased_index] += 1


def generate_start_population(population_size, demands):
    start_population = []
    for _ in range(population_size):
        start_population.append(random_chromosome(demands))

    return start_population


def find_objective_functions_dap(chromosomes, demands, links, module_capacity):
    funs = []
    for chromosome in chromosomes:
        loads = calculate_links_loads(chromosome, demands, links)
        overloads = calculate_links_overloads(loads, links, module_capacity)
        objective_function = max(overloads.values())
        funs.append(objective_function)

    return funs


def find_best_objective_function_dap(chromosomes, demands, links, module_capacity):
    best_objective_function = float('inf')
    best_chromosome = {}

    for chromosome in chromosomes:
        loads = calculate_links_loads(chromosome, demands, links)
        overloads = calculate_links_overloads(loads, links, module_capacity)
        objective_function = max(overloads.values())

        if objective_function < best_objective_function:
            best_objective_function = objective_function
            best_chromosome = chromosome

    return best_objective_function, copy.deepcopy(best_chromosome)


def find_worst_objective_function_dap(chromosomes, demands, links, module_capacity):
    worst_objective_function = -float('inf')
    worst_chromosome = {}

    for chromosome in chromosomes:
        loads = calculate_links_loads(chromosome, demands, links)
        overloads = calculate_links_overloads(loads, links, module_capacity)
        objective_function = max(overloads.values())

        if objective_function > worst_objective_function:
            worst_objective_function = objective_function
            worst_chromosome = chromosome

    return worst_chromosome


def find_best_objective_function_ddap(chromosomes, demands, links, module_capacity):
    best_objective_function = float('inf')
    best_chromosome = {}

    for chromosome in chromosomes:
        loads = calculate_links_loads(chromosome, demands, links)
        links_size = calculate_links_size(loads, links, module_capacity)
        objective_function = 0
        for i in range(len(links)):
            objective_function += links_size[str(i + 1)] * int(links[i].get('module_cost'))

        if objective_function < best_objective_function:
            best_objective_function = objective_function
            best_chromosome = chromosome

    return best_objective_function, copy.deepcopy(best_chromosome)


def find_worst_objective_function_ddap(chromosomes, demands, links, module_capacity):
    worst_objective_function = -float('inf')
    worst_chromosome = {}

    for chromosome in chromosomes:
        loads = calculate_links_loads(chromosome, demands, links)
        links_size = calculate_links_size(loads, links, module_capacity)
        objective_function = 0
        for i in range(len(links)):
            objective_function += links_size[str(i + 1)] * int(links[i].get('module_cost'))

        if objective_function > worst_objective_function:
            worst_objective_function = objective_function
            worst_chromosome = chromosome

    return worst_chromosome


def find_objective_functions_ddap(chromosomes, demands, links, module_capacity):
    funs = []
    for chromosome in chromosomes:
        loads = calculate_links_loads(chromosome, demands, links)
        links_size = calculate_links_size(loads, links, module_capacity)
        objective_function = 0
        for i in range(len(links)):
            objective_function += links_size[str(i + 1)] * int(links[i].get('module_cost'))
        funs.append(objective_function)
    return funs


def calculate_dap(demands, links, module_capacity, population_size, simulation_limit,
                  crossover_occur_probability, crossover_probability,
                  mutation_occur_probability, mutation_probability):
    iter_without_improvement = 0
    current_generation_number = 0

    parents_generation = generate_start_population(population_size, demands)
    while True:
        current_generation_number += 1
        objective_funs = find_objective_functions_dap(parents_generation, demands, links, module_capacity)
        current_best_objective_function, worst_fun_val = min(objective_funs), max(objective_funs)
        current_best_chromosome = copy.deepcopy(parents_generation[objective_funs.index(min(objective_funs))])

        # Worst functions need to have lowest weight, highest the highest. hence the reverse of values
        weights = [worst_fun_val - i for i in objective_funs]
        children = []
        for _ in range(0, population_size, 2):
            if random.random() <= crossover_occur_probability:
                parent1, parent2 = random.choices(parents_generation, weights, k=2)
                offspring1, offspring2 = crossover(parent1, parent2, crossover_probability)
                children.append(offspring1)
                children.append(offspring2)

        parents_generation += children
        # Parents have changed - recalculate objective functions
        objective_funs += find_objective_functions_dap(children, demands, links, module_capacity)
        # we remove the worst chromosomes because we have "population_size + crossover" chromosomes in next generation
        for _ in range(len(parents_generation) - population_size + 1):
            worst_fun_val = max(objective_funs)
            i = objective_funs.index(worst_fun_val)
            del objective_funs[i]  # Removes by index rather than by value
            del parents_generation[i]  # Remove corresponding bad chromosome, based on the worst fun value

        # mutation criteria
        mutate(parents_generation, mutation_occur_probability, mutation_probability)

        # add the best chromosome to next generation
        parents_generation.append(current_best_chromosome)
        objective_funs.append(current_best_objective_function)
        # find new best chromosome after crossovers
        next_generation_best_objective_function = min(objective_funs)
        next_generation_best_chromosome = parents_generation[objective_funs.index(min(objective_funs))]

        # Check if new generation is better than the previous
        if next_generation_best_objective_function >= current_best_objective_function:
            iter_without_improvement += 1
        else:
            iter_without_improvement = 0

        if iter_without_improvement >= simulation_limit:
            # return find_best_objective_function_dap(best_solutions, demands, links, module_capacity)
            return next_generation_best_chromosome, next_generation_best_objective_function


def calculate_ddap(demands, links, module_capacity, population_size, simulation_limit,
                   crossover_occur_probability, crossover_probability,
                   mutation_occur_probability, mutation_probability):
    iter_without_improvement = 0
    current_generation_number = 0

    parents_generation = generate_start_population(population_size, demands)
    while True:
        current_generation_number += 1
        objective_funs = find_objective_functions_ddap(parents_generation, demands, links, module_capacity)
        current_best_objective_function, worst_fun_val = min(objective_funs), max(objective_funs)
        current_best_chromosome = copy.deepcopy(parents_generation[objective_funs.index(min(objective_funs))])

        weights = [worst_fun_val - i for i in objective_funs]
        children = []
        for _ in range(0, population_size, 2):
            if random.random() <= crossover_occur_probability:
                parent1, parent2 = random.choices(parents_generation, weights, k=2)
                offspring1, offspring2 = crossover(parent1, parent2, crossover_probability)
                children.append(offspring1)
                children.append(offspring2)
        parents_generation += children

        # Parents have changed - recalculate objective functions
        objective_funs += find_objective_functions_ddap(children, demands, links, module_capacity)

        # we remove the worst chromosomes because we have "population_size + crossover" chromosomes in next generation
        for _ in range(len(parents_generation) - population_size + 1):
            worst_fun_val = max(objective_funs)
            i = objective_funs.index(worst_fun_val)
            del objective_funs[i]  # Removes by index rather than by value
            del parents_generation[i]  # Remove corresponding bad chromosome, based on the worst fun value

        # mutation criteria
        mutate(parents_generation, mutation_occur_probability, mutation_probability)

        # add the best chromosome to next generation
        parents_generation.append(current_best_chromosome)
        objective_funs.append(current_best_objective_function)

        # find new best chromosome after crossovers
        next_generation_best_objective_function = min(objective_funs)
        next_generation_best_chromosome = parents_generation[objective_funs.index(min(objective_funs))]

        # Check if new generation is better than the previous
        if next_generation_best_objective_function >= current_best_objective_function:
            iter_without_improvement += 1
        else:
            iter_without_improvement = 0

        if iter_without_improvement >= simulation_limit:
            # return find_best_objective_function_ddap(best_solutions, demands, links, module_capacity)
            return next_generation_best_objective_function, next_generation_best_chromosome


def main_loop(non_complex, demands, links):
    population_size = 1000
    simulation_limit = 100
    module_capacity = non_complex.get('module_capacity')
    crossover_occur_probability = 0.2
    crossover_probability = 0.5
    mutation_occur_probability = 0.25  # mutation on chromosome
    mutation_probability = 0.05  # mutation on more than one gene

    t1 = perf_counter()
    min_f, best_solution = calculate_dap(demands, links, module_capacity, population_size, simulation_limit,
                                         crossover_occur_probability, crossover_probability,
                                         mutation_occur_probability, mutation_probability)
    t2 = perf_counter()
    t_delta = t2 - t1
    print("Simulation finished - DAP:")
    print("\t Best solution: ", best_solution)
    print("\t Objective function: ", min_f)
    print(f"\t Time elapsed: {t_delta:.4f} s")

    t1 = perf_counter()
    min_f, best_solution = calculate_ddap(demands, links, module_capacity, population_size, simulation_limit,
                                          crossover_occur_probability, crossover_probability,
                                          mutation_occur_probability, mutation_probability)
    t2 = perf_counter()
    t_delta = t2 - t1
    print("Simulation finished - DDAP:")
    print("\t Best solution:", best_solution)
    print("\t Objective function:", min_f)
    print(f"\t Time elapsed: {t_delta:.4f} s")


def main():
    seed = 2077
    random.seed(seed)

    non_complex, demands, links = read_file('OPT-1 net4.txt')
    print("Finished reading file:")
    print('\t', demands)

    main_loop(non_complex, demands, links)


if __name__ == "__main__":
    main()
