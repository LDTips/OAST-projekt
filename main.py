import random
import math


def process_demands(f_arr, index):
    # demand_volue - h_d
    demands = []
    f_arr = f_arr[index+1:]  # f_arr[index] is the definition of the section
    arr_iter = iter(f_arr)  # Easier to do with iterator rather than classic for loop

    while True:
        try:
            current = next(arr_iter).split(' ')
            if '#' in current[0] or current[0] == '':  # Check if another section is not present
                break
        except StopIteration:  # Raised if iterator is exhausted
            break

        # Reads info from first line which is in format: demand_id   link_nodeA   link_nodeB   demand_volume
        link = (current[1], current[2])
        h_d = current[3]

        path_no = int(next(arr_iter))  # Amount of paths is defined after the first line

        paths = []
        for _ in range(path_no):
            current_path = next(arr_iter).strip().split(' ')
            nodes = {int(i) for i in current_path[1:]}  # Read vertices into a set. First digit is demand_id, we skip it
            paths.append(nodes)

        next(arr_iter)  # Skip the empty line that separates each link & demands

        demands.append(
            {
                "demand": link,
                "paths": paths,
                "h_d": int(h_d)
            }
        )

    return demands


def process_links(f_arr, index):
    links = []
    f_arr = f_arr[index+1:]  # f_arr[index] is the definition of the section
    arr_iter = iter(f_arr)  # Easier to do with iterator rather than classic for loop

    while True:
        current = next(arr_iter).split(' ')
        if current[0] == '':
            break
        links.append(
            {
                'link': (current[1], current[2]),
                'module_count': current[3],
                'module_cost': current[4]
            }
        )

    return links


def read_file(filepath):
    with open(filepath, 'r') as file:
        f_arr = file.read().splitlines()  # Removes newline characters compared to readlines()

    non_complex = {'module_capacity': 0, 'number_of_links': 0, 'number_of_demands': 0}  # These do not require parsing
    demands, links = [], []
    for index, line in enumerate(f_arr):
        section = line.split('   ')[0]
        if any('#' + i == section for i in non_complex.keys()):
            # Variable names in file are the same as in the non_complex dict. slice [1:] is used for strip the #
            non_complex[line[1:]] = int(f_arr[index+1])  # index+1 fetches the value from the next line
        elif section == "#number_of_links_in_the path":
            demands = process_demands(f_arr, index)
        elif section == "#link_id":
            links = process_links(f_arr, index)

    return non_complex, demands, links


def random_sum(n, total):
    rand_floats = [random.random() for _ in range(n)]
    rand_nums = [math.floor(i * total / sum(rand_floats)) for i in rand_floats]
    for _ in range(total - sum(rand_nums)):
        rand_nums[random.randint(0, n-1)] += 1
    return rand_nums


def random_flow_allocation(demands):
    flows = {str(i): [] for i in range(1, len(demands) + 1)}
    for index, demand in enumerate(demands):
        h_d = demand.get('h_d')
        n = len(demand.get('paths'))
        flows[str(index + 1)] = random_sum(n, h_d)

    return flows


def calculate_link_loads(flows, demands, links):
    loads = {str(i): None for i in range(1, len(links) + 1)}
    flows_arr = []
    demands_arr = []
    for i in flows.values():
        flows_arr.extend(i)
    for i in demands:
        demands_arr.extend(i.get('paths'))

    for i in range(1, len(links)+1):
        indexes = [demands_arr.index(j) for j in demands_arr if i in j]
        load = sum(flows_arr[k] for k in indexes)
        loads[str(i)] = load

    return loads


def calculate_link_overloads(loads, links, module_cap):
    overloads = {str(i): None for i in range(1, len(links) + 1)}
    capacities = [module_cap * int(link.get('module_count')) for link in links]

    for index, (load, capacity) in enumerate(zip(loads.values(), capacities)):
        overloads[str(index + 1)] = int(load) - capacity

    return overloads

def crossover(parent1_flows, parent2_flows):
    crossover_probability = 0.5
    offspring1_flows = {str(i): [] for i in range(1, len(parent1_flows) + 1)}
    offspring2_flows = {str(i): [] for i in range(1, len(parent1_flows) + 1)}
    for i in offspring1_flows:
        if random.random() <= crossover_probability:
            offspring1_flows[i] = parent2_flows[i]
            offspring2_flows[i] = parent1_flows[i]
        else:
            offspring1_flows[i] = parent1_flows[i]
            offspring2_flows[i] = parent2_flows[i]
    return offspring1_flows, offspring2_flows

def main_loop(non_complex, demands, links):
    limit = 50
    min_f = float('inf')
    iter_without_improvement = 0
    while True:
        flows = random_flow_allocation(demands)
        loads = calculate_link_loads(flows, demands, links)
        overloads = calculate_link_overloads(loads, links, non_complex.get('module_capacity'))

        new_min = max(overloads.values())
        if new_min < min_f:
            min_f = new_min
            best_overload = overloads
            iter_without_improvement = 0

        iter_without_improvement += 1
        if iter_without_improvement >= limit:
            return best_overload, min_f


def main():
    seed = 10
    random.seed(seed)

    non_complex, demands, links = read_file('OPT-1 net4.txt')
    print("Finished reading file")

    ol, min_f = main_loop(non_complex, demands, links)
    print(ol)
    print(min_f)



if __name__ == "__main__":
    main()
