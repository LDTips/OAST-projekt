

def process_demands(f_arr, index):
    # demand_volue - h_d
    demands = []
    f_arr = f_arr[index+1:]  # f_arr[index] is the definition of the section
    arr_iter = iter(f_arr)  # Easier to do with iterator rather than classic for loop

    while True:
        try:
            current = next(arr_iter).split(' ')
            if '#' in current or current == '':  # Check if another section is not present
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
                "link": link,
                "paths": paths,
                "h_d": h_d
            }
        )

    return demands


def process_links(f_arr, index):
    return "Processed links..."


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


def main():
    non_complex, demands, links = read_file('OPT-1 net4.txt')
    print("Finished reading")


if __name__ == "__main__":
    main()
