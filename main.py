

def process_demands(line, index):
    # demand_volue - h_d
    pass


def process_links(line, index):
    pass


def read_file(filepath):
    with open(filepath, 'r') as file:
        f_arr = file.read().splitlines()  # Removes newline characters compared to readlines()

    non_complex = {'module_capacity': 0, 'number_of_links': 0, 'number_of_demands': 0}
    for index, line in enumerate(f_arr):
        if any('#' + i == line.split('   ')[0] for i in non_complex.keys()):
            non_complex[line[1:]] = int(f_arr[index+1])


def main():
    read_file('OPT-1 net4.txt')


if __name__ == "__main__":
    main()
