import glob


def main():

    counts = {}
    with open('communities-2009-12-1.txt', 'r') as file:
        # for line in file.readlines():
        #     numbers = line.split()
        #     for num in numbers:
        #         counts[num] = counts.get(num, 0)+1
        lst = []
        for line in file.readlines():
            line.split()
            line = line.replace('\n', '')
            value = line.rstrip()
            print(len(value))
            lst.append(line)
        print(len(lst))



    # #print(counts)
    # for k,v in counts.items():
    #     if v !=1 :
    #         print('Error in communies')


    #print(len(counts.keys()))


if __name__ == '__main__':
    main()
