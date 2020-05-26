import os


def get_weighted_consensus(net_paths):
    for path in net_paths:
        print(path)


def get_majority_consensus(net_paths):
    if len(net_paths) % 2 != 1:
        print("equal")
        return
    for path in net_paths:
        print(path)


if __name__ == "__main__":
    get_majority_consensus([1, 2, 3, 4, 5, 6, 7])
