import matplotlib.pyplot as plt
import csv
import os

def plot_distribution(csv_file, label):

    x = []
    y = []

    with open(csv_file, "r") as f:

        reader = csv.DictReader(f)

        for row in reader:
            x.append(int(row["path_length"]))
            y.append(float(row["probability"]))

    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Path Length")
    plt.ylabel("Probability")
    plt.title(label)

    filename = f"plots/{label}.png"

    plt.savefig(filename)
    plt.close()

    print(f"Saved plot: {filename}")

def plot_multiple_distributions(file_label_pairs):

    plt.figure()

    for csv_file, label in file_label_pairs:

        x = []
        y = []

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                x.append(int(row["path_length"]))
                y.append(float(row["probability"]))

        plt.plot(x, y, label=label)

    plt.xlabel("Path Length")
    plt.ylabel("Probability")
    plt.legend()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/comparison2^12.png")
    plt.close()

    print("Saved comparison plot.")