from trainGP import train_gp
from data import Data
import numpy


def test_gp_full_data(test_dataset):
    d = Data(test_dataset)
    x = d.read_data()
    return x[0], x[1]


def run_gp(test_data_set):
    import math

    optimal_expression = train_gp(data_set="dataset2.txt", gen_depth=3, max_depth=3,
                                  population_size=4, max_iteration=2, selection_type="select_best",
                                  tournament_size=50, cross_over_rate=0.5, mutation_rate=0.99)

    x = test_gp_full_data(test_data_set)
    row = x[0]
    label = x[1]

    exp = list()
    exp.append(optimal_expression)
    optimal_expression = exp
    # row = [[0.185841328, 0.229878245, 0.150353322, 2.267962444, 1.72085425],
    #        [0.16285377, 0.293619897, 0.148429586, 2.112106101, 1.726711829],
    #        [0.149332758, 0.347589881, 0.139985797, 1.689751437, 1.734865801],
    #        [0.137193647, 0.416721256, 0.147865432, 2.116532577, 1.761369401],
    #        [0.082350665, 0.480389313, 0.174387346, 2.342011704, 1.766493641],
    #        [0.159720391, -0.781208802, -0.087774755, 0.333050959,1.899437307]]
    # label = [0, 0, 0, 0, 0, 1]
    #
    prediction = list()
    for i in optimal_expression:
        tmp = list()
        for j in row:
            new_exp = i.replace("X1", str(j[0])).replace("X2", str(j[1])).replace("X3", str(j[2])) \
                .replace("X4", str(j[3])).replace("X5", str(j[4]))
            eva = eval(new_exp)
            if eva >= 0:
                x = eva
                tmp.append(x)
            else:
                y = eva
                tmp.append(y)
        prediction.append(tmp)

    prob = list()

    for i in prediction:
        for j in i:
            try:
                sig = 1 / (1 + math.exp(-j))
            except OverflowError:
                sig = 0
            if sig > 0.5:
                prob.append(1)
            else:
                prob.append(0)
    print("expression: ", optimal_expression)
    # print("classifications")
    # print(prob)

    trufa = prob == label
    # for i in range(len(prob)):
    #     trufa.append(i == label[i])

    # print("true false array")
    # print(trufa)

    print("accuracy: ", sum(trufa) / len(trufa))


if __name__ == "__main__":
    run_gp('dataset2.txt')

# TODO - IMPLEMENT LEVEL CAP - OR AT LEAST HANDLE IT SOMEHOW
