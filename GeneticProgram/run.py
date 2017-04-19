from trainGP import train_gp


def run_gp():
    import math

    optimal_expression = train_gp(data_set="dataset2.txt", max_depth=3, population_size=500, max_iteration=500,
                                  cross_over_rate=0.9,
                                  mutation_rate=0.9)

    print("expression: ", optimal_expression)

    exp = list()
    exp.append(optimal_expression)
    # print("expression: ", exp)
    optimal_expression = exp

    row = [[0.185841328, 0.229878245, 0.150353322, 2.267962444, 1.72085425],
           [0.16285377, 0.293619897, 0.148429586, 2.112106101, 1.726711829],
           [0.149332758, 0.347589881, 0.139985797, 1.689751437, 1.734865801],
           [0.137193647, 0.416721256, 0.147865432, 2.116532577, 1.761369401],
           [0.082350665, 0.480389313, 0.174387346, 2.342011704, 1.766493641],
           [0.159720391, -0.781208802, -0.087774755, 0.333050959,1.899437307]]
    label = [0, 0, 0, 0, 0, 1]

    prediction = list()
    for i in optimal_expression:
        tmp = list()
        for j in row:
            # print(j)
            new_exp = i.replace("X1", str(j[0])).replace("X2", str(j[1])).replace("X3", str(j[2])) \
                .replace("X4", str(j[3])).replace("X5", str(j[4]))
            eva = eval(new_exp)
            print()
            print("eval: ", eva)
            if eva >= 0:
                print("evaluation : ", eva)
                print("Company likely to go bankrupt")
                x = eva
                print("val of x is = ", x)
                tmp.append(x)
                # tmp.append(1)
            else:
                print("evaluation here : ", eva)
                print("Company not likely to go bankrupt")
                y = eva
                print("val of y is = ", y)
                tmp.append(y)
        prediction.append(tmp)

    print("predictions")
    print(prediction)

    prob = list()

    for i in prediction:
        for j in i:
            sig = 1 / (1 + math.exp(-j))
            #
            # print("sig")
            # print(sig)
            print()
            if sig > 0.5:
                prob.append(1)
            else:
                prob.append(0)
    print("classifciations")
    print(prob)


if __name__ == "__main__":
    run_gp()

# TODO - PARAMETERISE THE TRAIN_GP FUNCTION
# TODO - IMPLEMENT LEVEL CAP
# TODO - TESTING
# TODO - USE SIGMOID FUNCTION TO MAKE CLASSIFICAITON AND PROBABILITY.
# TODO - SIMPLE GUI INTERFACE
