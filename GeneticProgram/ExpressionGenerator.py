from random import choice, random, sample
from data import Data
import numpy as np


class GenMember(object):
    """
    Class that is used to create valid mathematical expressions, get the fitness of the each of the individuals in the
    population, select two parents, and also to update the population once the children are ready to be added into the
    new population.

    """
    # Read the data from the text file
    d = Data('dataset2.txt')
    read_data = d.read_data(shuffle_d=False)
    data = read_data[0]
    labels = read_data[1]

    # the set of functional values. - consider expanding this.
    operations = ['+', '-', '*', '/']

    def generate_expression(self, max_depth=4):
        """
        Function to generate a valid mathematical expression. An expression consists of values from the functional
        set -> ['+', '-', '*', '/'] and values from a terminal set -> [random number between 0-50, X1,...,X5] where
        X1,..., are Altman's KPI ratios.
        :param max_depth: maximum depth of the regression tree.
        :return: valid expression <= maximum depth of tree.
        """

        # print out either a random number between 0 and 50, or a variable X1-X5.
        if max_depth == 1:
            terminals = [random() * 50, "X1", "X2", 'X3', "X4", "X5"]  # random() * 50,
            return self.__str__(choice(terminals))

        # include bracketing 20% of the time.
        rand = random()
        if rand <= 0.2:
            return '(' + self.generate_expression(max_depth - 1) + choice(self.operations) + self.generate_expression(
                max_depth - 1) + ')'
        else:
            return self.generate_expression(max_depth - 1) + choice(self.operations) + self.generate_expression(
                max_depth - 1)

    def __str__(self, num):
        """
        cast terminal value to a string.
        :param num: the value to be parsed as a string.
        :return: value parsed as a string
        """
        return str(num)

    def get_valid_expressions(self, max_depth, population_size):
        """
        function to ensure that each initial member of the population contains at least the variables X1,..,X5.
        :param max_depth: maximum depth of the tree.
        :param population_size: generate a user defined population size.
        :return: every individual in population as a list of strings.
        """
        expression_list = list()
        while len(expression_list) < population_size:
            # generate the expressions and cast them to strings.
            init = GenMember()
            exps = init.generate_expression(max_depth)
            str_exps = str(exps)
            expression_list.append(str_exps)
            # print out valid expressions which contain all the variables.
            expression_list = [item for item in expression_list if 'X1' and 'X2' and 'X3' and 'X4' and 'X5' in item]
        return expression_list

    def get_fitness(self, expressions, child=False):
        """
        Function to get the fitness of the population. Fitness function based on Number of Hits method.
        :param expressions: list of expressions being passed in. If not first iteration, then expression comes in
        as a single expression string and is converted to a list containing the child expression to be evaluated.
        :param child: if child is false, then assume first iteration -> get fitness of whole population. If child is
        true, then only get fitness of new children values, not total population.
        :return:
        """
        if child is True:
            exp_list = list()
            exp_list.append(expressions)
            expression = exp_list

        else:
            expression = expressions
        # get all the rows of the data being passed in to get the fitness.
        row = np.asarray(GenMember.data, dtype=object)

        # transpose the data to get all the X1 values in a list and repeat for X2,...,X5
        new_row = row.T
        # get the labels of the company data.
        labels = GenMember.labels

        # store the data in the variables to make evaluation of expression easier.
        X1 = new_row[0]  # length = len of data set
        X2 = new_row[1]
        X3 = new_row[2]
        X4 = new_row[3]
        X5 = new_row[4]
        predictions = list()

        for ex in expression:
            tmp = list()
            try:
                # evaluate the expression
                x = eval(ex)
                # if evaluation does not contain any variables from the terminal set
                if isinstance(x, float) or isinstance(x, int):
                    for l in range(len(X1)):
                        tmp = [x] * len(X1)
                    predictions.append(tmp)
                else:
                    # if the total is greater than 0 i.e. positive, append 0, else 1
                    for j in x:
                        if j >= 0:
                            tmp.append(1)
                        else:
                            tmp.append(0)
                    predictions.append(tmp)
            # if expression contains "/0" throw ZeroDivisionError and give individual a poor fitness.
            except ZeroDivisionError:
                # print("cannot divide by 0!!!")
                for k in range(len(X1)):
                    tmp = [9999] * len(X1)
                predictions.append(tmp)

        # get number of hits fitness.
        noh = list()
        for k in range(len(predictions)):
            tmp = list()
            [tmp.append(labels[j] == predictions[k][j]) for j in range(len(predictions[k]))]
            noh.append(tmp)
        fitness = [len(j) - sum(j) for j in noh]
        return fitness

    def tournament_selection(self, population, fitness, selection_size):
        """
        Function to select the parents of the population using tournament selection. Select n individuals from the
        population at random, and select the best two individuals from the selection to be the parents.
        :param population: the population generated - the list of expressions
        :param fitness: the population fitnesses
        :param selection_size: the number of individuals to compete against each other
        :return: two parents that will be used to create offspring - type: list(strings)
        """
        zipped_population = list(zip(population, fitness))
        # print("zipped population: ", zipped_population)

        # select potential candidate solutions to be assessed.
        candidates = sample(zipped_population, selection_size)
        # print("candidates:",candidates)

        # select the first parent with the best fitness out of the candidates
        parent_one = min(candidates, key=lambda t: t[1])
        # print(parent_one)
        p1_index = zipped_population.index(parent_one)
        # print(p1_index)
        # remove parent for now to prevent parent being selected twice
        zipped_population.pop(p1_index)
        # print("new popilation:", zipped_population)

        candidates = sample(zipped_population, selection_size)
        # select another sample and get the second parent
        parent_two = min(candidates, key=lambda t: t[1])
        p2_index = zipped_population.index(parent_two)
        zipped_population.pop(p2_index)

        # return the parents as a list of strings.
        parents = list()
        parents.append(parent_one)
        parents.append(parent_two)
        return parents

    def update_population(self, population, fitness, c1, child_fit1, c2, child_fit2):
        """
        Function to update the population, by comparing the two worst individuals in the current population,
        with the two new children produced. Insert the children into the population if they have a better fitness
        relative to the two worst in the population to improve the population fitness.
        :param population: the current population
        :param fitness: fitness of each individual in the current population
        :param c1: first child produced
        :param child_fit1: first child produced fitness
        :param c2: second child produced
        :param child_fit2: second child produced fitness
        :return: the new updated population with the new population fitnesses.
        """
        # print("current population")
        # print(population)
        # print("fitenss: ")
        # print(fitness)
        child1 = list()
        child2 = list()

        child1.append(c1)
        child2.append(c2)

        zipped_population = list(zip(population, fitness))
        # print("zipped popn",zipped_population)
        child2 = list(zip(child2, child_fit2))
        # print("child2: ", child2)

        # # print("worst candidate 1: ")
        worst_one = max(zipped_population, key=lambda t: t[1])
        w1_index = zipped_population.index(worst_one)
        # print("worst one: ", worst_one)
        # if the child fitness is better than the worst in the population, replace them with first child
        if child_fit1[0] <= worst_one[1]:
            zipped_population.pop(w1_index)
            zipped_population.append((c1, child_fit1[0]))

        # if the child fitness is better than the worst in the population, replace them with first child
        worst_two = max(zipped_population, key=lambda t: t[1])
        w2_index = zipped_population.index(worst_two)
        # print("worst2: ", worst_two)

        if child_fit2[0] <= worst_two[1]:
            zipped_population.pop(w2_index)
            zipped_population.append((c2, child_fit2[0]))

        # print("zipped population: ", zipped_population)
        new_population = [i[0] for i in zipped_population]
        new_population_fitness = [i[1] for i in zipped_population]

        return new_population, new_population_fitness

