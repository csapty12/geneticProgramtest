import sys
from random import randint, choice, random, sample
from math import log, sqrt
import numpy as np
import re
import copy


class GenExp(object):
    """

    """
    brackets_prob = 0.3
    operations = ['+', '-', '*']
    min_num = 0
    max_num = 20
    randomVariable1 = [randint(min_num, max_num), "X1"]
    inputs = [4, 8, 12, 13]
    output = [60, 64, 68, 69]

    def __init__(self, max_num, max_depth=None, depth=0):
        """

        :param max_num:
        :param max_depth:
        :param depth:
        """
        self.left = None  # create the left and right nodes for an expression.
        self.right = None
        self.grouped = None
        self.operators = None

        if max_depth is None:
            max_depth = log(max_num, 2) - 1

        if depth < max_depth and randint(0, max_depth) > depth:
            self.left = GenExp(max_num, max_depth, depth + 1)  # generate part of the expression (on the left)
        else:
            self.left = choice(GenExp.randomVariable1)

        if depth < max_depth and randint(0, max_depth) > depth:
            self.right = GenExp(max_num, max_depth, depth + 1)  # generate part of the expression (on the right)
        else:
            self.right = randint(GenExp.min_num, GenExp.max_num)

        self.grouped = random() < GenExp.brackets_prob  # if true, then bracket certain expressions will be allowed
        self.operators = choice(GenExp.operations)

    def __str__(self):
        """

        :return:
        """
        s = '{}{}{}'.format(self.left, self.operators, self.right)  # convert each value to string
        if self.grouped:
            return '({})'.format(s)
        else:
            return s

    @staticmethod
    def get_valid_expression(max_num, population_size):
        """
        function to get valid mathematical expressions
        :param max_num:
        :param population_size:
        :return:
        """
        expression_list = list()
        while len(expression_list) < population_size:
            expressions = GenExp(max_num)
            str_exps = str(expressions)
            expression_list.append(str_exps)
            # ensure that every expression contains a variable, e.g. X1-5(valid), rather than 6-5 (invalid)
            expression_list = [i for i in expression_list if "X1" in i]
        return expression_list

    @staticmethod
    def evaluate_expressions(expression):
        """
        function to evaluate the valid expression, where currently the value of X1 is a list of inputs.
        X1 is replaced with the current inputs to form a numerical expression (removing the X1 variable name)
        and returned
        """
        evaluated_list = list()
        x1 = GenExp.inputs
        for i in expression:
            # replace the string X1 with the value of the inputs.
            new_expression = [i.replace("X1", str(j)) for j in x1]
            evaluated_list.append(new_expression)
        return evaluated_list

    @staticmethod
    def get_totals(expression):
        """

        :param expression:
        :return:
        """
        totals = list()
        for i in expression:
            tmp = list()
            for j in i:
                sum1 = eval(j)
                tmp.append(sum1)
            totals.append(tmp)
        return totals

    # @staticmethod
    # def get_mean_squared_fitness(differences, outputs):
    #     print("differences::::")
    #     print(differences)
    #     mean_sq = list()
    #     for i in range(len(differences)):
    #         tmp = list()
    #         for j in range(len(differences[i])):
    #             x = differences[i][j] ** 2
    #             tmp.append(x)
    #
    #         mean_sq.append(tmp)
    #     mean = list()
    #     for i in mean_sq:
    #         x = np.mean(i)
    #         mean.append(x)
    #     return mean

    @staticmethod
    def get_mean_squared_fitness(totals, outputs):
        """
        first find the difference between actual output and expected output for each X1 value
        square the differences, sum them all up and divide by number len(x1)
        """
        # print("the outputs are::::: ")
        # print(outputs)
        # print("differences::::")
        print(totals)
        differences = list()
        for i in range(len(totals)):
            tmp = list()
            for j in range(len(totals[i])):
                x = totals[i][j] - outputs[j]
                tmp.append(x)
            differences.append(tmp)
        # print("differences: ")
        # print(differences)
        square = list()
        for i in range(len(differences)):
            tmp = list()
            for j in range(len(differences[i])):
                x = differences[i][j] ** 2
                tmp.append(x)
            square.append(tmp)
        # print("differences squared: ")
        # print(square)
        new_total = list()
        for i in range(len(square)):
            x = sum(square[i])
            new_total.append(x)
        # print("new totals: ")
        # print(new_total)

        root_mean_sq_err = list()
        for i in new_total:
            x = i / len(GenExp.inputs)
            err = sqrt(x)
            root_mean_sq_err.append(err)
        # print("root mean sq err")
        # print(root_mean_sq_err)
        return root_mean_sq_err

    @staticmethod  # need to change as need to select parents using fitness evaluation, not randomly
    def select_parents(population, fitness, num_parents):
        parents = sample(population, num_parents)
        return parents

    @staticmethod
    def split_parents(parents):
        split_list = [re.findall('\w+|\W', s) for s in parents]
        [i.append('end') for i in split_list]
        return split_list

    @staticmethod
    def gen_population(max_num, population_size):
        population = GenExp.get_valid_expression(max_num, population_size)  # max_num, population size
        print("population: ")  # this should be created only once. need to fix this
        print(population)
        return population


class Node(object):
    node_id = 0

    def __init__(self, value):
        Node.node_id += 1
        self.node_num = Node.node_id
        self.value = value
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.checked = False
        self.checkedAgain = False

    def add_child(self, value, left=True):
        if left:  # add child to left hand side of tree
            new_node = Node(value)
            self.left_child = new_node
            new_node.parent = self

        elif not left:  # is right child
            new_node = Node(value)
            self.right_child = new_node
            new_node.parent = self

    def __str__(self, level=0):
        ret = "\t" * level + self.__repr__() + "\n"
        if self.left_child is not None:
            ret += self.left_child.__str__(level + 1)
        if self.right_child is not None:
            ret += self.right_child.__str__(level + 1)
        return ret

    def __repr__(self):
        if self.parent is not None:
            return "Node (" + str(self.value) + ") parent val:" + str(self.parent.value)
        else:
            return "Node (" + str(self.value) + ")"


class ToPrefixParser(object):
    # every instance of a tree is a node.
    def __init__(self, val=None, left=None, right=None):
        self.val = val  # holds the value
        self.left = left  # holds the left child value
        self.right = right  # holds the right child value

    def __str__(self):
        return str(self.val)  # print out value of node

    @staticmethod
    def get_operation(token_list, expected):
        """
        compares the expected token to the first token on the list. if they match, remove it, return True
        this is to get the operator
        """
        if token_list[0] == expected:
            del token_list[0]
            return True
        else:
            return False

    def get_number(self, token_list):
        if self.get_operation(token_list, '('):
            x = self.get_expression(token_list)  # get the subexpression
            self.get_operation(token_list, ')')  # remove the closing parenthesis
            return x
        else:
            x = token_list[0]
            if not isinstance(x, str):
                return None
            token_list[0:1] = list()
            return ToPrefixParser(x, None, None)

    def get_product(self, token_list):
        a = self.get_number(token_list)
        if self.get_operation(token_list, '*'):
            b = self.get_product(token_list)
            return ToPrefixParser('*', a, b)
        elif self.get_operation(token_list, '/'):
            b = self.get_product(token_list)
            return ToPrefixParser('/', a, b)
        else:
            return a

    def get_expression(self, token_list):
        a = self.get_product(token_list)
        if self.get_operation(token_list, '-'):
            b = self.get_expression(token_list)
            return ToPrefixParser('-', a, b, )
        elif self.get_operation(token_list, '+'):
            b = self.get_expression(token_list)
            return ToPrefixParser('+', a, b)
        else:
            return a

    def print_tree_prefix(self, tree):
        if tree.left is None and tree.right is None:

            return tree.val
        else:

            left = self.print_tree_prefix(tree.left)
            right = self.print_tree_prefix(tree.right)
            return tree.val + " " + left + ' ' + right + ''

    def get_prefix_notation(self, token_list):
        prefix = list()
        prefix_list = list()
        for i in token_list:
            tree = self.get_expression(i)
            y = self.print_tree_prefix(tree)
            prefix.append(y)
        for i in prefix:
            prefix_list.append(i.split())
        return prefix_list


def main(current_population=None, iteration=0):

    current_population = GenExp.gen_population(8, 4)

    evaluate_expressions = GenExp.evaluate_expressions(
        current_population)  # replaces X1 with the actual values of the inputs
    print("evaluating expressions: ")
    print(evaluate_expressions)
    totals = GenExp.get_totals(evaluate_expressions)  # gets the totals of each of the list of lists
    print("totals: ")
    print(totals)
    fitness = GenExp.get_mean_squared_fitness(totals, GenExp.output)
    print("fitness values: ")
    print(fitness)

    # print()
    # print("=======================================================")
    # print("parents")
    # select_parents = GenExp.select_parents(current_population, fitness, 2)
    # print("parents selected: ", select_parents)
    # split_parents = GenExp.split_parents(select_parents)
    # print("split parents: ", split_parents)
    # pref = ToPrefixParser()
    # get_prefix_parents = pref.get_prefix_notation(split_parents)
    # print("getting prefix notation: ")
    # print(get_prefix_parents)


if __name__ == "__main__":
    main()
