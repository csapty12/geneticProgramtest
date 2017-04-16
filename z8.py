from random import choice, random, sample

import re
import copy
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    """
    Function to load in the text file. Function splits the data into two sets.
    set 1: company data
    set 2: company data labels - either a 0 or 1.
    :return: tuple - (company data, company class)
    """
    from numpy import loadtxt
    cbd = loadtxt('dataset2.txt')  # read in the data
    class_labels_cfd = cbd[:, -1]  # get the classification categories.
    class_labels_cfd = [int(x) for x in class_labels_cfd]
    class_labels_cfd = np.asarray(class_labels_cfd, dtype=int)

    data_cbd = cbd[:, 0:-1]
    # print(data_CBD)
    return data_cbd, class_labels_cfd


class GenMember(object):
    """
    Class that is used to create valid mathematical expressions, get the fitness of the each of the individuals in the
    population, select two parents, and also to update the population once the children are ready to be added into the
    new population.

    """

    # Read the data from the text file
    read_data = read_data()
    data = read_data[0]
    labels = read_data[1]

    # the set of functional values. - consider expanding this.
    operations = ['+', '-', '*', '/']

    def generate_expression(self, max_depth = 4):
        """
        Function to generate a valid mathematical expression. An expression consists of values from the functional
        set -> ['+', '-', '*', '/'] and values from a terminal set -> [random number between 0-50, X1,...,X5] where
        X1,..., are Altman's KPI ratios.
        :param max_depth: maximum depth of the regression tree.
        :return: valid expression <= maximum depth of tree.
        """

        # print out either a random number between 0 and 50, or a variable X1-X5.
        if max_depth == 1:
            terminals = [random() * 50, "X1", "X2", 'X3', "X4", "X5"]
            return self.__str__(choice(terminals))

        # include bracketing 20% of the time.
        rand = random()
        if rand <= 0.2:
            return '(' + self.generate_expression(max_depth - 1) + choice(self.operations) + self.generate_expression(max_depth - 1) + ')'
        else:
            return self.generate_expression(max_depth - 1) + choice(self.operations) + self.generate_expression(max_depth - 1)

    def __str__(self, num):
        """
        cast terminal value to a string.
        :param num:
        :return:
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

    def split_parents(self, parents):
        """
        function to split the parents to enable parents to be converted into prefix notation later.
        :param parents: the two parents selected from selection process
        :return: parents, split up into individual gene characteristics -> ["x1+1"] -> ["X1","+","1","end"]
        """
        split_list = [re.findall('\w+|\W', s[0]) for s in parents]

        [i.append("stop") for i in split_list]

        split_parents = [(split_list[i], parents[i][1]) for i in range(len(parents))]

        return split_parents

    def fix_dec(self, split_parents):
        """
        function to repair the split parents if expression contains floating point numbers.
        e.g ["1",".","234","+","X4"] ->["1.234","+","X4"]
        :param split_parents: the parents that have now been split up
        :return: the split parents with all correct values.
        """
        for p in split_parents:
            for item in p[0]:
                # print(item)
                if item == ".":
                    dec = p[0].index(item)

                    val1 = p[0][dec - 1]
                    val2 = p[0][dec + 1]
                    x = val1 + item + val2

                    p[0].insert(dec, x)
                    del p[0][dec - 1]
                    del p[0][dec]
                    del p[0][dec]
        return split_parents

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


# class to manipulate node objects that make up a tree.
class Node(object):
    """
    Class that creates a Node object which will either contain a functional operator e.g. +,-,*,/ or will hold a
    terminal value from the terminal set.
    """
    nodeid = 0

    def __repr__(self):
        """
        function to give a visual presentation of the tree.
        :return: node object and the parent that it is associated with.
        """
        if self.parent is not None:
            return "Node (" + str(self.value) + ") parent val:" + str(self.parent.value)
        else:
            return "Node (" + str(self.value) + ")"

    def __str__(self, level=0):
        """
        Function to print out a node at the current level within the tree.
        :param level: determines how many indents to put the each nod at.
        """
        ret = "\t" * level + self.__repr__() + "\n"
        if self.left_child is not None:
            ret += self.left_child.__str__(level + 1)
        if self.right_child is not None:
            ret += self.right_child.__str__(level + 1)
        return ret

    def __init__(self, value=None):
        """
        Constructor to initialise the nodes. Each node has an ID associated with it, a value, and the possibility of
        having two children.
        :param value:
        """
        Node.nodeid += 1
        self.nodenum = Node.nodeid
        self.value = value
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.checked = False
        self.checkedAgain = False

    def add_child(self, value, left=True):
        """
        Function to add a child into the tree. Start by adding children to the left branch of the parent, then add
        the next child to the right branch of the tree.
        :param value:
        :param left:
        :return:
        """
        if left is True:
            new_node = Node(value)
            self.left_child = new_node
            new_node.parent = self

        elif left is False:
            new_node = Node(value)
            self.right_child = new_node
            new_node.parent = self


# class to convert the infix notation into prefix notation.
class ToPrefixParser(object):
    """
    Class that converts infix notation to prefix notation, to get ready to construct a binary tree.
    """

    # every instance of a tree is a node.
    def __init__(self, val=None, left=None, right=None):

        self.val = val  # holds the value
        self.left = left  # holds the left child value
        self.right = right  # holds the right child value

    def __str__(self):
        return str(self.val)  # print out value of node

    def get_operation(self, token_list, expected):
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
            token_list[0:1] = []
            return ToPrefixParser(val=x)

    def get_product(self, token_list):
        a = self.get_number(token_list)

        if self.get_operation(token_list, '*'):
            b = self.get_product(token_list)
            return ToPrefixParser('*', a, b)
        elif self.get_operation(token_list, '/'):
            b = self.get_product(token_list)
            return ToPrefixParser("/", a, b)
        else:
            return a

    def get_expression(self, token_list):
        a = self.get_product(token_list)

        if self.get_operation(token_list, '-'):
            b = self.get_expression(token_list)
            return ToPrefixParser('-', a, b)
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
        # print("tk: ", token_list)
        prefix = list()

        prefix_list = list()
        pref_list = list()
        for i in token_list:
            tree = self.get_expression(i[0])
            y = self.print_tree_prefix(tree)
            prefix.append(y)
        for j in prefix:
            prefix_list.append(j.split())

        for k in range(len(prefix_list)):
            pref_list.append((prefix_list[k], token_list[k][1]))
        # print(pref_list)
        return pref_list


# manipulating the tree
class Tree(object):
    def __init__(self, root_node=None):
        self.root = root_node

    def make_tree(self, pref_list):
        # print("pref list: ",pref_list)
        nodes = list()
        nodenums = list()
        root_node = Node(pref_list[0])

        nodes.append(root_node)
        nodenums.append(root_node.nodenum)

        current_node = root_node

        pref_list.pop(0)

        # print("pref list now: ",pref_list)
        # while pref_list != []:
        while len(pref_list) > 0:
            # print("value of current node1: ",current_node)
            # print("pref list now2: ",pref_list)
            if current_node.value in GenMember.operations:
                # print("current node has value 3: ", current_node.value, "in param")
                if current_node.left_child is None:
                    current_node.add_child(pref_list[0], left=True)  # add a left child with its value
                    pref_list.pop(0)
                    current_node = current_node.left_child
                    # print("current node is now lc 4: ",current_node.value)
                    nodes.append(current_node)
                    nodenums.append(current_node.nodenum)
                elif current_node.left_child is not None and current_node.right_child is not None:
                    current_node = current_node.parent
                    nodes.append(current_node)
                    nodenums.append(current_node.nodenum)

                else:
                    current_node.add_child(pref_list[0], left=False)
                    pref_list.pop(0)
                    current_node = current_node.right_child
                    # print("current node now in right 5: ",current_node.value)
                    nodes.append(current_node)
                    nodenums.append(current_node.nodenum)
                    # print(current_node.value, " appended to l1")

            elif current_node.value not in GenMember.operations:
                # print("current node value 6: ", current_node.value, " not in param")
                current_node = current_node.parent
                # print("back at parent 7: ", current_node.value)
                if current_node.left_child is not None and current_node.right_child is not None:
                    current_node = current_node.parent
                    nodes.append(current_node)
                    nodenums.append(current_node.nodenum)

        return root_node, nodes, nodenums

    def print_full_tree(self, tree):
        return tree

    def find_subtree(self, tree, list_nodes, rnd_val):

        # print("list nodes: ", list_nodes)

        # print("value to be found: ",rnd_val)
        current_node = tree
        # print("current node value : ", current_node.value)
        if current_node.value == rnd_val[0] and current_node.nodenum == rnd_val[1]:
            # print("found")
            current_node.checked = True
            return current_node
        else:
            # if the current node left child exists:
            if current_node.left_child is not None and current_node.left_child.checked is False:
                # mark the current node as checked
                current_node.checked = True
                # move into the left child node.
                current_node = current_node.left_child
                return self.find_subtree(current_node, list_nodes, rnd_val)
            else:
                # if the curent node left child doesnt exist i.e is a leaf node
                current_node.checked = True
                # move to the parent
                if current_node.right_child is not None and current_node.right_child.checked is False:
                    current_node.checked = True
                    current_node = current_node.right_child
                    return self.find_subtree(current_node, list_nodes, rnd_val)

                else:
                    current_node = current_node.parent
                    # print("current node is now: ", current_node)
                    # if the current node left and right child both have been cheked, move to the curren node parent
                    if current_node.left_child.checked is True and current_node.right_child.checked is True:
                        current_node = current_node.parent
                        return self.find_subtree(current_node, list_nodes, rnd_val)
                    else:
                        # move pointer to the right child
                        current_node = current_node.right_child
                        return self.find_subtree(current_node, list_nodes, rnd_val)

    def select_random_val(self, list_nodes):
        # print("list nodes: ", list_nodes)
        ln = [(i.value, i.nodenum) for i in list_nodes]
        # print("ln: ",ln)

        root = list_nodes[0].nodenum
        x = list_nodes.pop(0)
        while True:
            y = choice(list_nodes)
            if y.nodenum != root:
                break
        list_nodes.insert(0, x)
        return y.value, y.nodenum, y

    def swap_nodes(self, tree_one, tree_two, list_nodes_one, list_nodes_two, node_one, node_two):

        node_one_parent = node_one.parent
        node_two_parent = node_two.parent
        if node_one_parent.left_child.value == node_one.value \
                and node_one_parent.left_child.nodenum == node_one.nodenum:
            node_one_parent.left_child = node_two

            node_one_parent.left_child.parent = node_one_parent

        else:
            node_one_parent.right_child = node_two

            node_one_parent.right_child.parent = node_one_parent

        if node_two_parent.left_child.value == node_two.value and node_two_parent.left_child.nodenum == \
                node_two.nodenum:
            node_two_parent.left_child = node_one
            node_two_parent.left_child.parent = node_two_parent

        else:
            node_two_parent.right_child = node_one
            node_two_parent.right_child.parent = node_two_parent

        return tree_one, tree_two, list_nodes_one, list_nodes_two

    def mutate_node(self, tree, list_nodes, node, fitness):
        # print("fitness:")
        # print(fitness)
        # print("node value: ", node.value, node.nodenum)
        # print("node type: ", type(node.value))

        # print("node list: ", list_nodes)
        # perform mutation
        if node.value in ['+', '-', '*', '/']:
            # print("here")
            node.value = choice(['+', '-', '*', '/'])
            # print("new mutated node: ",node.value, node.nodenum)
            # print(node)
            # print("new list of nodes: ", list_nodes)
            # print()
            # print("new tree")
            # print(tree)
            return tree, list_nodes  # return the new tree, new list_nodes, new mutated node.
        else:
            """
            based on the fitness, alter value by +0.1 if negative, -0.1 if positive. contstantly learning
            """
            if node.value not in ["X1", "X2", "X3", "X4", "X5"]:
                val = float(node.value)
                # print("value: ", val, type(val))
                if fitness > 0:
                    val -= 0.1
                    # print("value now",val)
                    node.value = str(val)
                else:
                    val += 0.1
                    # print("value now: ", val)
                    node.value = str(val)
            else:
                node.value = choice(["X1", "X2", "X3", "X4", "X5"])

            # print("new mutated node: ",node.value, node.nodenum, type(node.value))
            # print(node)
            # print("new list of nodes: ", list_nodes)
            return tree, list_nodes

    def get_child_one(self, child_one):
        return child_one

    def get_child_two(self, child_one, child_two):
        return child_two[len(child_one):]

    def make_list_nodes(self, tree, l1=list()):
        root_node = tree
        current_node = root_node
        # print(current_node)

        if current_node.checkedAgain is True and current_node.parent is None and current_node.left_child.checkedAgain \
                is True and current_node.right_child.checkedAgain is True:

            return l1
        else:
            # print("in here fam")
            if current_node.left_child is not None and current_node.left_child.checkedAgain is False:
                current_node.checkedAgain = True
                l1.append(current_node)
                current_node = current_node.left_child
                # print("current node 1: ", current_node.value)
                return self.make_list_nodes(current_node)
            else:
                # print("now im here")
                current_node.checkedAgain = True
                if current_node.right_child is not None and current_node.right_child.checkedAgain is False:
                    # print("moving into this bit")
                    # current_node.checkedAgain = True
                    current_node = current_node.right_child
                    # l1.append(current_node)
                    # print("current node : ", current_node.value)
                    return self.make_list_nodes(current_node)
                else:
                    # print("shit gone down")
                    if current_node not in l1:
                        l1.append(current_node)

                    current_node = current_node.parent
                    # print("current node : ", current_node.value)
                    if current_node.left_child.checkedAgain is True and current_node.right_child.checkedAgain is True \
                            and current_node.parent is not None:
                        current_node = current_node.parent
                        return self.make_list_nodes(current_node)
                    elif current_node.left_child.checkedAgain is True and current_node.right_child.checkedAgain \
                            is True and current_node.parent is None:
                        return self.make_list_nodes(current_node)
                    else:
                        current_node = current_node.right_child
                        return self.make_list_nodes(current_node)

    def build_child(self, tree, list_nodes):
        return tree, list_nodes


class ToInfixParser:
    def __init__(self):
        self.stack = []

    @staticmethod
    def deconstruct_tree(list_nodes):
        # print(list_nodes)
        pref = list()
        for i in list_nodes:
            pref.append(str(i.value))
        return pref

    def add_to_stack(self, p):
        if p in ['+', '-', '*', '/']:
            op1 = self.stack.pop()
            op2 = self.stack.pop()
            self.stack.append('({} {} {})'.format(op1, p, op2))
        else:
            self.stack.append(p)

    def convert_to_infix(self, l):
        l.reverse()
        for e in l:
            self.add_to_stack(e)
        return self.stack.pop()


def main():
    import sys
    import time
    start = time.time()
    loop_break = False
    to_pref = ToPrefixParser()
    tree = Tree()
    x_val = list()
    y_val = list()

    max_depth = 5
    population_size = 500
    max_iteration = 1000
    cross_over_rate = 0.9
    mutation_rate = 0.1
    current_population = GenMember()
    population = current_population.get_valid_expressions(max_depth, population_size)

    x = 1

    while x <= max_iteration:
        #     # print()
        # #     print("population!: ", population)
        #     print()
        if x == 1:
            population_fitness = current_population.get_fitness(population)
            # print("fitness: ",population_fitness)
            # else:

            # print("population = ", population)
            # print("fitness: ", population_fitness)
        for i in range(len(population_fitness)):
            if population_fitness[i] <= 2:
                # if get_fitness[i] ==0:
                print("#########################################################################")
                print(True)

                print("Iteration: ", x)
                print("fitness index:", population_fitness.index(population_fitness[i]))
                print("fitness: ", population_fitness[i])
                print()
                # print(population)
                print(population[i])
                # evale = current_population.eval_expressions(population[i])
                # print(evale)
                loop_break = True

            if loop_break is True:
                end = time.time()
                elapsed_time = end - start
                print("time elapsed: ", elapsed_time)
                print("here")
                return population[i]

        # # print(get_fitness)

        if x % 10 == 0:
            # print("parents: ", select_parents )

            # print("iteration: ", x)
            sys.stdout.write("iteration: {} \n".format(x))

            x_val.append(x)
            abs_list = [abs(j) for j in population_fitness]
            min_val = min(abs_list)
            # print("current best fitness: ", min_val)
            sys.stdout.write("current best fitness:{} \n".format(min_val))
            sys.stdout.flush()
            y_val.append(min_val)
            # print("time elapsed: ", time.time())
            # print("x_val: ", x_val)
            # print("y_val: ", y_val)

        if x == max_iteration:
            print("max iteration met")
            # print("fitness: ", get_fitness)
            abs_list = [abs(j) for j in population_fitness]
            min_val = min(abs_list)
            print("best fitness: ", min_val)
            index = abs_list.index(min_val)
            print("index: ", index)
            # print("population: ", population)
            print("equation: ", population[index])
            acc = 1 - (min_val / len(GenMember.data))
            print("acc: ", acc)
            print("acc: ", round(acc,2) * 100, "%")
            end = time.time()
            elapsed_time = end - start

            print("time elapsed: ", elapsed_time)
            # print("ben was right")

            plt.figure()
            plt.plot(x_val, y_val, "b", label="fitness")
            plt.xlabel("iteration")
            plt.ylabel("fitness")
            plt.legend(loc="best")
            plt.show()

            return population[index]

        select_parents = current_population.tournament_selection(population, population_fitness, 50)
        # print("parents selected", select_parents)
        split_parents = current_population.split_parents(select_parents)
        fix_decimals = current_population.fix_dec(split_parents)
        get_prefix_parents = to_pref.get_prefix_notation(fix_decimals)
        # print("prefix notation: ")
        # print("parent prefix: ", get_prefix_parents)
        #     #
        #     # print()
        #     # print("parent trees")
        parent_tree1 = get_prefix_parents[0]
        parent_tree2 = get_prefix_parents[1]
        parent_tree1_fitness = get_prefix_parents[0][1]
        parent_tree2_fitness = get_prefix_parents[1][1]

        #     # print("here")
        #     # print(parent_tree1_fitness)
        #     # print(parent_tree2_fitness)

        #     # print("p1 prefix:",parent_tree1)
        #     # print("p2 prefix:",parent_tree2)

        #     # print("making trees!")
        make_parent_tree_one = tree.make_tree(parent_tree1[0])
        make_parent_tree_two = tree.make_tree(parent_tree2[0])

        #     # print("Printing trees")
        #     # print("Tree one")
        #     # show_parent_tree_one = tree.print_full_tree(make_parent_tree_one[0])
        #     # print("parent 1")
        #     # print(show_parent_tree_one)
        # show_parent_tree_one_nodes = tree.print_full_tree(make_parent_tree_one[1])
        #     # print(show_parent_tree_one_nodes)
        #     # print("Tree two")
        #     # show_parent_tree_two = tree.print_full_tree(make_parent_tree_two[0])
        #     # print()
        #     # print("parent2: ")
        #     # print(show_parent_tree_two)
        # show_parent_tree_two_nodes = tree.print_full_tree(make_parent_tree_two[1])
        #     # print(show_parent_tree_two_nodes)
        # nodes_parent_tree_one = tree.print_full_tree(make_parent_tree_one[2])
        #     # print("parent one nodes: ", nodes_parent_tree_one)
        # nodes_parent_tree_two = tree.print_full_tree(make_parent_tree_two[2])
        #     # print("parent two nodes: ", nodes_parent_tree_two)

        #     # make a copy of the parents
        make_parent_tree_one_clone = copy.deepcopy(make_parent_tree_one)
        #     # show_parent_tree_one_clone = tree.print_full_tree(make_parent_tree_one_clone[0])
        #     # print("here")
        parent_tree1_fitness_clone = parent_tree1_fitness
        #     # print(parent_tree1_fitness_clone)
        #     # print(show_parent_tree_one_clone)

        make_parent_tree_two_clone = copy.deepcopy(make_parent_tree_two)
        #     # show_parent_tree_two_clone = tree.print_full_tree(make_parent_tree_two_clone[0])
        parent_tree2_fitness_clone = parent_tree2_fitness
        #     # print(parent_tree2_fitness_clone)
        #     # print(show_parent_tree_two_clone)

        nodes_parent_tree_one_clone = tree.print_full_tree(make_parent_tree_one_clone[2])
        #     # print("parent one nodes: ", nodes_parent_tree_one)
        nodes_parent_tree_two_clone = tree.print_full_tree(make_parent_tree_two_clone[2])
        #     # print("parent two nodes: ", nodes_parent_tree_two)

        rnd = random()
        #     # print("rnd : ", rnd)
        if rnd <= cross_over_rate:
            #         # print("crossing over")
            select_xover_node_one = tree.select_random_val(make_parent_tree_one_clone[1])
            #         # print("blooop: ",select_xover_node_one)
            select_xover_node_two = tree.select_random_val(make_parent_tree_two_clone[1])

            #         # print("selected xover point 1: ", select_xover_node_one)
            #         # print("selected xover point 2: ", select_xover_node_two)

            random_node_one = tree.find_subtree(make_parent_tree_one_clone[0], make_parent_tree_one_clone[1],
                                                select_xover_node_one)
            random_node_two = tree.find_subtree(make_parent_tree_two_clone[0], make_parent_tree_two_clone[1],
                                                select_xover_node_two)

            #         # print('swapping: ', random_node_one.value, random_node_one.nodenum, " with ", random_node
            # _two.value,random_node_two.nodenum)

            new_trees = tree.swap_nodes(make_parent_tree_one_clone[0], make_parent_tree_two_clone[0],
                                        nodes_parent_tree_one_clone, nodes_parent_tree_two_clone, random_node_one,
                                        random_node_two)
        else:
            #         # print("not crossing over")
            new_trees = [make_parent_tree_one_clone[0], make_parent_tree_two_clone[0]]
            #     # print()
        child_one = new_trees[0]
        child_two = new_trees[1]
        #     # print("child one")
        #     # print(child_one)
        #     # print()
        #     # print("building child two")
        #     # print(child_two)

        child_one_list_node = list(tree.make_list_nodes(child_one))
        child_two_list_node = list(tree.make_list_nodes(child_two))
        child_two_list_node = tree.get_child_two(child_one_list_node, child_two_list_node)

        #     # print("child one nodes: ", child_one_list_node)
        #     # print()
        #     # print("child two nodes: ", child_two_list_node)

        #     # print("mutating nodes: ")
        rnd = random()
        if rnd <= mutation_rate:
            #         # print("mutating nodes: ")
            node_to_mutate_one = tree.select_random_val(child_one_list_node)
            #         # print("node to mutate one: ",node_to_mutate_one)
            #         # print()
            node_to_mutate_two = tree.select_random_val(child_two_list_node)
            #         # print("node to mutate two: ",node_to_mutate_two)
            #         # print()

            new_child_one = tree.mutate_node(child_one, child_one_list_node, node_to_mutate_one[2],
                                             parent_tree1_fitness_clone)
            #         # print(new_child_one[0])
            #         #
            new_child_two = tree.mutate_node(child_two, child_two_list_node, node_to_mutate_two[2],
                                             parent_tree2_fitness_clone)
            #         # print(new_child_two[0])

        else:
            #         #
            #         # print("not mutating:")
            new_child_one = tree.build_child(child_one, child_one_list_node)
            new_child_two = tree.build_child(child_two, child_two_list_node)

            #     # print("deconstructing trees")
        p = ToInfixParser()
        #     # print("deconstructing child 1")
        deconstruct_child_one = ToInfixParser.deconstruct_tree(new_child_one[1])
        #     # print(deconstruct_child_one)

        c1 = p.convert_to_infix(deconstruct_child_one)
        c1 = c1.replace(" ", "")

        # print("child one: ", c1)
        # print("population :", population)

        # population.append(c1)

        # print("deconstructing child 2")
        deconstruct_child_two = ToInfixParser.deconstruct_tree(new_child_two[1])
        # print(deconstruct_child_two)

        c2 = p.convert_to_infix(deconstruct_child_two)
        c2 = c2.replace(" ", "")
        # print("child two:", c2)
        # print("jere")
        # get the fitness of the
        new_fit1 = current_population.get_fitness(c1, child=True)
        # print("child: ", c1)
        # print("fitness: ", new_fit1)
        new_fit2 = current_population.get_fitness(c2, child=True)
        # print("child 2: ", c2)
        # print("fitness: ", new_fit2)

        # print("population fitness:", population_fitness)
        update_population1 = current_population.update_population(population, population_fitness,
                                                                  c1, new_fit1, c2, new_fit2)

        population = update_population1[0]
        # print(" new population: ", population)
        population_fitness = update_population1[1]
        # print(" new population fitness:: ", population_fitness)

        x += 1


if __name__ == "__main__":
    optimal_expression = main()
    print("expression: ", optimal_expression)

    exp = list()
    exp.append(optimal_expression)
    # print("expression: ", exp)
    optimal_expression = exp

    # """
    # need to feed in the data into here
    # and print out the accuracy and classification
    # """
    # row = [0.414778159, 0.233613556, 0.094397251, 0.962392775, 1.016819994] # 0
    row = [0.092685526, 0.03905168, -0.005799479, 0.379883666, 1.024567409]  # 1
    # print("new expression: ", expression)
    for i in optimal_expression:
        new_exp = i.replace("X1", str(row[0])).replace("X2", str(row[1])).replace("X3", str(row[2])) \
            .replace("X4", str(row[3])).replace("X5", str(row[4]))

    # print("company data: ", row)
    eva = eval(new_exp)
    if eva >= 0:
        print("class 1 bankrupt")
    else:
        print("class 0 Non bankrupt")
