from random import choice, random, sample
import re
import copy
import numpy as np
import matplotlib.pyplot as plt


class Data(object):
    """
    Class to read data, and  manipulate the data such to be shuffled, as well as split th
    """

    def __init__(self, text_file):
        self.text_file = text_file

    def read_data(self, shuffle_d=False):
        """
        Function to load in the text file. Function splits the data into two sets.
        set 1: company data
        set 2: company data labels - either a 0 or 1.
        :return: tuple - (company data, company class)
        """
        from numpy import loadtxt
        from numpy.random import shuffle
        cfd = loadtxt(self.text_file)  # read in the data

        # if the shuffle flag true, then shuffle the data.
        if shuffle_d is True:
            shuffle(cfd)

        class_labels_cfd = cfd[:, -1]  # get the classification categories - [0,1].
        class_labels_cfd = [int(x) for x in class_labels_cfd]
        class_labels_cfd = np.asarray(class_labels_cfd, dtype=int)

        data_cfd = cfd[:, 0:-1]
        return data_cfd, class_labels_cfd


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

    def get_operation(self, expression, expected):
        """
        Function to compare the item in the expression list is the expected item.
        If the string values match, then pop it from the token list.
        :param expression: the expression list
        :param expected: the expected value of the list index
        :return: boolean
        """

        if expression[0] == expected:
            expression.pop(0)
            return True
        else:
            return False

    def is_number(self, expression):
        """
        Function that checks to see whether or not the value to be checked is a number or not.
        If the next value is a number, then return the value itself. Since it is a number, it will not have a left
        or right child as this is a leaf value. This function also handles parentheses to ensure that sub-expressions
        are handled.
        :param expression: the expression
        :return: a numerical value or None
        """
        if self.get_operation(expression, '('):
            x = self.get_expression(expression)  # get the subexpression
            self.get_operation(expression, ')')  # remove the closing parenthesis
            return x
        else:
            x = expression[0]
            if not isinstance(x, str):
                return None
            expression[0:1] = []
            return ToPrefixParser(val=x)

    def get_product(self, expression):
        """
        Function to put the * and / operator into the appropraite place when converting to prefix notation.
        * and / have a higher precedence than + and -, therefore these should be handled first.
        :param expression: expression being passed through
        :return: prefix notation of expression containing * and / in the right places.
        """
        a = self.is_number(expression)

        if self.get_operation(expression, '*'):
            b = self.get_product(expression)
            return ToPrefixParser('*', a, b)
        elif self.get_operation(expression, '/'):
            b = self.get_product(expression)
            return ToPrefixParser("/", a, b)
        else:
            return a

    def get_expression(self, expression):
        """
        Function to handle the - and + operators. get_sum tries to build a tree with a product on the left and a sum on
        the right. But if it doesn’t find a +, it just builds a product.
        :param expression: expression being passed in
        :return: the product or - or + in the correct places in prefix notation
        """
        op1 = self.get_product(expression)

        if self.get_operation(expression, '-'):
            op2 = self.get_expression(expression)
            return ToPrefixParser('-', op1, op2)
        elif self.get_operation(expression, '+'):
            op2 = self.get_expression(expression)
            return ToPrefixParser('+', op1, op2)
        else:
            return op1

    def print_tree_prefix(self, tree):
        """
        Function that takes in the tree, and prints out the tree in the correct prefix notation with 'stop' at the
        end of the prefix notation list -> ['*','3','4','stop']
        :param tree: the prefix notation list

        :return: the tree in appropraite positions.
        """
        if tree.left is None and tree.right is None:
            return tree.val
        else:
            left = self.print_tree_prefix(tree.left)
            right = self.print_tree_prefix(tree.right)
            return tree.val + " " + left + ' ' + right + ''

    def get_prefix_notation(self, parent_expression):
        """
        Function to take the parents expressions from infix notation and convert them to prefix notation.
        :param parent_expression: the parent expression in infix notation
        :return: parents in infix notation.
        """
        prefix = list()
        prefix_list = list()
        pref_list = list()
        for i in parent_expression:
            tree = self.get_expression(i[0])
            y = self.print_tree_prefix(tree)
            prefix.append(y)
        for j in prefix:
            prefix_list.append(j.split())

        for k in range(len(prefix_list)):
            pref_list.append((prefix_list[k], parent_expression[k][1]))
        return pref_list


# manipulating the tree
class Tree(object):
    """
    Class to make a tree, find a subtree, and and perform genetic operations on the tree.
    """
    def __init__(self, root_node=None):
        """
        Constructor to initialise the root node of a tree.
        :param root_node: None until value given to the node.
        """
        self.root = root_node

    def make_tree(self, pref_list):
        """
        Function to build the tree structure using the prefix expression
        :param pref_list: prefix list
        :return: root node, list of nodes and node ID's
        """

        nodes = list()
        nodenums = list()
        root_node = Node(pref_list[0])

        nodes.append(root_node)
        nodenums.append(root_node.nodenum)

        current_node = root_node  # use current node to point the current being being used.
        pref_list.pop(0)

        while len(pref_list) > 0:
            if current_node.value in GenMember.operations:
                if current_node.left_child is None:
                    current_node.add_child(pref_list[0], left=True)  # add a left child with its value
                    pref_list.pop(0)
                    current_node = current_node.left_child
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
                    nodes.append(current_node)
                    nodenums.append(current_node.nodenum)

            elif current_node.value not in GenMember.operations:
                current_node = current_node.parent

                if current_node.left_child is not None and current_node.right_child is not None:
                    current_node = current_node.parent
                    nodes.append(current_node)
                    nodenums.append(current_node.nodenum)

        return root_node, nodes, nodenums

    def print_full_tree(self, tree):
        """
        Function to print out the the tree in the tree structure, list of nodes or the node ID's
        :param tree: the tree
        :return: the representation of the tree.
        """
        return tree

    def find_subtree(self, tree, list_nodes, rnd_val):
        """
        Function to find a subtree within the tree to ensure that subtree exists. Function uses a depth first
        search to locate the subtree.
        :param tree: the tree to search
        :param list_nodes: list of nodes of that tree.
        :param rnd_val: the random value selected which find subtree is searching for.
        :return: the node that has been located within the subtree.
        """
        current_node = tree
        if current_node.value == rnd_val[0] and current_node.nodenum == rnd_val[1]:
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
                    # if the current node left and right child both have been cheked, move to the curren node parent
                    if current_node.left_child.checked is True and current_node.right_child.checked is True:
                        current_node = current_node.parent
                        return self.find_subtree(current_node, list_nodes, rnd_val)

                    else:
                        # move pointer to the right child
                        current_node = current_node.right_child
                        return self.find_subtree(current_node, list_nodes, rnd_val)

    def select_random_val(self, list_nodes):
        """
        Function to select a random node value from the list of nodes.
        :param list_nodes: list of nodes
        :return: the selected node value, the selected node ID, the selected node.
        """

        # pop the root node out to prevent root node being selected.
        root = list_nodes[0].nodenum
        x = list_nodes.pop(0)
        while True:
            y = choice(list_nodes)
            if y.nodenum != root:
                break
        list_nodes.insert(0, x)
        return y.value, y.nodenum, y

    def swap_nodes(self, tree_one, tree_two, node_one, node_two):
        """
        Function to take two trees and their selected subtrees to swap them over to simulate genetic crossover.
        :param tree_one: parent tree one
        :param tree_two: parent tree two
        :param node_one: parent tree one randomly selected node
        :param node_two: parent tree two randomly selected node
        :return: child tree one, child tree two.
        """

        # get the parents of each node selected
        node_one_parent = node_one.parent
        node_two_parent = node_two.parent

        # check value and nodenum to ensure correct subtree is being swapped.
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

        return tree_one, tree_two

    def mutate_node(self, tree, list_nodes, node):
        """
        Function to mutate the randomly selected node based on its current value and arity
        :param tree: tree to be mutated
        :param list_nodes: list of nodes associated with the tree
        :param node: the node selected for mutation
        :return: the updated tree, the updated list of nodes associated with the tree
        """

        # check if node selected is an operator.
        if node.value in ['+', '-', '*', '/']:
            # select operator based on same arity of node to be changed
            node.value = choice(['+', '-', '*', '/'])
            return tree, list_nodes  # return the new tree, new list_nodes, new mutated node.

        else:
            # check if terminal value and not a variable
            if node.value not in ["X1", "X2", "X3", "X4", "X5"]:
                # alter the value by a small amount
                val = float(node.value)
                val -= 0.1
                node.value = str(val)

            else:
                # if value is a variable, then select another variable
                node.value = choice(["X1", "X2", "X3", "X4", "X5"])

            return tree, list_nodes

    def get_child_one(self, child_one):
        """
        Function to get the first child that is produced
        :param child_one: the child produced
        :return: the first child
        """
        return child_one

    def get_child_two(self, child_one, child_two):
        """
        Function to get the second child independently
        :param child_one: the first child
        :param child_two: the second child
        :return: only the values of the second child.
        """
        return child_two[len(child_one):]

    def make_list_nodes(self, tree, l1=list()):
        """
        Function to make a list of nodes based on the tree that has been inputted
        :param tree: the tree to be converted to a list of nodes.
        :param l1: empty list which is appended to recursively.
        :return: the list of nodes
        """
        root_node = tree
        current_node = root_node

        if current_node.checkedAgain is True and current_node.parent is None and current_node.left_child.checkedAgain \
                is True and current_node.right_child.checkedAgain is True:
            return l1

        else:
            if current_node.left_child is not None and current_node.left_child.checkedAgain is False:
                current_node.checkedAgain = True
                l1.append(current_node)
                current_node = current_node.left_child
                return self.make_list_nodes(current_node)

            else:
                current_node.checkedAgain = True
                if current_node.right_child is not None and current_node.right_child.checkedAgain is False:
                    current_node = current_node.right_child
                    return self.make_list_nodes(current_node)

                else:
                    if current_node not in l1:
                        l1.append(current_node)

                    current_node = current_node.parent
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
        """
        function to return the children trees and list of nodes.
        :param tree: child tree
        :param list_nodes: list of nodes for that child tree.
        :return: tuple (tree, list of nodes)
        """
        return tree, list_nodes


class ToInfixParser:
    """
    Class to convert the prefix expression to infix notation.
    """
    def __init__(self):
        self.stack = []

    @staticmethod
    def deconstruct_tree(list_nodes):
        """

        :param list_nodes: list of nodes belonging to tree
        :return: the values of the tree in prefix notation.
        """
        pref = list()
        for i in list_nodes:
            pref.append(str(i.value))
        return pref

    def conv_inf(self, prefix_expr):
        """
        Function to convert the prefix expression back to infix notation.
        :param prefix_expr: prefix expression to be converted.
        :return: infix expression.
        """
        for e in prefix_expr[::-1]:
            if e not in GenMember.operations:
                self.stack.append(e)

            else:
                operand1 = self.stack.pop(-1)
                operand2 = self.stack.pop(-1)
                self.stack.append("({}{}{})".format(operand1, e, operand2))

        return self.stack.pop()[1:-1]


def train_gp(data_set, max_depth=3, population_size=500, max_iteration=100, cross_over_rate=0.9, mutation_rate=0.1):
    """
    Function to train the genetic program using the training dataset, based on user defined parameters.
    :param data_set: data set to be read into the program
    :param max_depth: max depth of a tree
    :param population_size: maximum popululation size
    :param max_iteration: stopping criteria if no solution is found within a reasonable iteration limit
    :param cross_over_rate: frequency of crossover expressed as a value between [0,1]
    :param mutation_rate: frequency of mutation expressed as a value between [0,1]
    :return: optimal expression found through training.
    """
    import sys
    import time
    start = time.time()
    loop_break = False
    to_pref = ToPrefixParser()
    tree = Tree()
    x_val = list()
    y_val = list()

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
        for index in range(len(population_fitness)):
            if population_fitness[index] <= 133:
                # if get_fitness[i] ==0:
                print("#########################################################################")
                print(True)

                print("Iteration: ", x)
                print("fitness index:", population_fitness.index(population_fitness[index]))
                print("fitness: ", population_fitness[index])
                print()
                # print(population)
                print(population[index])
                # evale = current_population.eval_expressions(population[i])
                # print(evale)
                loop_break = True

            if loop_break is True:
                end = time.time()
                elapsed_time = end - start
                print("time elapsed: ", elapsed_time)
                print("here")
                return population[index]

        # # print(get_fitness)

        if x % 10 == 0:
            # print("parents: ", select_parents )

            # print("iteration: ", x)
            # sys.stdout.write("iteration: {} \n".format(x))

            x_val.append(x)
            abs_list = [abs(f) for f in population_fitness]
            min_val = min(abs_list)
            # print("current best fitness: ", min_val)
            # sys.stdout.write("current best fitness:{} \n".format(min_val))
            sys.stdout.flush()
            y_val.append(min_val)
            # print("time elapsed: ", time.time())
            # print("x_val: ", x_val)
            # print("y_val: ", y_val)

        if x == max_iteration:
            print("max iteration met")
            # print("fitness: ", get_fitness)
            abs_list = [abs(fit) for fit in population_fitness]
            min_val = min(abs_list)
            print("best fitness: ", min_val)
            index = abs_list.index(min_val)
            print("index: ", index)
            # print("population: ", population)
            print("equation: ", population[index])
            acc = 1 - (min_val / len(GenMember.data))
            print("acc: ", acc)
            print("acc: ", round(acc, 2) * 100, "%")
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
        split_parents = to_pref.split_parents(select_parents)
        fix_decimals = to_pref.fix_dec(split_parents)
        get_prefix_parents = to_pref.get_prefix_notation(fix_decimals)
        # print("prefix notation: ")
        # print("parent prefix: ", get_prefix_parents)
        #     #
        #     # print()
        #     # print("parent trees")
        parent_tree1 = get_prefix_parents[0]
        parent_tree2 = get_prefix_parents[1]
        # parent_tree1_fitness = get_prefix_parents[0][1]
        # parent_tree2_fitness = get_prefix_parents[1][1]

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
        # parent_tree1_fitness_clone = parent_tree1_fitness
        #     # print(parent_tree1_fitness_clone)
        #     # print(show_parent_tree_one_clone)

        make_parent_tree_two_clone = copy.deepcopy(make_parent_tree_two)
        """
        #     # show_parent_tree_two_clone = tree.print_full_tree(make_parent_tree_two_clone[0])
        parent_tree2_fitness_clone = parent_tree2_fitness
        #     # print(parent_tree2_fitness_clone)
        #     # print(show_parent_tree_two_clone)
        """
        """
        nodes_parent_tree_one_clone = tree.print_full_tree(make_parent_tree_one_clone[2])
        # print("parent one nodes: ", nodes_parent_tree_one)
        nodes_parent_tree_two_clone = tree.print_full_tree(make_parent_tree_two_clone[2])
        # print("parent two nodes: ", nodes_parent_tree_two)
        """

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
                                        random_node_one, random_node_two)
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

            new_child_one = tree.mutate_node(child_one, child_one_list_node, node_to_mutate_one[2])
            #         # print(new_child_one[0])
            #         #
            new_child_two = tree.mutate_node(child_two, child_two_list_node, node_to_mutate_two[2])
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
        # print(deconstruct_child_one)

        c1 = p.conv_inf(deconstruct_child_one)
        c1 = c1.replace(" ", "")

        # print("child one: ", c1)
        # print("population :", population)

        # population.append(c1)

        # print("deconstructing child 2")
        deconstruct_child_two = ToInfixParser.deconstruct_tree(new_child_two[1])
        # print(deconstruct_child_two)

        c2 = p.conv_inf(deconstruct_child_two)
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
    import math

    optimal_expression =train_gp(data_set="dataset2.txt", max_depth=3, population_size=500, max_iteration=1500,
                              cross_over_rate=0.9,
                              mutation_rate=0.9)

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

    test_dataset = Data('testingDataSet.txt')
    # print("testing dataset: ")
    # test_dataset = test_dataset.read_data()
    # data = test_dataset[0]
    # labels = test_dataset[1]

    # print("data: ")
    # print(data)
    row = [[0.185841328, 0.229878245, 0.150353322, 2.267962444, 1.72085425],
           [0.16285377, 0.293619897, 0.148429586, 2.112106101, 1.726711829],
           [0.149332758, 0.347589881, 0.139985797, 1.689751437, 1.734865801],
           [0.137193647, 0.416721256, 0.147865432, 2.116532577, 1.761369401],
           [0.082350665, 0.480389313, 0.174387346, 2.342011704, 1.766493641]]
    label = [0, 0, 0, 0, 0]

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
            print("sig")
            print(sig)
            print()

# TODO - IMPLEMENT LEVEL CAP
# TODO - TESTING
# TODO - USE SIGMOID FUNCTION TO MAKE CLASSIFICAITON AND PROBABILITY.
# TODO - SIMPLE GUI INTERFACE 