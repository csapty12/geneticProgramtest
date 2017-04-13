from random import randint, choice, random, sample, uniform
from math import log, exp
import re
import copy
import numpy as np
import matplotlib.pyplot as plt



def read_data():
    from numpy import loadtxt
    cbd = loadtxt('dataset2.txt')  # read in the data
    class_labels_cbd = cbd[:, -1]  # get the classification categories.
    class_labels_cbd = [int(x) for x in class_labels_cbd]
    class_labels_cbd = np.asarray(class_labels_cbd, dtype = int)
    # print(class_labels_CBD)
    # print(len(class_labels_CBD))

    data_cbd = cbd[:, 0:-1]
    # print(data_CBD)
    return data_cbd, class_labels_cbd


class GenExp(object):
    OPS = ['+', '-', '*', "/"]
    bracketing = 0.3
    MIN_NUM, MAX_NUM = 1.0, 50.0

    # input1 = [[0.185841328, 0.229878245, 0.150353322, 2.267962444, 1.72085425],
    #           [0.16285377, 0.293619897, 0.148429586, 2.112106101, 1.726711829],
    #           [0.133612951, -0.012046736, 0.006117654, 0.217226284, 0.415622133],
    #           [0.014110552, 0.306378197, 0.025120196, 1.410020817, 0.220180238],
    #           [0.089459236, 0.498017156, 0.130726372, 4.138192315, 0.327646754],
    #           [0.18270599, 0.314689998, 0.146978993, 0.960817256, 2.441976174],
    #           [-0.182114883, -0.782310705, 0.050261097, 0.131185786, 1.157310705],
    #           [0.015860893, -0.104748498, 0.034454832, 0.068872667, 0.442914184],
    #           [0.266438247, 0.144166996, 0.358351154, 6.392246629, 0.957753361],
    #           [0.169908241, -0.091765234, 0.082865501, 0.420012564, 0.591372793]]

    # labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    read_data = read_data()

    input1 = read_data[0]
    labels = read_data[1]
    # randomVariable1 = [uniform(MIN_NUM, MAX_NUM), "X1", "X2", 'X3', "X4", "X5"]

    # function to initialise the expressions in any order, there is no structure for this.
    def __init__(self, maxNumbers, maxdepth=None, depth=0):
        random_variable1 = [uniform(self.MIN_NUM, self.MAX_NUM), "X1", "X2", 'X3', "X4", "X5"]
        self.left = None  # create the left and right nodes for an expression.
        self.right = None

        if maxdepth is None:
            maxdepth = log(maxNumbers, 2) - 1

        if depth < maxdepth and randint(0, maxdepth) > depth:
            self.left = GenExp(maxNumbers, maxdepth, depth + 1)  # generate part of the expression (on the left)
        else:
            self.left = choice(random_variable1)

        if depth < maxdepth and randint(0, maxdepth) > depth:
            self.right = GenExp(maxNumbers, maxdepth, depth + 1)  # generate part of the expression (on the right)
        else:
            self.right = randint(GenExp.MIN_NUM, GenExp.MAX_NUM)

        self.grouped = random() < GenExp.bracketing
        self.operator = choice(GenExp.OPS)

    def __str__(self):
        s = '{}{}{}'.format(self.left, self.operator, self.right)
        s = str(s)
        if self.grouped:
            return '({})'.format(s)
        else:
            return s

    def get_valid_expressions(self, maxNumbers, populationSize):
        expression_list = list()
        while len(expression_list) < populationSize:
            exps = GenExp(maxNumbers)
            str_exps = str(exps)
            expression_list.append(str_exps)
            # print out valid expressions with varibales
            expression_list = [i for i in expression_list if 'X1' and 'X2' and 'X3' and 'X4' and 'X5' in i]
        return expression_list

    def get_fitness(self, expression, child = False):
        if child == True:
            # print("trueeeee")
            exp = list()
            exp.append(expression)
            # print("expppp: ", exp)
            expression = exp

        else:
            expression = expression        
        row = np.asarray(GenExp.input1, dtype = object)
        # print("row:", row)
        new_row = row.T
        labels = GenExp.labels

        X1 = new_row[0] #len 10
        X2 = new_row[1]
        X3 = new_row[2]
        X4 = new_row[3]
        X5 = new_row[4]
        pred = list()
        
        for i in expression:
            
            # print(i)
            tmp = list()
            try:
           
                x = eval(i)
                # print("xxxxx: ",x)
                # print(type(x))
                if(isinstance(x,float) or isinstance(x,int)):
                    # print("yeeeasssss")
                    for i in range(len(X1)):
                        tmp = [9999] * len(X1)
                    pred.append(tmp)
                else:

                    for j in x:
                        if j >=0:
                            tmp.append(1)
                        else:
                            tmp.append(0)
                    pred.append(tmp)
                
            except ZeroDivisionError:
                # print("cannot divide by 0!!!")
                for i in range(len(X1)):
                    tmp = [9999] * len(X1)
                pred.append(tmp)
                

        # print("pred: ",pred)
        trfa = list()
        for i in range(len(pred)):
            tmp = list()
            [tmp.append(labels[j] == pred[i][j]) for j in range(len(pred[i]))]
            trfa.append(tmp)
        fitness = [len(i) - sum(i) for i in trfa]
        return fitness

    # def tournament_selection(self, population, fitness, selection_size):
    #     zipped_population = list(zip(population, fitness))
    #     # print("zipped population: ", zipped_population)
    #     parent_one = min(zipped_population, key=lambda t: t[1])
    #     # print(parent_one)
    
    #     p1_index = zipped_population.index(parent_one)
    #     # print("p1_index: ", p1_index)
    #     p1_tru = zipped_population[p1_index]
    #     zipped_population.pop(p1_index)
    
    #     # print("zipped population: ", zipped_population)
    
    #     parent_two = min(zipped_population, key=lambda t: t[1])
    #     p2_index = zipped_population.index(parent_two)
    #     # # print(p2_index)
    #     p2_tru = zipped_population[p2_index]
    #     # # print("hhhneww:,",p2_tru)
    #     # # print("candidate p2: ",candidate_p2)
    #     zipped_population.pop(p2_index)
    #     # # print("new candidates: ", candidates)
    
    #     parents = list()
    #     parents.append(p1_tru)
    #     parents.append(p2_tru)
    
    #     return parents
    

    def tournament_selection(self, population, fitness, selection_size):
        """
        function to select 4 random individuals in the population, and select the best one as the winner to be the parent
        do this twice to select the parents. 
        
        """
        zipped_population = list(zip(population,fitness))
        # print("zipped population: ", zipped_population)
        candidates = sample(zipped_population, selection_size)
        # print("candidates:",candidates)
        
        parent_one = min(candidates, key=lambda t: t[1])
        # print(parent_one)
        p1_index = zipped_population.index(parent_one)
        # print(p1_index)
        zipped_population.pop(p1_index)
        # print("new popilation:", zipped_population)

        candidates = sample(zipped_population, selection_size)
        parent_two = min(candidates, key=lambda t: t[1])
        p2_index = zipped_population.index(parent_two)
        zipped_population.pop(p2_index)
        # print("new popilation:", zipped_population)

        # print("parents1: ", parent_one)
        # print("parents2: ", parent_two)

        parents = list()
        parents.append(parent_one)
        parents.append(parent_two)
        return parents


    def split_parents(self, parents):
        # print("parents:",parents)
        split_list = [re.findall('\w+|\W', s[0]) for s in parents]

        [i.append("end") for i in split_list]

        split_parents = [(split_list[i], parents[i][1]) for i in range(len(parents))]

        return split_parents

    def fix_dec(self, split_parents):
        for i in split_parents:
            for item in i[0]:
                # print(item)
                if item == ".":
                    dec = i[0].index(item)

                    val1 = i[0][dec - 1]
                    val2 = i[0][dec + 1]
                    x = val1 + item + val2

                    i[0].insert(dec, x)
                    del i[0][dec - 1]
                    del i[0][dec]
                    del i[0][dec]
        return split_parents

    def update_population(self, population, fitness, c1,child_fit1,c2,child_fit2):
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
        child2 = list(zip(child2,child_fit2))
        # print("child2: ", child2)
        
        # # print("worst candidate 1: ")
        worst_one = max(zipped_population, key=lambda t: t[1])
        w1_index = zipped_population.index(worst_one)
        # print("worst one: ", worst_one)
        if child_fit1[0] <= worst_one[1]:
            zipped_population.pop(w1_index)
            zipped_population.append((c1,child_fit1[0]))

        # print("zipped population: ", zipped_population)

        worst_two = max(zipped_population, key=lambda t: t[1])
        w2_index = zipped_population.index(worst_two)
        # print("worst2: ", worst_two)

        if child_fit2[0] <= worst_two[1]:
            zipped_population.pop(w2_index)
            zipped_population.append((c2,child_fit2[0]))

        # print("zipped population: ", zipped_population)
        new_population = [i[0] for i in zipped_population]
        new_population_fitness = [i[1] for i in zipped_population]
        # print("new populationnnnnnnnnn:", new_population)
        # print("new population fitneess:", new_population_fitness)

        return new_population, new_population_fitness



# class to manipulate node objects that make up a tree.
class Node(object):
    nodeid = 0

    def __repr__(self):
        if self.parent is not None:
            return "Node (" + str(self.value) + ") parent val:" + str(self.parent.value)
        else:
            return "Node (" + str(self.value) + ")"

    def __str__(self, level=0):
        ret = "\t" * level + self.__repr__() + "\n"
        if self.left_child is not None:
            ret += self.left_child.__str__(level + 1)
        if self.right_child is not None:
            ret += self.right_child.__str__(level + 1)
        return ret

    def __init__(self, value=None):
        Node.nodeid += 1
        self.nodenum = Node.nodeid
        self.value = value
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.checked = False
        self.checkedAgain = False

    def add_child(self, value, left=True):
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
            return ToPrefixParser(x, None, None)

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
            return ToPrefixParser('-', a, b, )
        elif self.get_operation(token_list, '+'):
            b = self.get_expression(token_list)
            return ToPrefixParser('+', a, b)
        else:
            return a

    def print_tree_prefix(self, tree):
        if tree.left is None and tree.right is None:
            # sys.stdout.write("%s " % (tree.val))
            return tree.val
        else:
            # sys.stdout.write("%s " % (tree.val))
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
        for i in prefix:
            prefix_list.append(i.split())

        for i in range(len(prefix_list)):
            pref_list.append((prefix_list[i], token_list[i][1]))
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
            if current_node.value in GenExp.OPS:
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

            elif current_node.value not in GenExp.OPS:
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
                    val = val - 0.1
                    # print("value now",val)
                    node.value = str(val)
                else:
                    val = val + 0.1
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


def main(max_num, popn_size, max_iter, cross_over_rate =0.1, mutation_rate = 0.1, debug=False):
    import sys
    import time
    start = time.time()
    loop_break = False
    # print("Inputs: ", GenExp.input1)
    # print("Outputs: ", GenExp.output)
    current_population = GenExp(max_num)
    to_pref = ToPrefixParser()
    tree = Tree()
    x_val = list()
    y_val = list()

    population = current_population.get_valid_expressions(max_num, popn_size)  # (maxNumber,population size)
    # print("population!: ", population)
    # population_fitness = current_population.get_fitness(population)
    # print("population fitness: ",population_fitness)
    # print("best original fitness: ", min(population_fitness))

    x = 1

    while x <= max_iter:
    #     # print()
    # #     print("population!: ", population)
    #     print()
        if x ==1:
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
                break
        if loop_break == True:
            end = time.time()
            elapsedTime = end - start
            print("time elapsed: ", elapsedTime)
            print("here")
            return population[i]

        # # print(get_fitness)

        if x % 10== 0:
            # print("parents: ", select_parents )

            print("iteration: ", x)
            sys.stdout.write("iteration: {} \n".format(x))

            x_val.append(x)
            abs_list = [abs(i) for i in population_fitness]
            min_val = min(abs_list)
            print("current best fitness: ", min_val)
            sys.stdout.write("current best fitness:{} \n".format(min_val))
            sys.stdout.flush()
            y_val.append(min_val)
            # print("time elapsed: ", time.time())
            # print("x_val: ", x_val)
            # print("y_val: ", y_val)

        if x == max_iter:
            print("max iteration met")
            # print("fitness: ", get_fitness)
            abs_list = [abs(i) for i in population_fitness]
            min_val = min(abs_list)
            print("best fitness: ", min_val)
            index = abs_list.index(min_val)
            print("index: ", index)
            # print("population: ", population)
            print("equation: ", population[index])
            print("acc: ", 1-(min_val/len(GenExp.input1)))
            end = time.time()
            elapsedTime = end - start

            print("time elapsed: ", elapsedTime)
            print("ben was right")

            # plt.figure()
            # plt.plot(x_val, y_val, "b", label="fitness")
            # plt.legend(loc="best")
            # plt.show()

            return population[index]

        select_parents = current_population.tournament_selection(population, population_fitness, 4)
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
        show_parent_tree_one_nodes = tree.print_full_tree(make_parent_tree_one[1])
    #     # print(show_parent_tree_one_nodes)
    #     # print("Tree two")
    #     # show_parent_tree_two = tree.print_full_tree(make_parent_tree_two[0])
    #     # print()
    #     # print("parent2: ")
    #     # print(show_parent_tree_two)
        show_parent_tree_two_nodes = tree.print_full_tree(make_parent_tree_two[1])
    #     # print(show_parent_tree_two_nodes)
        nodes_parent_tree_one = tree.print_full_tree(make_parent_tree_one[2])
    #     # print("parent one nodes: ", nodes_parent_tree_one)
        nodes_parent_tree_two = tree.print_full_tree(make_parent_tree_two[2])
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

    #         # print('swapping: ', random_node_one.value, random_node_one.nodenum, " with ", random_node_two.value,random_node_two.nodenum)

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
        #get the fitness of the 
        new_fit1 = current_population.get_fitness(c1, child = True)
        # print("child: ", c1)
        # print("fitness: ", new_fit1)
        new_fit2 = current_population.get_fitness(c2, child = True)
        # print("child 2: ", c2)
        # print("fitness: ", new_fit2)

        # print("population fitness:", population_fitness)
        update_population1 = current_population.update_population(population, population_fitness, 
            c1,new_fit1,c2,new_fit2)

        
        population = update_population1[0]
        # print(" new population: ", population)
        population_fitness = update_population1[1]
        # print(" new population fitness:: ", population_fitness)

        x += 1


if __name__ == "__main__":
    # read_data()

    expression = main(max_num = 128, popn_size = 300, max_iter = 5, cross_over_rate = 0.9, mutation_rate = 0.1, debug=True)
    print("optimal expression: ", expression)
    exp = list()
    exp.append(expression)
    # print("expression: ", exp)
    expression = exp


    # """
    # need to feed in the data into here
    # and print out the accuracy and classification
    # """
    # row = [0.414778159, 0.233613556, 0.094397251, 0.962392775, 1.016819994] # 0
    row = [0.092685526, 0.03905168,  -0.005799479,  0.379883666, 1.024567409]# 1
    # print("new expression: ", expression)
    for i in expression:
        new_exp = i.replace("X1",str(row[0])).replace("X2",str(row[1])).replace("X3",str(row[2]))\
        .replace("X4",str(row[3])) .replace("X5",str(row[4]))

    # print("company data: ", row)
    eva = eval(new_exp)
    if eva >= 0:
        print("class 1 bankrupt")
    else:
        print("class 0 Non bankrupt")

    