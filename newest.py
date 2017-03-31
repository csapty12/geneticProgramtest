import sys
from random import randint, choice, random, sample
from math import log, sqrt
import numpy as np
import re
import copy

OPS = ['+', '-', '*']
GROUP_PROB = 0.3
MIN_NUM, MAX_NUM = 0, 20

inputs = [4, 8, 12, 13]
output = [60, 64, 68, 69]
ideal_solution = "X1+8*(4+3)"
randomVariable1 = [randint(MIN_NUM, MAX_NUM), "X1"]

nodeid = 0

class GenExp:
    """
	Class to generate a population, evaluate it, find the fitness function
	select parents for genetic operations and put children back into population

	"""

    def __init__(self, maxNumbers, maxdepth=None, depth=0):
        self.left = None  # create the left and right nodes for an expression.
        self.right = None

        if maxdepth is None:
            maxdepth = log(maxNumbers, 2) - 1

        if depth < maxdepth and randint(0, maxdepth) > depth:
            self.left = GenExp(maxNumbers, maxdepth, depth + 1)  # generate part of the expression (on the left)
        else:
            self.left = choice(randomVariable1)

        if depth < maxdepth and randint(0, maxdepth) > depth:
            self.right = GenExp(maxNumbers, maxdepth, depth + 1)  # generate part of the expression (on the right)
        else:
            self.right = randint(MIN_NUM, MAX_NUM)

        self.grouped = random() < GROUP_PROB  # if true, then bracketing around certain expressions will be allowed
        self.operator = choice(OPS)

    def __str__(self):
        s = '{0!s}{1}{2!s}'.format(self.left, self.operator, self.right)
        if self.grouped:
            return '({0})'.format(s)
        else:
            return s

    def get_valid_expressions(self, maxNumbers, populationSize):
        expression_list = list()
        while len(expression_list) < populationSize:
            exps = GenExp(maxNumbers)
            str_exps = str(exps)
            expression_list.append(str_exps)
            expression_list = [i for i in expression_list if 'X1' in i]  # print out valid expressions with varibales
        return expression_list

    def eval_expressions(self, expression):
        """
		function to evaluate the valid expression, where currently the value of X1 is a list of inputs.
		X1 is replaced with the current inputs to form a numerical expression (removing the X1 variable name) and returned
		"""
        eval_list = []
        X1 = inputs
        for i in expression:
            new_exps = [i.replace("X1", str(j)) for j in X1]
            eval_list.append(new_exps)
        return eval_list

    def get_totals(self, expression):
        """
		evaluate the expression and give the output.
		"""
        totals = []
        for i in expression:
            temp_totals = []
            for j in i:
                x = eval(j)
                temp_totals.append(x)
            totals.append(temp_totals)
        return totals

    # def get_mean_squared_fitness(self, differences):
    #     mean_sq = []

    #     for i in range(len(differences)):
    #         tmp = []
    #         for j in range(len(differences[i])):
    #             x = differences[i][j] ** 2
    #             tmp.append(x)

    #         mean_sq.append(tmp)
    #     mean = []
    #     for i in mean_sq:
    #         x = np.mean(i)
    #         # print(x)
    #         mean.append(x)
    #     return mean


    def get_mean_squared_fitness(self,totals):
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
                x = totals[i][j] - output[j]
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
            x = i / len(inputs)
            err = sqrt(x)
            root_mean_sq_err.append(err)
        # print("root mean sq err")
        # print(root_mean_sq_err)
        return root_mean_sq_err

    def select_parents(self, population, fitness, num_parents):
        # print("population: ", population)
        parents = sample(population, num_parents)

        return parents

    def split_parents(self, parents):
        split_list = [re.findall('\w+|\W', s) for s in parents]

        for i in split_list:
            i.append("end")
        return split_list

# building up the tree
class Node(object):
    def __repr__(self):
        if self.parent != None:
            return "Node (" + str(self.value) + ") parent val:" + str(self.parent.value)
        else:
            return "Node (" + str(self.value) + ")"

    def __str__(self, level=0):
        ret = "\t" * level + self.__repr__() + "\n"
        if self.left_child is not None:
            ret += (self.left_child).__str__(level + 1)
        if self.right_child is not None:
            ret += (self.right_child).__str__(level + 1)
        return ret

    def __init__(self, value):
        global nodeid
        nodeid += 1
        self.nodenum = nodeid
        self.value = value
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.checked = False
        self.checkedAgain = False

    def add_child(self, value, left=True):
        if left == True:
            new_node = Node(value)
            self.left_child = new_node
            new_node.parent = self

        elif left == False:
            new_node = Node(value)
            self.right_child = new_node
            new_node.parent = self

# manipulating the tree
class FullTree(object):
    def __init__(self, root_node):
        self.root = root_node

    @classmethod
    def swap_nodes(cls, node_one, node_two):
        # need to account for the root nodes.


        node_one_parent = node_one.parent

        node_two_parent = node_two.parent

        # figure out if node is left or right child
        if node_one_parent.left_child.value == node_one.value:
            node_one_parent.left_child = node_two
        else:
            # it is right child
            node_one_parent.right_child = node_two

        if node_two_parent.left_child.value == node_two.value:
            node_two_parent.left_child = node_one
        else:
            node_two_parent.right_child = node_one

        return


# bullshit class that needs to get sorted......
class Tree(object):
    # every instance of a tree is a node.
    def __init__(self, val, left=None, right=None, glob=True):
        if glob == True:
            self.val = val  # holds the value
            self.left = left  # holds the left child value
            self.right = right  # holds the right child value

        else:
            self.val = val  # holds the value
            self.left = left  # holds the left child value
            self.right = right  # holds the right child value

    def __str__(self):
        return str(self.val)  # print out value of node


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


def get_number(token_list):
    if get_operation(token_list, '('):
        x = get_expression(token_list)  # get the subexpression
        get_operation(token_list, ')')  # remove the closing parenthesis
        return x
    else:
        x = token_list[0]
        if type(x) != type("o"): return None
        token_list[0:1] = []
        return Tree(x, None, None)


def get_product(token_list):
    a = get_number(token_list)
    if get_operation(token_list, '*'):
        b = get_product(token_list)
        return Tree('*', a, b)
    else:
        return a


def get_expression(token_list):
    a = get_product(token_list)
    if get_operation(token_list, '-'):
        b = get_expression(token_list)
        return Tree('-', a, b, )
    elif get_operation(token_list, '+'):
        b = get_expression(token_list)
        return Tree('+', a, b)
    else:
        return a


def print_tree_prefix(tree):
    if tree.left == None and tree.right == None:
        # sys.stdout.write("%s " % (tree.val))
        return tree.val
    else:
        # sys.stdout.write("%s " % (tree.val))
        left = print_tree_prefix(tree.left)
        right = print_tree_prefix(tree.right)
        return tree.val + " " + left + ' ' + right + ''


def get_prefix_notation(token_list):
    prefix = []
    prefix_list = []
    for i in token_list:
        tree = get_expression(i)
        y = print_tree_prefix(tree)
        prefix.append(y)
    for i in prefix:
        prefix_list.append(i.split())
    return prefix_list


def make_tree(pref_list):
    nodes = []
    nodenums = []
    root_node = Node(pref_list[0])

    nodes.append(root_node)
    nodenums.append(root_node.nodenum)

    current_node = root_node

    pref_list.pop(0)

    # print("pref list now: ",pref_list)
    while pref_list != []:
        # print("value of current node1: ",current_node)
        # print("pref list now2: ",pref_list)
        if current_node.value in ['-', '+', '*']:
            # print("current node has value 3: ", current_node.value, "in param")
            if current_node.left_child == None:
                current_node.add_child(pref_list[0], left=True)  # add a left child with its value
                pref_list.pop(0)
                current_node = current_node.left_child
                # print("current node is now lc 4: ",current_node.value)
                nodes.append(current_node)
                nodenums.append(current_node.nodenum)
            elif current_node.left_child != None and current_node.right_child != None:
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


        elif current_node.value not in ['-', '+', '*']:
            # print("current node value 6: ", current_node.value, " not in param")
            current_node = current_node.parent
            # print("back at parent 7: ", current_node.value)
            if current_node.left_child != None and current_node.right_child != None:
                current_node = current_node.parent
                nodes.append(current_node)
                nodenums.append(current_node.nodenum)

    return root_node, nodes, nodenums


def print_full_tree(tree):
    return tree

def get_child_one(child_one):
    return child_one

def find_subtree(tree, list_nodes, rnd_val):
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
        if current_node.left_child != None and current_node.left_child.checked == False:
            # mark the current node as checked
            current_node.checked = True
            # move into the left child node.
            current_node = current_node.left_child
            return find_subtree(current_node, list_nodes, rnd_val)
        else:
            # if the curent node left child doesnt exist i.e is a leaf node
            current_node.checked = True
            # move to the parent
            if current_node.right_child != None and current_node.right_child.checked == False:
                current_node.checked = True
                current_node = current_node.right_child
                return find_subtree(current_node, list_nodes, rnd_val)


            else:
                current_node = current_node.parent
                # print("current node is now: ", current_node)
                # if the current node left and right child both have been cheked, move to the curren node parent
                if current_node.left_child.checked == True and current_node.right_child.checked == True:
                    current_node = current_node.parent
                    return find_subtree(current_node, list_nodes, rnd_val)
                else:
                    # move pointer to the right child
                    current_node = current_node.right_child
                    return find_subtree(current_node, list_nodes, rnd_val)

def select_random_val(list_nodes):
    ln = []
    for i in list_nodes:
        ln.append((i.value, i.nodenum))
    # print("ln: ",ln)
    
    root = list_nodes[0].nodenum
    x = list_nodes.pop(0)
    while True:
        y = choice(list_nodes)
        if y.nodenum != root:
            break
    list_nodes.insert(0,x)
    return y.value, y.nodenum, y

def mutate_node(tree, list_nodes, node):
    # print("tree: \n")
    # print(tree)
    # print()
    # print("list nodes: ", list_nodes)
    # print()
    # print("selected node: ",node.value, node.nodenum)
    # print("selected node subtree: ")
    # print(node)
    # perform mutation
    if node.value in ['+', '-', '*']:
        node.value = choice(['+', '-', '*'])
        # print("new mutated node: ",node.value, node.nodenum)
        # print(node)
        # print("new list of nodes: ", list_nodes)
        # print()
        # print("new tree")
        # print(tree)
        return tree, list_nodes  # return the new tree, new list_nodes, new mutated node.
    else:
        node.value = choice(randomVariable1)
        # print("new mutated node: ",node.value, node.nodenum)
        # print(node)
        # print("new list of nodes: ", list_nodes)
        # print()
        # print(tree)
        return tree, list_nodes


def make_list_nodes(tree, l1 = []):
    root_node = tree
    current_node = root_node
    # print(current_node)
    
    if current_node.checkedAgain==True and current_node.parent == None and current_node.left_child.checkedAgain == True and current_node.right_child.checkedAgain==True:
        
        return l1
    else:
        # print("in here fam")
        if current_node.left_child != None and current_node.left_child.checkedAgain == False:
            current_node.checkedAgain = True
            l1.append(current_node)
            current_node = current_node.left_child
            # print("current node 1: ", current_node.value)
            return make_list_nodes(current_node)
        else:
            # print("now im here")
            current_node.checkedAgain = True
            if current_node.right_child != None and current_node.right_child.checkedAgain == False:
                # print("moving into this bit")
                # current_node.checkedAgain = True
                current_node = current_node.right_child
                # l1.append(current_node)
                # print("current node : ", current_node.value)
                return make_list_nodes(current_node)
            else:
                # print("shit gone down")
                if current_node not in l1:
                    l1.append(current_node)

                current_node = current_node.parent
                # print("current node : ", current_node.value)
                if current_node.left_child.checkedAgain== True and current_node.right_child.checkedAgain == True and current_node.parent !=None:
                    current_node = current_node.parent
                    return make_list_nodes(current_node)
                elif current_node.left_child.checkedAgain== True and current_node.right_child.checkedAgain == True and current_node.parent ==None:
                    return make_list_nodes(current_node)
                else:
                    current_node = current_node.right_child
                    return make_list_nodes(current_node)

def get_child_two(child_one,child_two):
   
    return child_two[len(child_one):]

def swap_nodes(tree_one, tree_two, list_nodes_one, list_nodes_two, node_one, node_two):
    # print("ln1: ", list_nodes_one)
    # print("ln2: ", list_nodes_two)
    # # print("node one: ", node_one.value, node_one.nodenum)
    # # print("node two: ", node_two.value, node_two.nodenum)
    # # tmp = [1,2,5,3,2,4,2,4,2,2]
    # # new = [6,7,8,6,7,9,7,6,6,6]
    # # # print("old list1: ",tmp)
    # # # print("old list2: ", new)
    # # # indices_one = [i for i, x in enumerate(tmp) if x == 2]
    # # # indices_two = [i for i, x in enumerate(new) if x == 6]
    # # # print("indicies one: ", indices_one)
    # # # print("indicies two: ", indices_two)
    # indices_one = [i for i, x in enumerate(list_nodes_one) if x == node_one.nodenum]
    # # # print("indiciesssss: ", indices_one)
    # indices_two = [i for i, x in enumerate(list_nodes_two) if x == node_two.nodenum]
    # # # print("indicies 2: ", indices_two)
    
    # for i in indices_one:
    #     x = list_nodes_one[i]
    #     for j in indices_two:
    #         list_nodes_one[i],list_nodes_two[j] = list_nodes_two[j],list_nodes_one[i]
    # print("new n1l: ", list_nodes_one)
    # print("new nl2: ", list_nodes_two)

    node_one_parent = node_one.parent
    node_two_parent = node_two.parent
    if node_one_parent.left_child.value == node_one.value and node_one_parent.left_child.nodenum == node_one.nodenum:
        node_one_parent.left_child = node_two

        node_one_parent.left_child.parent = node_one_parent

    else:
        node_one_parent.right_child = node_two

        node_one_parent.right_child.parent = node_one_parent

    if node_two_parent.left_child.value == node_two.value and node_two_parent.left_child.nodenum == node_two.nodenum:
        node_two_parent.left_child = node_one
        node_two_parent.left_child.parent = node_two_parent

    else:
        node_two_parent.right_child = node_one
        node_two_parent.right_child.parent = node_two_parent

    return tree_one, tree_two, list_nodes_one, list_nodes_two
    # split_parents = [['(', '14', '+', '12', '*', '(', '14', '-', '3', ')', '*', 'X1', '+', '8', '-', '14', '*', '5', ')', 'end'], ['(', 'X1', '+', '(', '14', '*', '16', ')', ')', 'end']]

#     split_parents = [['X1', '-', '14', 'end'], ['(', 'X1', '-', '15', ')', '+', '13', 'end']]
#     split_parents = [['(', 'X1', '-', '17', '*', '15', ')', 'end'], ['X1', '+', '13', 'end']]
# 	  split_parents = [['(', 'X1', '+', 'X1', '+', '5', ')', '-', '18', '+', '17', '+', '10', 'end'], ['(', '(', 'X1', '*', '13', ')', '*', '18', '*', '6', ')', 'end']]


def deconstruct_tree(list_nodes):
    # print(list_nodes)
    pref = list()
    for i in list_nodes:
        pref.append(str(i.value))
    return pref

class Parser:
    def __init__ (self):
        self.stack = []

    def add_to_stack (self, p):
        if p in ['+', '-', '*']:
            op1 = self.stack.pop ()
            op2 = self.stack.pop ()
            self.stack.append ('({} {} {})' .format (op1, p, op2) )
        else:
            self.stack.append (p)

    def convert_to_infix (self, l):
        l.reverse ()
        for e in l:
            self.add_to_stack (e)
        return self.stack.pop ()

def update_population(population, fitness):
    # print("current population")
    # print(population)
    # print("fitenss: ")
    # print(fitness)
    worst_fit = max(fitness)
    # print("worst fit: ", worst_fit)
    for i in range(len(fitness)):
        if fitness[i] == worst_fit:
            # print(i)
            fitness.pop(i)
            population.pop(i)
            break
    # print("new population")
    # print(population)
    # print("new fitness: ")
    # print(fitness)
    return population, fitness


def main2():
    test = GenExp(256)
    population = test.get_valid_expressions(256,500)  # (maxNumber,Population size)
    x = 1
    while x<=300:
        # print("Population: ", population)

        eval_exp = test.eval_expressions(population)
        # print("eval exp: ", eval_exp) ##################################################################
        get_totals = test.get_totals(eval_exp)
        # print("totals: ", get_totals)
        get_fitness = test.get_mean_squared_fitness(get_totals)
        # print("fitness error: ", get_fitness)
        # if  x ==100:
        #     print("current best fitness:")
        #     print(min(get_fitness))

        # start the loop here 
        # print()
        # print("=======================================================")
        # print("parents")
        select_parents = test.select_parents(population, get_fitness, 2)
        #    # print("parents selected: ", select_parents)
        split_parents = test.split_parents(select_parents)
        # print("split parents: ", split_parents)
        # split_parents = [['(', 'X1', '-', '17', '*', '15', ')', 'end'], ['X1', '+', '13', 'end']]
        # split_parents = [['(', 'X1', '+', 'X1', '+', '5', ')', '-', '18', '+', '17', '+', '10', 'end'],
        #                  ['(', '(', 'X1', '*', '13', ')', '*', '18', '*', '6', ')', 'end']]



        # # split_parents = [
        # #     ['(', '14', '+', '12', '*', '(', '14', '-', '3', ')', '*', 'X1', '+', '8', '-', '14', '*', '5', ')', 'end'],
        # #     ['(', 'X1', '+', '(', '14', '*', '16', ')', ')', 'end']]
        get_prefix_parents = get_prefix_notation(split_parents)
        # print(get_prefix_parents)
        parent_tree1 = get_prefix_parents[0]
        parent_tree2 = get_prefix_parents[1]

        # make the parent trees
        make_parent_tree_one = make_tree(parent_tree1)
        make_parent_tree_two = make_tree(parent_tree2)



        # print the trees
        show_parent_tree_one = print_full_tree(make_parent_tree_one[0])
        show_parent_tree_two = print_full_tree(make_parent_tree_two[0])
        show_parent_tree_one_nodes = print_full_tree(make_parent_tree_one[1])
        show_parent_tree_two_nodes = print_full_tree(make_parent_tree_two[1])

        nodes_parent_tree_one = print_full_tree(make_parent_tree_one[2])
        # print("parent one nodes: ", nodes_parent_tree_one)
        nodes_parent_tree_two = print_full_tree(make_parent_tree_two[2])
        # print("parent two nodes: ", nodes_parent_tree_two)

        # make a copy of the parents
        make_child_tree_one = copy.deepcopy(make_parent_tree_one)
        show_child_tree_one = print_full_tree(make_child_tree_one[0])

        make_child_tree_two = copy.deepcopy(make_parent_tree_two)
        show_child_tree_two = print_full_tree(make_child_tree_two[0])

        # these are currently identicle to the parents
        show_child_tree_one_nodes = print_full_tree(make_child_tree_one[1])
        show_child_tree_two_nodes = print_full_tree(make_child_tree_two[1])
        # print("parent one: ")
        # print(show_child_tree_one)
        # print("parent 2")
        # print(show_child_tree_two)
        # select_child_node_one = (make_child_tree_one[0].right_child.value, make_child_tree_one[0].right_child.nodenum)
        # select_child_node_two = (make_child_tree_two[0].right_child.value, make_child_tree_two[0].right_child.nodenum)
        select_child_node_one = select_random_val(make_child_tree_one[1])
        select_child_node_two = select_random_val(make_child_tree_two[1])

        # print("selected node 1: ", select_child_node_one)
        # print("selected node 2: ", select_child_node_two)


        random_node_one = find_subtree(make_child_tree_one[0], make_child_tree_one[1], select_child_node_one)
        random_node_two = find_subtree(make_child_tree_two[0], make_child_tree_two[1], select_child_node_two)

        # print('swapping: ', random_node_one.value, random_node_one.nodenum, " with ", random_node_two.value,
        #       random_node_two.nodenum)

        new_trees = swap_nodes(make_child_tree_one[0], make_child_tree_two[0],
                               nodes_parent_tree_one, nodes_parent_tree_two, random_node_one, random_node_two)
        child_one = new_trees[0]
        child_two = new_trees[1]
        # print("child one")
        # print(child_one)
        # print()
        # print("building child two")
        # print(child_two)
        # print("complete!")
        # print()
        
        # print()
        # print("making nodes:")

        child_one_list_node = list(make_list_nodes(child_one))
        child_two_list_node = list(make_list_nodes(child_two))
        child_two_list_node = get_child_two(child_one_list_node, child_two_list_node)

        # print("child one nodes: ", child_one_list_node)
        # print()
        # print("child two nodes: ", child_two_list_node)

        # print("mutating nodes: ")
        node_to_mutate_one = select_random_val(child_one_list_node)
        # print("node to mutate one: ",node_to_mutate_one)
        print()
        node_to_mutate_two = select_random_val(child_two_list_node)
        # print("node to mutate two: ",node_to_mutate_two)
        print()

        new_child_one = mutate_node(child_one, child_one_list_node, node_to_mutate_one[2])
        # print(new_child_one[0])

        new_child_two = mutate_node(child_two, child_two_list_node, node_to_mutate_two[2])
        # print(new_child_two[0])
        print()
        print()
        # print("deconstructing trees")

        # print("deconstructing child 1")
        deconstruct_child_one= deconstruct_tree(new_child_one[1])
        # print(deconstruct_child_one)

        p = Parser()
        c1 = p.convert_to_infix(deconstruct_child_one)
        c1 = c1.replace(" ", "")
        population.append(c1)

        # print("deconstructing child 2")
        deconstruct_child_two= deconstruct_tree(new_child_two[1])
        # print(deconstruct_child_two)

        p = Parser()
        c2 = p.convert_to_infix(deconstruct_child_two)
        c2 = c2.replace(" ", "")
        population.append(c2)
        # print(population)
        eval_exp = test.eval_expressions(population)
        get_totals = test.get_totals(eval_exp)
        get_fitness = test.get_mean_squared_fitness(get_totals)
        # print("getting  new fitness: ")
        # print(get_fitness)
        update_popn1 = update_population(population, get_fitness)
        # print("popn update one: ")
        # print(update_popn1[0])
        # print("fitness of update 1")
        # print(update_popn1[1])
        print()
        print()
        print()
        update_popn2 = update_population(update_popn1[0], update_popn1[1])
        # print("popn update two: ")
        # print(update_popn2[0])
        # print("fitness of update 2")
        print(update_popn2[1])
        population = update_popn2[0]
        x+=1
    new_popn = update_popn2[1]
    min_val = min(new_popn)
    print("best fitness so far: ", min_val)
    for i in range(len(new_popn)):
        if new_popn[i] == min_val:
            print("index: ",i)
            print("equation ", population[i])


    # update_pop = update_population(population, get_fitness)
    # new_population = update_population(update_pop[0],update_pop[1])

    # print()
    # print()
    # print("new population")
    # print(new_population[0])
    # print("new fitnesses: ")
    # print(new_population[1])
    # population = new_population[0]


   

if __name__ == "__main__":
    main2()
    
