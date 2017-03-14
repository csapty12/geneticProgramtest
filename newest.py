import sys
from random import randint, choice, random, sample
from math import log
import numpy as np
import re

OPS = ['+', '-', '*']
GROUP_PROB = 0.3
MIN_NUM, MAX_NUM = 0, 20

inputs = [4, 8, 12, 13]
output = [60, 64, 68, 69]
ideal_solution = "x1+8*(4+3)"
randomVariable1 = [randint(MIN_NUM, MAX_NUM), "X1"]
num_nodes = 0
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
            self.right = GenExp(maxNumbers, maxdepth,
                                depth + 1)  # generate part of the expression (on the right)
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
        expression_list = []
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

    def get_mean_squared_fitness(self, differences):
        mean_sq = []

        for i in range(len(differences)):
            tmp = []
            for j in range(len(differences[i])):
                x = differences[i][j] ** 2
                tmp.append(x)

            mean_sq.append(tmp)
        mean = []
        for i in mean_sq:
            x = np.mean(i)
            # print(x)
            mean.append(x)
        return mean

    def select_parents(self, population, fitness, num_parents):
        print("population: ", population)
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
        # need to account for the root node. maybe a try catch...
        try:
            node_one_parent = node_one.parent

        except:
            node_one_parent = node_one
        try:
            node_two_parent = node_two.parent
        except:
            node_two_parent = node_two

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
    def __init__(self, val, left=None, right=None, nodesNumber=0, glob=True):
        if glob == True:
            global num_nodes
            num_nodes += 1
            self.nodeID = num_nodes
            self.val = val  # holds the value
            self.left = left  # holds the left child value
            self.right = right  # holds the right child value

        else:
            self.nodeID = nodesNumber
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
    l1 = []
    # print('\n\n')
    print("pref list: ", pref_list)
    root_node = Node(pref_list[0])
    l1.append(root_node)
    pref_list.pop(0)
    current_node = root_node
    while pref_list != []:
        if current_node.value in ['-', '+', '*']:

            if current_node.left_child == None:
                current_node.add_child(pref_list[0], left=True)  # add a left child with its value
                pref_list.pop(0)
                current_node = current_node.left_child
                l1.append(current_node)

            elif current_node.left_child != None:
                current_node.add_child(pref_list[0], left=False)
                pref_list.pop(0)
                current_node = current_node.right_child
                l1.append(current_node)

        elif current_node.value not in ['-', '+', '*']:
            current_node = current_node.parent
            if current_node.left_child != None and current_node.right_child != None:
                current_node = current_node.parent
                l1.append(current_node)

    return root_node, l1

def print_full_tree(tree):
	return tree


def find_subtree(tree,list_nodes):
	print("list nodes: ", list_nodes)
	x = select_random_val(list_nodes)
	print(x)
	current_node = tree
	if current_node.value == x:
		print("found")
		current_node.checked = True
		return current_node



			


	else:
		if current_node.value != x and current_node.checked == False:
			current_node.checked = True
			print("current node: ", current_node.value, " checked: ", current_node.checked)
			if current_node.left_child != None and current_node.left_child.checked == False:
				current_node = current_node.left_child
				print("next node to check ", current_node.value,  " checked: ", current_node.checked)
				return find_subtree(current_node, list_nodes)
			else:
				current_node = current_node.parent
				find_subtree(current_node, list_nodes)
		elif current_node.value != x and current_node.checked == True and current_node.left_child.checked == True and current_node.right_child.checked==False:
			print("current node: ", current_node.value, " checked: ", current_node.checked)
			if current_node.right_child!=None:
				current_node = current_node.right_child
				print("next node to check ", current_node.value,  " checked: ", current_node.checked)
				return find_subtree(current_node, list_nodes)


		# elif current_node.value != x and current_node.checked == True and current_node.left_child.checked == True and current_node.right_child.checked == True:
		# 	current_node = current_node.parent


		


		# if current_node.checked == True and current_node.right_child.checked == True and current_node.left_child.checked == True:
		# 	current_node = current_node.parent



		# if current_node.checked == True and current_node.parent.left_child.checked == True and current_node.parent.checked == True:
		# 	current_node = current_node.parent


			
			# if current_node.right_child != None:
			# 	current_node = current_node.right_child
			# 	current_node.checked = True
			# 	print("current node now here: ", current_node)
			# 	return find_subtree(current_node, list_nodes)



		

def select_random_val(tree):
    return '*'


def main():
    # test = GenExp(256)
    # generate_expressions = test.get_valid_expressions(256, 500)  # (maxNumber,Population size)
    # print("Population: ", generate_expressions)
    # eval_exp = test.eval_expressions(generate_expressions)
    # get_totals = test.get_totals(eval_exp)
    # print("totals: ", get_totals)
    # get_fitness = test.get_mean_squared_fitness(get_totals)
    # print("fitness error: ", get_fitness)
    # print()
    # print("=======================================================")
    # print("parents")
    # select_parents = test.select_parents(generate_expressions, get_fitness, 2)
    # print("parents selected: ", select_parents)
    # split_parents = test.split_parents(select_parents)
    # print("split parents: ", split_parents)
    split_parents = [['X1', '-', '14', 'end'], ['(', 'X1', '-', '15', ')', '+', '*', '13','7', 'end']]
    get_prefix = get_prefix_notation(split_parents)

    # print(get_prefix)
    tree1 = get_prefix[0]
    tree2 = get_prefix[1]
    # select_random_value = select_random_val(tree2)

    # print("randomly selected value: ", select_random_value)
    make_tree_one = make_tree(tree1)


    
    make_tree_two = make_tree(tree2)


    # prints out the tree as a list 
    print(make_tree_one[1])
    # prints out the tree as a list 
    print(make_tree_two[1])


    t1 = print_full_tree(make_tree_one[0])
    t2 = print_full_tree(make_tree_two[0])

    # select_random1 = select_random_val(x)
    # select_random2 = select_random_val(make_tree_two)
    x = find_subtree(make_tree_two[0],make_tree_two[1])
    print(x)


    # print('swapping: ', t1.left_child.value, ' and ', t2.left_child.value)
    # FullTree.swap_nodes(t1.left_child, t2.left_child)
    # print('make tree 1: ')
    # print(t1)
    # print('make tree 2: ')
    # print(t2)






    


if __name__ == "__main__":
    main()