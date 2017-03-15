import sys
from random import randint, choice, random, sample
from math import log
import numpy as np
import re
import copy

OPS = ['+', '-', '*']
GROUP_PROB = 0.3
MIN_NUM, MAX_NUM = 0, 20

inputs = [4, 8, 12, 13]
output = [60, 64, 68, 69]
ideal_solution = "x1+8*(4+3)"
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
    l1 = []
    # # print('\n\n')
    # print("pref list: ", pref_list)
    # pref_list = ['+', '14', '+', '*', '12', '*', '-', '14', '3', 'X1', '-', '8', '*', '14', '5']
    # pref_list= ['+', '14', '+', '*', '12', '*', '-', '14', '3', 'X1','-','8', '*', '14', '5']
    root_node = Node(pref_list[0])
    l1.append(root_node)
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
                l1.append(current_node)
            elif current_node.left_child != None and current_node.right_child != None:
                current_node = current_node.parent
                l1.append(current_node)

            else:

                current_node.add_child(pref_list[0], left=False)
                pref_list.pop(0)
                current_node = current_node.right_child
                # print("current node now in right 5: ",current_node.value)
                l1.append(current_node)
                # print(current_node.value, " appended to l1")


        elif current_node.value not in ['-', '+', '*']:
            # print("current node value 6: ", current_node.value, " not in param")
            current_node = current_node.parent
            # print("back at parent 7: ", current_node.value)
            if current_node.left_child != None and current_node.right_child != None:
                current_node = current_node.parent
                l1.append(current_node)


    return root_node, l1




def print_full_tree(tree):
    return tree


def find_subtree(tree, list_nodes,rnd_val):
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
                return find_subtree(current_node, list_nodes,rnd_val)
    
    
            else:
                current_node = current_node.parent
                # print("current node is now: ", current_node)
                # if the current node left and right child both have been cheked, move to the curren node parent
                if current_node.left_child.checked == True and current_node.right_child.checked == True:
                    current_node = current_node.parent
                    return find_subtree(current_node, list_nodes,rnd_val)
                else:
                    # move pointer to the right child
                    current_node = current_node.right_child
                    return find_subtree(current_node, list_nodes,rnd_val)


def select_random_val(tree):
    # print("tttrreeeeeeeeeeeee: ", tree)
    #problem lays here, make sure the the root node is never selectee. 
    root = tree[0].nodenum

    x = tree.pop(0)
    
    while True:
        x = choice(tree)
        if x.nodenum != root:
            break
    # print("blaah: ",x.value, x.nodenum)
    return x.value,x.nodenum


def main():
    test = GenExp(8)
    generate_expressions = test.get_valid_expressions(8, 4)  # (maxNumber,Population size)
    # print("Population: ", generate_expressions)
    eval_exp = test.eval_expressions(generate_expressions)
    get_totals = test.get_totals(eval_exp)
    # print("totals: ", get_totals)
    get_fitness = test.get_mean_squared_fitness(get_totals)
    # print("fitness error: ", get_fitness)
    # print()
    # print("=======================================================")
    # print("parents")
    select_parents = test.select_parents(generate_expressions, get_fitness, 2)
    # print("parents selected: ", select_parents)
    split_parents = test.split_parents(select_parents)
    print("split parents: ", split_parents)
    # split_parents = [['(', '14', '+', '12', '*', '(', '14', '-', '3', ')', '*', 'X1', '+', '8', '-', '14', '*', '5', ')', 'end'], ['(', 'X1', '+', '(', '14', '*', '16', ')', ')', 'end']]

    # split_parents = [['X1', '-', '14', 'end'], ['(', 'X1', '-', '15', ')', '+', '13', 'end']]
    # split_parents = [['(', 'X1', '-', '17', '*', '15', ')', 'end'], ['X1', '+', '13', 'end']]
    # split_parents = [['(', 'X1', '+', 'X1', '+', '5', ')', '-', '18', '+', '17', '+', '10', 'end'], ['(', '(', 'X1', '*', '13', ')', '*', '18', '*', '6', ')', 'end']]
    get_prefix_parents = get_prefix_notation(split_parents)

    print(get_prefix_parents)
    parent_tree1 = get_prefix_parents[0]
    parent_tree2 = get_prefix_parents[1]

    # make the parent trees 
    make_parent_tree_one = make_tree(parent_tree1)
    make_parent_tree_two = make_tree(parent_tree2)

 


    show_parent_tree_one = print_full_tree(make_parent_tree_one[0])
    show_parent_tree_two = print_full_tree(make_parent_tree_two[0])

    # print(show_parent_tree1)
    # print(show_parent_tree2)

    make_child_tree_one = copy.deepcopy(make_parent_tree_one)
    show_child_tree_one = print_full_tree(make_child_tree_one[0])
    

    make_child_tree_two = copy.deepcopy(make_parent_tree_two)
    show_child_tree_two = print_full_tree(make_child_tree_two[0])
    

    select_child_node_one = select_random_val(make_child_tree_one[1])
    select_child_node_two = select_random_val(make_child_tree_two[1])
    print("selected node 1: ", select_child_node_one)
    print("selected node 2: ", select_child_node_two)
    
    random_node_one = find_subtree(make_child_tree_one[0],make_child_tree_one[1],select_child_node_one)
    random_node_two = find_subtree(make_child_tree_two[0],make_child_tree_two[1],select_child_node_two)



    print('swapping: ', random_node_one.value, random_node_one.nodenum , " with ",random_node_two.value, random_node_two.nodenum)
    FullTree.swap_nodes(random_node_one, random_node_two)


    print("parent 1: ")
    print(show_parent_tree_one)
    print('\n')
    print("parent 2: ")
    print(show_parent_tree_two)
    print('\n')
    print('makeing new tree 1: ')
    print(show_child_tree_one)
    print('\n')
    print('make new tree 2: ')
    print(show_child_tree_two)
    print('\n')
    

if __name__ == "__main__":
    main()