from random import randint, choice, random, sample
from math import log, sqrt
import re
import copy


class GenExp:
    OPS = ['+', '-', '*']
    GROUP_PROB = 0.3
    MIN_NUM, MAX_NUM = 0, 20

    input1 = [4, 8, 12, 13]
    input2 = [1, 2, 3, 4]
    input3 = [5, 9, 2, 7]
    # input4 = [6, 4, 2, 1]
    # input5 = [7, 2, 6, 9]
    # output = [46, 66, 94,101 ]
    # X1 * 8 + (10 - X1)
    # (X1*10)-X2+(6+X2)
    # (X1+X3-6)+(X2+13)*18
    output = [8, 33, 72, 122]
    # output = [255, 400, 296, 320 ]
    # ideal_solution = "X1+X2*(7*X2)-3"
    randomVariable1 = [randint(MIN_NUM, MAX_NUM), "X1", "X2", "X3"]

    def __init__(self, maxNumbers, maxdepth=None, depth=0):
        self.left = None  # create the left and right nodes for an expression.
        self.right = None

        if maxdepth is None:
            maxdepth = log(maxNumbers, 2) - 1

        if depth < maxdepth and randint(0, maxdepth) > depth:
            self.left = GenExp(maxNumbers, maxdepth, depth + 1)  # generate part of the expression (on the left)
        else:
            self.left = choice(GenExp.randomVariable1)

        if depth < maxdepth and randint(0, maxdepth) > depth:
            self.right = GenExp(maxNumbers, maxdepth, depth + 1)  # generate part of the expression (on the right)
        else:
            self.right = randint(GenExp.MIN_NUM, GenExp.MAX_NUM)

        self.grouped = random() < GenExp.GROUP_PROB
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

            expression_list = [i for i in expression_list if 'X1' in i if "X2" in i if "X3" in i]  # print out valid expressions with varibales
        return expression_list

    def eval_expressions(self, expression):

        eval_list = list()

        x1 = GenExp.input1
        x2 = GenExp.input2
        x3 = GenExp.input3

        for i in expression:
            first_eval = [i.replace("X1", str(j)) for j in x1]
            tmp = list()
            for k in first_eval:
                snd_eval = [k.replace("X2", str(l)) for l in x2]
                tmp2 = list()
                for m in snd_eval:
                    trd_eval = [m.replace("X3", str(n)) for n in x3]
                    tmp2.append(trd_eval)
                tmp.append(tmp2)
            eval_list.append(tmp)
        # print("eval listttttt: ")
        # print(eval_list)

        
        return eval_list


    def get_totals(self, expression):

        totals = list()

        for i in expression:
            new_tmp = list()
            for j in i:
                tmp1 = list()
                for k in j:
                    tmp2 = list()
                    for l in k:
                        x = eval(l)
                        tmp2.append(x)
                    tmp1.append(tmp2)
                new_tmp.append(tmp1)
            totals.append(new_tmp)
        return totals


    # def test_fit(self,totals):
    #     import time
    #     t0 = time.time()
    #     differences = list()
    #     for i in range(len(totals)):
    #         tmp1 = list()
    #         for j in range(len(totals[i])):
    #             tmp2 = list()
    #             for k in range(len(totals[i][j])):
    #                 tmp3 = list()
    #                 for l in range(len(totals[i][j][k])):
    #                     x = totals[i][j][k][l] - GenExp.output[l]
    #                     sq = x **2
    #                     tmp3.append(sq)
    #                 tmp2.append(tmp3)
            
    #                 for l in range(len(tmp2)):
    #                     sum1 = sum(tmp2[l])
                
    #                 tmp1.append(sum1)
    #         differences.append(tmp1)
    #     print("differences:")
    #     print(differences)
    #     print()

    #     mean = [sum(i)/len(i) for i in differences]
    #     squareroot = [sqrt(i) for i in mean]
    #     print(mean)


    #     # squareroot = [sqrt(i) for i in mean]
    #     # print("sqrt:")
    #     # print(squareroot)

    #     t1 = time.time()
    #     total = t1-t0
    #     print("time taken: ", total)

    def get_mean_squared_fitness(self, totals):
        """
        first find the difference between actual output and expected output for each X1 value
        square the differences, sum them all up and divide by number len(x1)
        """
        import time
        t0 = time.time()
        differences = list()
        for i in range(len(totals)):
            tmp1 = list()
            for j in range(len(totals[i])):
                tmp2 = list()
                for k in range(len(totals[i][j])):
                    tmp3 = list()
                    for l in range(len(totals[i][j][k])):
                        x = totals[i][j][k][l] - GenExp.output[l]
                        sq = x **2
                        tmp3.append(sq)
                    tmp2.append(tmp3)
                tmp1.append(tmp2)
            differences.append(tmp1)
        # print("differeces")
        # print(differences)

        print()


        print()
        new_total = list()
        for i in range(len(differences)):
            tmp1 = list()
            for j in range(len(differences[i])):
                tmp2 = list()
                for k in range(len(differences[i][j])):
                    x = sum(differences[i][j][k])
                    tmp2.append(x)
                tmp1.append(tmp2)
            new_total.append(tmp1)
        # print("old totals:")
        # print(new_total)



        root_mean_sq_err = list()
        for i in new_total:
            tmp= list()
            for j in i:
                tmp2 = list()
                for k in j:
                    x = k/len(GenExp.input1)
                    err = sqrt(x)
                    tmp2.append(err)
                tmp.append(tmp2)
            root_mean_sq_err.append(tmp)

        average_err = list()
        for i in root_mean_sq_err:
            for j in i:
                x = sum(j)
                y = x/len(j)
            average_err.append(y)


        return average_err


    def select_parents(self, population, num_parents):
        # print("population: ", population)
        parents = choice(population, num_parents)

        return parents

    def tournament_selection(self, population, fitness, selection_size):
        # print("population")
        # print(population)
        # print("population fitnesses")
        # print(fitness)
        # print("selection size: ")
        # print(selection_size)
        zipped_population = list(zip(population, fitness))
        # print("zipped list:")
        # print(zipped_population)
        candidates = sample(zipped_population, selection_size)
        # print("selection")
        # print(candidates)
        parent_one = min(candidates, key=lambda t: t[1])

        p1_index = candidates.index(parent_one)
        # print(p1_index)
        candidates.pop(p1_index)
        # print("candidates:")
        # print(candidates)
        parent_two = min(candidates, key=lambda t: t[1])
        p2_index = candidates.index(parent_two)
        # print(p2_index)
        candidates.pop(p2_index)
        # print("candidates:")
        # print(candidates)
        parents = list()
        # print(parent_one)
        # print(parent_two)
        parents.append(parent_one[0])
        parents.append(parent_two[0])
        # print(parents)

        return parents

    def split_parents(self, parents):
        split_list = [re.findall('\w+|\W', s) for s in parents]

        for i in split_list:
            i.append("end")
        return split_list

    def update_population(self, population, fitness):
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
        prefix = []

        prefix_list = []
        for i in token_list:
            tree = self.get_expression(i)
            y = self.print_tree_prefix(tree)
            prefix.append(y)
        for i in prefix:
            prefix_list.append(i.split())
        return prefix_list


# manipulating the tree
class Tree(object):
    def __init__(self, root_node=None):
        self.root = root_node

    def make_tree(self, pref_list):

        nodes = []
        nodenums = []
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
            if current_node.value in ['-', '+', '*']:
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

            elif current_node.value not in ['-', '+', '*']:
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
        list_nodes.insert(0, x)
        return y.value, y.nodenum, y

    def swap_nodes(self, tree_one, tree_two, list_nodes_one, list_nodes_two, node_one, node_two):
        node_one_parent = node_one.parent
        node_two_parent = node_two.parent
        if node_one_parent.left_child.value == node_one.value\
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

    def mutate_node(self, tree, list_nodes, node):

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
            node.value = choice(GenExp.randomVariable1)
            # print("new mutated node: ",node.value, node.nodenum)
            # print(node)
            # print("new list of nodes: ", list_nodes)
            # print()
            # print(tree)
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
                    elif current_node.left_child.checkedAgain is True and current_node.right_child.checkedAgain\
                            is True and current_node.parent is None:
                        return self.make_list_nodes(current_node)
                    else:
                        current_node = current_node.right_child
                        return self.make_list_nodes(current_node)

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
        if p in ['+', '-', '*']:
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


def main2(max_num, popn_size, num_iterations):
    import time
    t0 = time.time()
    current_population = GenExp(max_num)
    to_pref = ToPrefixParser()
    tree = Tree()

    population = current_population.get_valid_expressions(max_num, popn_size)  # (maxNumber,Population size)
    print("population")
    print(population)
    x = 1
    # while x <= num_iterations:
        # print("Population: ", population)

        # eval_exp = current_population.eval_expressions(population)
        # # print("eval exp: ", eval_exp) ##################################################################
        # get_totals = current_population.get_totals(eval_exp)
        # # print("totals: ", get_totals)
        # get_fitness = current_population.get_mean_squared_fitness(get_totals)
        # # test_fitness = current_population.test_fit(get_totals)

        # # print("fitness error: ", get_fitness)

        # if min(get_fitness) <= 3:
        #     # print("getting the fitnesssss: ", get_fitness)
        #     print("the new min fitness is : ", min(get_fitness))
        #     print("it exists")
        #     break

        # # print()
        # # print("=======================================================")
        # # print("parents")
        # # select_parents = current_population.select_parents(population, 2)
        # # print("parents selected: ", select_parents)

        # select_parents = current_population.tournament_selection(population, get_fitness, 4)
        # # print("selecting parents:")
        # # print(select_parents)
        # split_parents = current_population.split_parents(select_parents)

        # # # # split_parents = [
        # # # ['(', '14', '+', '12', '*', '(', '14', '-', '3', ')', '*', 'X1', '+', '8', '-', '14', '*', '5', ')', 'end'],
        # # # ['(', 'X1', '+', '(', '14', '*', '16', ')', ')', 'end']]
        # get_prefix_parents = to_pref.get_prefix_notation(split_parents)
        # # print(get_prefix_parents)
        # parent_tree1 = get_prefix_parents[0]
        # parent_tree2 = get_prefix_parents[1]

        # # make the parent trees
        # make_parent_tree_one = tree.make_tree(parent_tree1)
        # make_parent_tree_two = tree.make_tree(parent_tree2)

        # # print the trees
        # # show_parent_tree_one = tree.print_full_tree(make_parent_tree_one[0])
        # # show_parent_tree_two = tree.print_full_tree(make_parent_tree_two[0])
        # # show_parent_tree_one_nodes = tree.print_full_tree(make_parent_tree_one[1])
        # # show_parent_tree_two_nodes = tree.print_full_tree(make_parent_tree_two[1])

        # nodes_parent_tree_one = tree.print_full_tree(make_parent_tree_one[2])
        # # print("parent one nodes: ", nodes_parent_tree_one)
        # nodes_parent_tree_two = tree.print_full_tree(make_parent_tree_two[2])
        # # print("parent two nodes: ", nodes_parent_tree_two)

        # # make a copy of the parents
        # make_child_tree_one = copy.deepcopy(make_parent_tree_one)
        # # show_child_tree_one = tree.print_full_tree(make_child_tree_one[0])

        # make_child_tree_two = copy.deepcopy(make_parent_tree_two)
        # # show_child_tree_two = tree.print_full_tree(make_child_tree_two[0])

        # # these are currently identicle to the parents
        # # show_child_tree_one_nodes = tree.print_full_tree(make_child_tree_one[1])
        # # show_child_tree_two_nodes = tree.print_full_tree(make_child_tree_two[1])
        # # print("parent one: ")
        # # print(show_child_tree_one)
        # # print("parent 2")
        # # print(show_child_tree_two)
        # select_child_node_one = tree.select_random_val(make_child_tree_one[1])
        # select_child_node_two = tree.select_random_val(make_child_tree_two[1])

        # random_node_one = tree.find_subtree(make_child_tree_one[0], make_child_tree_one[1], select_child_node_one)
        # random_node_two = tree.find_subtree(make_child_tree_two[0], make_child_tree_two[1], select_child_node_two)

        # # print('swapping: ', random_node_one.value, random_node_one.nodenum, " with ", random_node_two.value,
        # #       random_node_two.nodenum)

        # new_trees = tree.swap_nodes(make_child_tree_one[0], make_child_tree_two[0],
        #                             nodes_parent_tree_one, nodes_parent_tree_two, random_node_one, random_node_two)
        # child_one = new_trees[0]
        # child_two = new_trees[1]

        # child_one_list_node = list(tree.make_list_nodes(child_one))
        # child_two_list_node = list(tree.make_list_nodes(child_two))
        # child_two_list_node = tree.get_child_two(child_one_list_node, child_two_list_node)

        # # print("child one nodes: ", child_one_list_node)
        # # print()
        # # print("child two nodes: ", child_two_list_node)

        # # print("mutating nodes: ")
        # node_to_mutate_one = tree.select_random_val(child_one_list_node)
        # # print("node to mutate one: ",node_to_mutate_one)
        # # print()
        # node_to_mutate_two = tree.select_random_val(child_two_list_node)
        # # print("node to mutate two: ",node_to_mutate_two)
        # # print()

        # new_child_one = tree.mutate_node(child_one, child_one_list_node, node_to_mutate_one[2])
        # # print(new_child_one[0])

        # new_child_two = tree.mutate_node(child_two, child_two_list_node, node_to_mutate_two[2])
        # # print(new_child_two[0])
        # # print()
        # # print()
        # # print("deconstructing trees")
        # p = ToInfixParser()
        # # print("deconstructing child 1")
        # deconstruct_child_one = ToInfixParser.deconstruct_tree(new_child_one[1])
        # # print(deconstruct_child_one)

        # c1 = p.convert_to_infix(deconstruct_child_one)
        # c1 = c1.replace(" ", "")
        # population.append(c1)

        # # print("deconstructing child 2")
        # deconstruct_child_two = ToInfixParser.deconstruct_tree(new_child_two[1])
        # # print(deconstruct_child_two)

        # c2 = p.convert_to_infix(deconstruct_child_two)
        # c2 = c2.replace(" ", "")
        # population.append(c2)
        # # print(population)
        # eval_exp = current_population.eval_expressions(population)
        # get_totals = current_population.get_totals(eval_exp)
        # get_fitness = current_population.get_mean_squared_fitness(get_totals)
        # # print("getting  new fitness: ")
        # # print("getting fitness: ",get_fitness)
        # update_popn1 = current_population.update_population(population, get_fitness)
        # # print("popn update one: ")
        # # print(update_popn1[0])
        # # print("fitness of update 1")
        # # print(update_popn1[1])
        # # print()
        # # print()
        # # print()
        # update_popn2 = current_population.update_population(update_popn1[0], update_popn1[1])
        # # print("new population : ", update_popn2[0])
        # # print("new fitnesses: ", update_popn2[1])
        # population = update_popn2[0]

        # # print("popn update two: ")
        # # print(update_popn2[0])
        # # print("fitness of update 2")
        # # print("updated population fitness: ", update_popn2[1])

        
        # population = update_popn2[0]
        # # print("the newest population is ::::::: ", population)

        # print("current iteration: ", x)
        # print("current best fitness: ",  min(update_popn2[1]))
        # x += 1

    # new_popn = get_fitness
    # min_val = min(new_popn)
    # print("best fitness so far: ", min_val)
    # print("number of iterations: ", x-1)
    # for i in range(len(new_popn)):
    #     if new_popn[i] == min_val:
    #         print("index: ", i)
    #         best = population[i]
    #         print("equation ", best)
    # t1 = time.time()
    # time = t1-t0
    # print("time taken: ", time)


if __name__ == "__main__":
    main2(16,4,1)
