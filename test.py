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

class Node(object):
	def __init__(self,val, left_child = None, right_child= None):
		self.val= val
		self.left_child= left_child
		self.right_child = right_child
		self.parent = None


	def add_child(self,value,left=None, right = None):
		rnd = choice([0,1])
		if rnd==0:
			new_node = Node(value,left,right)
			self.left_child = new_node
			new_node.parent = self
		else:
			new_node = Node(value,left,right)
			self.right_child = new_node
			new_node.parent = self

	def get_operation(self,token_list,expected):
		if token_list[0] == expected:
			del token_list[0]
			return True
		else:
			return False

	def get_number(self,token_list):
		if self.get_operation(token_list,'('):
			x = self.get_expression(token_list)
			self.get_operation(token_list,')')

			return x
		else:
			x = token_list[0]
			if type(x) != type('o'): return None
			token_list[0:1] = []
			print("adding ",x)
			return self.add_child(x)

	def get_product(self,token_list):
		a = self.get_number(token_list)
		if self.get_operation(token_list,'*'):
			b = self.get_product(token_list)
			print("adding *")
			return self.add_child('*',a,b)
		else:
			return a

	def get_expression(self,token_list):
		a = self.get_product(token_list)
		if self.get_operation(token_list,'-'):
			b = self.get_expression(token_list)
			print("adding -")
			return self.add_child('-',a,b)
		elif self.get_operation(token_list,'+'):
			b = self.get_expression(token_list)
			print("adding +")
			return self.add_child('+',a,b)
		else:
			return a

	def print_tree_prefix(self,tree):
	    if tree.left == None:
	        sys.stdout.write("%s " % (tree.val))
	        return tree.val
	    else:
	        sys.stdout.write("%s " % (tree.val))
	        left = print_tree_prefix(tree.left)
	        right = print_tree_prefix(tree.right)
	        return tree.val + " " + left + ' ' + right + ''

class GenExp(object):
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




def main():
	l1 = ['(','X1', '+', '4', '*', 'X1', '-', '12', 'end']
	l2 = ['(','X1',')']
	t1= Node(l1[0])
	t2 = t1.get_expression(l1)

if __name__=="__main__":
	main()
