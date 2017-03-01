import sys
from random import randint,choice,random,sample
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

class Node(object):
	def __init__(self,val,left = None, right = None, nodeID =None):

		self.val = val # holds the value
		self.left= left # holds the left child value
		self.right = right # holds the right child value
		self.nodeID = nodeID

	def __str__(self):

		return str(self.val) #print out value of node 




class Tree(object):
	num_nodes =0
	nodes = []
	def __init__(self):
		num_nodes =0
		nodes = []
	def add_node(self,val,left = None, right= None):
		self.num_nodes +=1
		self.nodes.append(Node(val,left,right,self.num_nodes))


	def get_operation(self,token_list,expected):
		"""
		compares the expected token to the first token on the list. if they match, remove it, return True
		this is to get the operator
		"""
		if token_list[0] ==expected:
			del  token_list[0]
			return True
		else:
			return False

	def get_number(self,token_list):
	    if self.get_operation(token_list, '('):
	        x = self.get_expression(token_list)         # get the subexpression
	        self.get_operation(token_list, ')')      # remove the closing parenthesis
	        return x
	    else:
	        x = token_list[0]
	        if type(x) != type("o"): return None
	        token_list[0:1] = []
	        self.add_node(x, None, None)



	def get_product(self,token_list):
	    a = self.get_number(token_list,)
	    if self.get_operation(token_list, '*'):
	        b = self.get_product(token_list)
	        self.add_node('*', a, b)
	    else:
	        return a


	def get_expression(self,token_list):
		a = self.get_product(token_list)
		if self.get_operation(token_list, '-'):
			b = self.get_expression(token_list)
			self.add_node('-', a, b)
		elif self.get_operation(token_list, '+'):
			b = self.get_expression(token_list)
			self.add_node('+', a, b)
		else:
			return a




# class GenExp:
#     """
#     Class to generate a population, evaluate it, find the fitness function
#     select parents for genetic operations and put children back into population

#     """

#     def __init__(self, maxNumbers, maxdepth=None, depth=0):
#         self.left = None  # create the left and right nodes for an expression.
#         self.right = None

#         if maxdepth is None:
#             maxdepth = log(maxNumbers, 2) - 1

#         if depth < maxdepth and randint(0, maxdepth) > depth:
#             self.left = GenExp(maxNumbers, maxdepth, depth + 1)  # generate part of the expression (on the left)
#         else:
#             self.left = choice(randomVariable1)

#         if depth < maxdepth and randint(0, maxdepth) > depth:
#             self.right = GenExp(maxNumbers, maxdepth,
#                                         depth + 1)  # generate part of the expression (on the right)
#         else:
#             self.right = randint(MIN_NUM, MAX_NUM)

#         self.grouped = random() < GROUP_PROB  # if true, then bracketing around certain expressions will be allowed
#         self.operator = choice(OPS)

#     def __str__(self):
#         s = '{0!s}{1}{2!s}'.format(self.left, self.operator, self.right)
#         if self.grouped:
#             return '({0})'.format(s)
#         else:
#             return s


#     def get_valid_expressions(self, maxNumbers, populationSize):
#         expression_list = []
#         while len(expression_list) < populationSize:
#             exps = GenExp(maxNumbers)
#             str_exps = str(exps)
#             expression_list.append(str_exps)
#             expression_list = [i for i in expression_list if 'X1' in i]  # print out valid expressions with varibales
#         return expression_list

#     def eval_expressions(self, expression):
#         """
#         function to evaluate the valid expression, where currently the value of X1 is a list of inputs.
#         X1 is replaced with the current inputs to form a numerical expression (removing the X1 variable name) and returned
#         """
#         eval_list = []
#         X1 = inputs
#         for i in expression:
#             new_exps = [i.replace("X1", str(j)) for j in X1]
#             eval_list.append(new_exps)
#         return eval_list



#     def get_totals(self, expression):
#         """
# 		evaluate the expression and give the output. 
#         """
#         totals = []
#         for i in expression:
#             temp_totals = []
#             for j in i:
#                 x = eval(j)
#                 temp_totals.append(x)
#             totals.append(temp_totals)
#         return totals


#     def get_mean_squared_fitness(self, differences):
#         mean_sq = []

#         for i in range(len(differences)):
#             tmp = []
#             for j in range(len(differences[i])):
#                 x = differences[i][j] ** 2
#                 tmp.append(x)

#             mean_sq.append(tmp)
#         mean = []
#         for i in mean_sq:
#             x = np.mean(i)
#             # print(x)
#             mean.append(x)
#         return mean


#     def select_parents(self, population, fitness, num_parents):
#         print("population: ", population)
#         parents = sample(population, num_parents)


#         return parents

#     def split_parents(self,parents):
#     	split_list = [re.findall('\w+|\W', s) for s in parents]

#     	for i in split_list:
#     		i.append("end")
#     	return split_list


# def main2():
# 	print("======================================================")
# 	print("Aim to find a function to map my inputs to my outputs.")
# 	print("inputs: ", inputs)
# 	print("outputs: ", output)
# 	print("======================================================")
# 	test = GenExp(4)
# 	generate_expressions = test.get_valid_expressions(4, 4)  # (maxNumber,Population size)
# 	# print("Population: ", generate_expressions)

# 	eval_exp = test.eval_expressions(generate_expressions)
# 	get_totals = test.get_totals(eval_exp)
# 	# print("totals: ", get_totals)
# 	get_fitness = test.get_mean_squared_fitness(get_totals)
# 	# print("fitness error: ",get_fitness)
# 	# print()
# 	# print("=======================================================")
# 	# print("parents")
# 	select_parents = test.select_parents(generate_expressions, get_fitness, 2)
# 	# print("parents selected: ",select_parents)
# 	split_parents = test.split_parents(select_parents)
# 	print("split parents: ", split_parents)

def main():
	split_parents = [['X1', '-', '1', 'end'], ['X1', '+', '0', 'end']]
	print(split_parents)

	tree = Tree()
	for i in split_parents:
		tree = tree.get_expression(i)


	


if __name__=="__main__":
	main()