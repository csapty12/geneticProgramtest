from math import log, fabs
from random import randint, random, choice, sample
import re
import numpy as np
import itertools
import collections

OPS = ['+', '-', '*']
GROUP_PROB = 0.3
MIN_NUM, MAX_NUM = 0, 20

X2 = 5
X3 = 8
inputs = [4, 8, 12, 13]
output = [60, 64, 68, 69]
ideal_solution = "x1+8*(4+3)"
randomVariable1 = [randint(MIN_NUM, MAX_NUM), "X1"]


class GeneticProgram:
    """
    Class to generate a population, evaluate it, find the fitness function
    select parents for genetic operations and put children back into population

    """

    def __init__(self, maxNumbers, maxdepth=None, depth=0):
        """
        The constructor is what builds the expressions. The parameters include maxNumbers, which indicates the maximum number of operands
        that an expresssion could have; maxdepth is initially set to None as this is an added feature which will need to be catered for in
        the following functions; depth is initally 0 as the expression initially does not exists, however the depth increases by 1 each time
        as the epxression is slowly built up.
        """
        self.left = None  # create the left and right nodes.
        self.right = None

        if maxdepth is None:
            maxdepth = log(maxNumbers, 2) - 1

        if depth < maxdepth and randint(0, maxdepth) > depth:
            self.left = GeneticProgram(maxNumbers, maxdepth, depth + 1)  # generate part of the expression (on the left)
        else:
            self.left = choice(randomVariable1)

        if depth < maxdepth and randint(0, maxdepth) > depth:
            self.right = GeneticProgram(maxNumbers, maxdepth,
                                        depth + 1)  # generate part of the expression (on the right)
        else:
            self.right = randint(MIN_NUM, MAX_NUM)

        self.grouped = random() < GROUP_PROB  # if true, then bracketing around certain expressions will be allowed
        self.operator = choice(OPS)

    def __str__(self):
        """
        function to print out the expressions when they are generated in a legal, formatted method, including the ability to bracket
        expressions as well to make the expressions generated more random.
        """
        s = '{0!s}{1}{2!s}'.format(self.left, self.operator, self.right)
        if self.grouped:
            return '({0})'.format(s)
        else:
            return s

    def get_valid_expressions(self, maxNumbers, populationSize):
        """
        function to get only valid expressions, using variables included.
        function uses maxNumbers to indicate most number of values in the expression.
        function uses population size to indicate how many valid expression to build.
        """
        expression_list = []
        while len(expression_list) < populationSize:
            exps = GeneticProgram(maxNumbers)
            str_exps = str(exps)
            expression_list.append(str_exps)
            expression_list = [i for i in expression_list if 'X1' in i]
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

    def get_values(self, expression):
        """
        Input:
        expression: what an expression is

        Output:
        totals: this is what totals actually is
        """
        totals = []
        for i in expression:
            temp_totals = []
            for j in i:
                x = eval(j)
                temp_totals.append(x)
            totals.append(temp_totals)
        return totals

    def get_difference(self, totals, outputs=output):
        differences = []
        for i in range(len(totals)):
            tmp = []
            for j in range(len(totals[i])):
                diff = fabs(totals[i][j] - outputs[j])
                tmp.append(diff)
            differences.append(tmp)
        return differences

    def get_mean_squared_fitness(self, differences):
        mean_sq = []

        for i in range(len(differences)):
            # print(differences[i])
            tmp = []
            for j in range(len(differences[i])):
                x = differences[i][j] ** 2
                tmp.append(x)

            mean_sq.append(tmp)
        # print(mean_sq)

        mean = []
        for i in mean_sq:
            x = np.mean(i)
            # print(x)
            mean.append(x)
        return mean

    def select_parents(self, population, fitness, num_parents):
        """
        input: population, solution fitness
        output: two randomly selected parents
        """
        print("population: ", population)
        parents = sample(population, num_parents)

        return parents

    def get_valid_substring(self,parents,eval,):
        """
        function to cross over the parents.
        """
        p = ['(X1*16)+(X1-7)', '(X1-6)+(13+3)', 'X1+12', '13+10+X1+17']
        eval_parents = self.eval_expressions(p)
        get_parent_values = self.get_values(eval_parents)

        for i in p:
        	print("parents:", i)
        split_list = self.strip_list(p)
        print(split_list)
        for i in range(len(split_list)):
            if split_list[i][0]=='(':
                if split_list[i][-1]== ')':
                    split_list[i].pop(0)
                    split_list[i].pop(-1)
                    #evaluate the expression. if the expression remains the same, keep it, otherwise keep the old bracketing 

        new_list = split_list

        string_split =[]
        for i in new_list:
            str1 = ''.join(i)
            string_split.append(str1)
        print(string_split)

        
        # for i in range(len(string_split)):
        	
        # 	eval_expressions = self.eval_expressions(string_split)

        # 	try:
        # 	 	get_values = GeneticProgram.get_values(self,eval_expressions)

        # 	except:
        # 		print("here")
        # 		string_split.insert(0,'(')
        # 		string_split.append(')')
        # print(string_split)



        




        
        	


    def strip_list(self,parents):
    	split_list = [re.findall('\w+|\W', s) for s in parents]
    	return split_list   



    def get_random_substring(self,expression):
        rndLeft = randint(0,len(expression)-2)
        rndRight = randint(rndLeft+2, len(expression))
        rnd= randint(0,1)

        if rnd ==0:
            exp = expression[rndLeft:]
        elif rnd ==1:
            exp = expression[:rndRight]
        return exp

    def crossover(self, parents, sub_string):

        """
        crossover fo parents occrs here
        """
        parent1 = parents[0]
        print("parent1: ",parent1)
        parent2 = parents[1]
        print("parent2: ",parent2)
        sub1 = sub_string[0]
        print("sub1: ",sub1)
        sub2 = sub_string[1]
        print("sub2: ",sub2)

        # for i in parents:
        #     print("parents: ", i)
        # for i in sub_string:
        #     print("substring: ",i)
        pass
    def mutation(self, population):
        """
        function to mutate part of the string
        """
        print(population)


def main():
    print("======================================================")
    print("Aim to find a function to map my inputs to my outputs.")
    print("inputs: ", inputs)
    print("outputs: ", output)
    print("======================================================")
    test = GeneticProgram(4)
    generate_expressions = test.get_valid_expressions(4, 4)  # (maxNumber,Population size)
    print("Population: ", generate_expressions)
    eval_expressions = test.eval_expressions(generate_expressions)
    # print("evaluate population: ",eval_expressions)
    get_values = test.get_values(eval_expressions)
    # print("solution totals: ",get_values)
    differences = test.get_difference(get_values)

    # print("differences: ",differences)
    get_mean_sq_fitness = test.get_mean_squared_fitness(differences)
    print("mean squared fitness: ", get_mean_sq_fitness)
    print()
    print("=======================================================")
    print("parents")
    select_parents = test.select_parents(generate_expressions, get_mean_sq_fitness, 2)
    get_valid_substring = test.get_valid_substring(select_parents,2)
    print(get_valid_substring)
    # crossover = test.crossover(select_parents,get_valid_substring)


if __name__ == '__main__':
    main()


