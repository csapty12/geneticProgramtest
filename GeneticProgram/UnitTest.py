import unittest
from convertToPrefix import ToPrefixParser
from ExpressionGenerator import GenMember
from convertToInfix import ToInfixParser


class ExTest(unittest.TestCase):
    # -------------------------------------- Simple population ----------------------------------------
    population1 = ['X2*X4', 'X5-X4']
    fitness1 = [257, 171]
    # --------------------------------------------------------------------------------------------------
    # -------------------------------------- more complex population -----------------------------------
    population2 = ['X4+X4/X5*X3', '(26.577531100079906-X1+X2-X5)', 'X5/X2/(0.7427951033406766+X2)',
                   '9.698163263654191-X5-X1-X3']
    fitness2 = [559, 296, 559, 298]

    # --------------------------------------------------------------------------------------------------

    def test_select_best(self):
        """
        Test case to check that the two best parents are always being selected from a sample population
        """
        current_population = GenMember()
        self.assertEqual(current_population.select_best_parents(ExTest.population1, ExTest.fitness1),
                         [('X5-X4', 171), ('X2*X4', 257)])
        self.assertEqual(current_population.select_best_parents(ExTest.population2, ExTest.fitness2),
                         [('(26.577531100079906-X1+X2-X5)', 296), ('9.698163263654191-X5-X1-X3', 298)])

    def test_split_parents(self):
        """
        Test case to check that the parents selected have been split up into the appropriate split list
        :return:
        """
        to_pref = ToPrefixParser()
        self.assertEqual(to_pref.split_parents(
            [('X1-X5', 257), ('X5-X4', 171)]), [(['X1', '-', 'X5', 'stop'], 257), (['X5', '-', 'X4', 'stop'], 171)])

        self.assertEqual(
            to_pref.split_parents([('(26.577531100079906-X1+X2-X5)', 296), ('9.698163263654191-X5-X1-X3', 298)]),
            [(['(', '26.577531100079906', '-', 'X1', '+', 'X2', '-', 'X5', ')', 'stop'], 296),
             (['9.698163263654191', '-', 'X5', '-', 'X1', '-', 'X3', 'stop'], 298)])

    def test_get_prefix_notation(self):
        """
        Test case to check that an expression passed in would be converted to the correct prefix notation
        :return:
        """
        to_pref = ToPrefixParser()
        self.assertEqual(
            to_pref.get_prefix_notation([(['X1', '-', 'X5', 'stop'], 257), (['X5', '-', 'X4', 'stop'], 171)]),
            [(['-', 'X1', 'X5'], 257), (['-', 'X5', 'X4'], 171)])

        self.assertEqual(to_pref.get_prefix_notation(
            [(['(', '26.577531100079906', '-', 'X1', '+', 'X2', '-', 'X5', ')', 'stop'], 296),
             (['9.698163263654191', '-', 'X5', '-', 'X1', '-', 'X3', 'stop'], 298)]),
            [(['-', '26.577531100079906', '+', 'X1', '-', 'X2', 'X5'], 296),
             (['-', '9.698163263654191', '-', 'X5', '-', 'X1', 'X3'], 298)])

    def test_get_infix_notation(self):
        """
        Test case to check that a prefix expression is correctly converted back into infix notation
        """
        to_inf = ToInfixParser()
        self.assertEqual(to_inf.get_infix_notation(['-', 'X3', 'X4']), 'X3-X4')
        self.assertEqual(to_inf.get_infix_notation(['*', 'X2', 'X1']), 'X2*X1')
        self.assertEqual(to_inf.get_infix_notation(['-', '26.577531100079906', '+', 'X1', '-', 'X2', 'X5']),
                         '26.577531100079906-(X1+(X2-X5))')
        self.assertEqual(to_inf.get_infix_notation(['-', '9.698163263654191', '-', 'X5', '-', 'X1', 'X3']),
                         '9.698163263654191-(X5-(X1-X3))')

    def test_update_population(self):
        """
        Test case to check that the child solution will replace the worst member in the popualtion if it has a better
        fitness value than the worst current member of the population
        """
        population = GenMember()
        child1 = ['X2-X4*X1/X3']
        child1_fitness = [250]
        child2 = ['X1+0.4555/2+X2']
        child2_fitness = [190]
        update_population1 = population.update_population(ExTest.population1, ExTest.fitness1,
                                                          child1, child1_fitness, child2, child2_fitness)
        print('updated population')
        print(update_population1)


if __name__ == '__main__':
    unittest.main()
