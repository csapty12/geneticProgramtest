from ExpressionGenerator import GenMember
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

