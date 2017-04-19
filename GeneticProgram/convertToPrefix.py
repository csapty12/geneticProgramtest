import re
class ToPrefixParser(object):
    """
    Class that converts infix notation to prefix notation, to get ready to construct a binary tree.
    """

    # every instance of a tree is a node.
    def __init__(self, val=None, left=None, right=None):

        self.val = val  # holds the value
        self.left = left  # holds the left child value
        self.right = right  # holds the right child value

    def __str__(self):
        return str(self.val)  # print out value of node

    def split_parents(self, parents):
        """
        function to split the parents to enable parents to be converted into prefix notation later.
        :param parents: the two parents selected from selection process
        :return: parents, split up into individual gene characteristics -> ["x1+1"] -> ["X1","+","1","end"]
        """
        split_list = [re.findall('\w+|\W', s[0]) for s in parents]

        [i.append("stop") for i in split_list]

        split_parents = [(split_list[i], parents[i][1]) for i in range(len(parents))]

        return split_parents

    def fix_dec(self, split_parents):
        """
        function to repair the split parents if expression contains floating point numbers.
        e.g ["1",".","234","+","X4"] ->["1.234","+","X4"]
        :param split_parents: the parents that have now been split up
        :return: the split parents with all correct values.
        """
        for p in split_parents:
            for item in p[0]:
                if item == ".":
                    dec = p[0].index(item)

                    val1 = p[0][dec - 1]
                    val2 = p[0][dec + 1]
                    x = val1 + item + val2

                    p[0].insert(dec, x)
                    del p[0][dec - 1]
                    del p[0][dec]
                    del p[0][dec]
        return split_parents

    def get_operation(self, expression, expected):
        """
        Function to compare the item in the expression list is the expected item.
        If the string values match, then pop it from the token list.
        :param expression: the expression list
        :param expected: the expected value of the list index
        :return: boolean
        """

        if expression[0] == expected:
            expression.pop(0)
            return True
        else:
            return False

    def is_number(self, expression):
        """
        Function that checks to see whether or not the value to be checked is a number or not.
        If the next value is a number, then return the value itself. Since it is a number, it will not have a left
        or right child as this is a leaf value. This function also handles parentheses to ensure that sub-expressions
        are handled.
        :param expression: the expression
        :return: a numerical value or None
        """
        if self.get_operation(expression, '('):
            x = self.get_expression(expression)  # get the subexpression
            self.get_operation(expression, ')')  # remove the closing parenthesis
            return x
        else:
            x = expression[0]
            if not isinstance(x, str):
                return None
            expression[0:1] = []
            return ToPrefixParser(val=x)

    def get_product(self, expression):
        """
        Function to put the * and / operator into the appropraite place when converting to prefix notation.
        * and / have a higher precedence than + and -, therefore these should be handled first.
        :param expression: expression being passed through
        :return: prefix notation of expression containing * and / in the right places.
        """
        a = self.is_number(expression)

        if self.get_operation(expression, '*'):
            b = self.get_product(expression)
            return ToPrefixParser('*', a, b)
        elif self.get_operation(expression, '/'):
            b = self.get_product(expression)
            return ToPrefixParser("/", a, b)
        else:
            return a

    def get_expression(self, expression):
        """
        Function to handle the - and + operators. get_sum tries to build a tree with a product on the left and a sum on
        the right. But if it doesn’t find a +, it just builds a product.
        :param expression: expression being passed in
        :return: the product or - or + in the correct places in prefix notation
        """
        op1 = self.get_product(expression)

        if self.get_operation(expression, '-'):
            op2 = self.get_expression(expression)
            return ToPrefixParser('-', op1, op2)
        elif self.get_operation(expression, '+'):
            op2 = self.get_expression(expression)
            return ToPrefixParser('+', op1, op2)
        else:
            return op1

    def print_tree_prefix(self, tree):
        """
        Function that takes in the tree, and prints out the tree in the correct prefix notation with 'stop' at the
        end of the prefix notation list -> ['*','3','4','stop']
        :param tree: the prefix notation list

        :return: the tree in appropraite positions.
        """
        if tree.left is None and tree.right is None:
            return tree.val
        else:
            left = self.print_tree_prefix(tree.left)
            right = self.print_tree_prefix(tree.right)
            return tree.val + " " + left + ' ' + right + ''

    def get_prefix_notation(self, parent_expression):
        """
        Function to take the parents expressions from infix notation and convert them to prefix notation.
        :param parent_expression: the parent expression in infix notation
        :return: parents in infix notation.
        """
        prefix = list()
        prefix_list = list()
        pref_list = list()
        for i in parent_expression:
            tree = self.get_expression(i[0])
            y = self.print_tree_prefix(tree)
            prefix.append(y)
        for j in prefix:
            prefix_list.append(j.split())

        for k in range(len(prefix_list)):
            pref_list.append((prefix_list[k], parent_expression[k][1]))
        return pref_list

