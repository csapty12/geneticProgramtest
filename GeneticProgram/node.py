
class Node(object):
    """
    Class that creates a Node object which will either contain a functional operator e.g. +,-,*,/ or will hold a
    terminal value from the terminal set.
    """
    nodeid = 0

    def __repr__(self):
        """
        function to give a visual presentation of the tree.
        :return: node object and the parent that it is associated with.
        """
        if self.parent is not None:
            return "Node (" + str(self.value) + ") parent val:" + str(self.parent.value)
        else:
            return "Node (" + str(self.value) + ")"

    def __str__(self, level=0):
        """
        Function to print out a node at the current level within the tree.
        :param level: determines how many indents to put the each nod at.
        """
        ret = "\t" * level + self.__repr__() + "\n"
        if self.left_child is not None:
            ret += self.left_child.__str__(level + 1)
        if self.right_child is not None:
            ret += self.right_child.__str__(level + 1)
        return ret

    def __init__(self, value=None):
        """
        Constructor to initialise the nodes. Each node has an ID associated with it, a value, and the possibility of
        having two children.
        :param value:
        """
        Node.nodeid += 1
        self.nodenum = Node.nodeid
        self.value = value
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.checked = False
        self.checkedAgain = False

    def add_child(self, value, left=True):
        """
        Function to add a child into the tree. Start by adding children to the left branch of the parent, then add
        the next child to the right branch of the tree.
        :param value:
        :param left:
        :return:
        """
        if left is True:
            new_node = Node(value)
            self.left_child = new_node
            new_node.parent = self

        elif left is False:
            new_node = Node(value)
            self.right_child = new_node
            new_node.parent = self

