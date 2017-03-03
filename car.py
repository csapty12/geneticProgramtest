class Node(object):
 
    def __repr__(self):
        if self.parent != None:
            return "Node (" + str(self.value) + ") parent val:" + str(self.parent.value)
        else:
            return "Node (" + str(self.value) + ")"
    def __str__(self, level=0):
        ret = "\t"*level+self.__repr__()+"\n"
        if self.left_child is not None:
            ret += (self.left_child).__str__(level+1)
        if self.right_child is not None:
            ret += (self.right_child).__str__(level+1)
        return ret
 
 
    def __init__(self, value):
        self.value = value
 
        self.left_child = None
        self.right_child = None
        self.parent = None
 
    def add_child(self, value, left=False):
        if left == True:
            new_node = Node(value)
            self.left_child = new_node
            new_node.parent = self
 
        else:
            new_node = Node(value)
            self.right_child = new_node
            new_node.parent = self
 
class Tree(object):
    def __init__(self, root_node):
        self.root = root_node
 
    @classmethod
    def swap_nodes(cls, node_one, node_two):
        node_one_parent = node_one.parent
        node_two_parent = node_two.parent
 
        #figure out if node is left or right child
        if node_one_parent.left_child.value == node_one.value:
            node_one_parent.left_child = node_two
        else:
            #it is right child
            node_one_parent.right_child = node_two
 
        if node_two_parent.left_child.value == node_two.value:
            node_two_parent.left_child = node_one
        else:
            node_two_parent.right_child = node_one
 
        return
 
if __name__ == '__main__':
    root_node_one = Node(1)
 
    root_node_one.add_child(2) #right_child
    root_node_one.add_child(3, True) #left child
 
    root_node_one.left_child.add_child(4)
    root_node_one.left_child.add_child(5, True)
 
    root_node_one.right_child.add_child(6)
    root_node_one.right_child.add_child(7,True)
 
    root_node_one.right_child.left_child.add_child(8)
    root_node_one.right_child.left_child.add_child(9, True)
 
    print(root_node_one)
 
 
    root_node_two = Node('a')
 
    root_node_two.add_child('b') #right_child
    root_node_two.add_child('c', True) #left child
 
    root_node_two.left_child.add_child('d')
    root_node_two.left_child.add_child('e', True)
 
    root_node_two.right_child.add_child('f')
    root_node_two.right_child.add_child('g',True)
 
    root_node_two.left_child.left_child.add_child('h')
    root_node_two.left_child.left_child.add_child('i', True)
 
    print(root_node_two)
 
    print("Swapping:", root_node_one.right_child.left_child.value, "and", root_node_two.left_child.right_child.value)
    Tree.swap_nodes(root_node_one.right_child.left_child, root_node_two.left_child.right_child)
 
    print(root_node_one)
    print(root_node_two)