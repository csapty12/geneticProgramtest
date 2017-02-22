import sys
class Tree(object):

	#every instance of a tree is a node.  
	def __init__(self,val, left = None, right = None):
		self.val = val
		self.left= left
		self.right = right

	def __str__(self):

		return str(self.val) #print out value of node 


#print out the expression in prefix notation -> express 1+2*3
def print_tree_prefix(tree):
	if tree ==None:
		return
	else:
		sys.stdout.write("%s " %(tree.val))
		print_tree_prefix(tree.left)
		print_tree_prefix(tree.right)

def print_tree_postfix(tree):
	if tree ==None:
		return
	else:
		print_tree_postfix(tree.left)
		print_tree_postfix(tree.right)
		sys.stdout.write("%s " %(tree.val))

def print_tree_inorder(tree):
	if tree ==None:
		return
	else:
		print_tree_inorder(tree.left)
		sys.stdout.write("%s " %(tree.val))
		print_tree_inorder(tree.right)

def show_tree(tree, level = 0):
	if tree == None: 
		return 
	else:
		show_tree(tree.right,level+1)
		print('  '*level + str(tree.val))
		show_tree(tree.left,level+1)

print("prefix notation")
print_tree_prefix(exp_tree)
print()
print()
print("postfix notation")
print_tree_postfix(exp_tree)
print()
print()
print("in order notation")
print_tree_inorder(exp_tree)
print()
print()
print("tree")
show_tree(exp_tree)
print()
print()
my_list = ['(','3', '+','7',')','*','9']
new_list= []

for i in my_list:
	try:
		new_list.append(int(i))
	except:
		new_list.append(i)


new_list.append('end')
print(new_list)


def get_operation(token_list,expected):
	"""
	compares the expected token to the first token on the list. if they match, remove it, return True
	this is to get the operator
	"""
	if token_list[0] ==expected:
		del  token_list[0]
		return True
	else:
		return False

def get_number(token_list):
    x = token_list[0]
    if type(x) != type("o"):
    	return None
    else:
	    del token_list[0]
	    return Tree (x, None, None)

# def get_string(token_list):


def get_product(token_list):
    a = get_number(token_list)
    if get_operation(token_list, '*'):
        b = get_product(token_list)
        return Tree ('*', a, b)
    else:
        return a


def get_expression(token_list):
    a = get_product(token_list)
    if get_operation(token_list, '-'):
        b = get_expression(token_list)
        return Tree ('-', a, b)
    elif get_operation(token_list, '+'):
    	b = get_expression(token_list)
    	return Tree ('+', a, b)

    else:
        return a


token_list = ['X1', '*', '11', '-', '5', '+', '7', 'end']
tree = get_expression(token_list)
print_tree_prefix(tree)
print()
print()