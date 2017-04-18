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

  
        


    def conv_inf(self, l):
        for e in l[::-1]:
            if e not in ['+','-','*','/']:
                self.stack.append(e)
                print(self.stack)
            else:
                operand1 = self.stack.pop(-1)
                operand2 = self.stack.pop(-1)
                self.stack.append("({}{}{})".format(operand1, e, operand2))
        return self.stack.pop()[1:-1]


l = ['-', 'X2', '+', 'X5', '-', 'X5', 'X2']
t = ToInfixParser()
print(t.conv_inf(l))


# (X2 - (X5 + (X5 - X2)))