from tkinter import *

root = Tk()
root.title("Genetic Program To Predict Company Failure. WILL YOU FAIL?!")

# Add a grid
mainframe = Frame(root)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
mainframe.pack()

# Create a Tkinter variable
tkvar = StringVar(root)

# Dictionary with options
choices = {'Tournament Selection', 'Select Best'}
tkvar.set('Please Select')  # set the default option

popupMenu = OptionMenu(mainframe, tkvar, *choices)
Label(mainframe, text="Genetic Program To Predict Company Failure").grid(columnspan=2, row=0, sticky=N)
Label(mainframe, text="").grid(row=1, sticky=N)
Label(mainframe, text="(Current Assets - Current Liabilities)/ Total Assets:").grid(row=2, sticky=E)
Label(mainframe, text="Retained Earnings / Total Assets:").grid(row=3, sticky=E)
Label(mainframe, text="Earnings Before Interest & Tax / Total Assets").grid(row=4, sticky=E)
Label(mainframe, text="Market Value of Equity / Total Debt").grid(row=5, sticky=E)
Label(mainframe, text="Sales / Total Assets").grid(row=6, sticky=E)
e1 = Entry(mainframe)
e2 = Entry(mainframe)
e3 = Entry(mainframe)
e4 = Entry(mainframe)
e5 = Entry(mainframe)
e1.grid(row=2, column=1)
e2.grid(row=3, column=1)
e3.grid(row=4, column=1)
e4.grid(row=5, column=1)
e5.grid(row=6, column=1)

Label(mainframe, text="Selection Type:").grid(row=7, sticky=E)
popupMenu.grid(row=7, column=1)
Button(mainframe, text='Quit', command=mainframe.quit).grid(row=8, column=0, sticky=W, pady=4)
Button(mainframe, text='Train GP').grid(row=8, column=1, sticky=E)


# on change dropdown value
def change_dropdown(*args):
    print(tkvar.get())


# link function to change dropdown
tkvar.trace('w', change_dropdown)

root.mainloop()
