def out_func(func):
    print('----out----')
    func()
    def in_func():
        print('----inner-----')
        func()
    return in_func

@out_func
def func():
    print('----func----')

func()
