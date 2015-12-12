def test(a=None, b=None, c=None):
    if a is not None:
        print 'a is ', a
    if b is not None:
        print 'b is ', b
    if c is not None:
        print 'c is ', c


args = {'b': 3, 'c': 5}

test(**args)
