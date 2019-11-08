def make_chrstr(chrlist):

    cl = chrlist.copy()
    cl.append(0)
    chrstr = ''
    first = cl[0]
    och = first
    for ch in cl:
        if ch == och:
            och += 1
        elif first != och-1:
            if len(chrstr) != 0:
                chrstr += ', '
            chrstr += '%d-%d' % (first, och-1)
            first = ch
            och = ch+1
        else:
            if len(chrstr) != 0:
                chrstr += ', '
            chrstr += '%d' % first
            first = ch
            och = ch+1

    return chrstr


def read_chrstr(chrstr):

    chrstr = chrstr.strip('[]')
    c = chrstr.split(',')
    chrlist = []
    for el in c:
        el = el.split('-')
        if len(el) == 1:
            chrlist.append(int(el[0]))
        else:
            chrlist += [i for i in range(int(el[0]), int(el[1])+1)]
    chrlist.sort()

    return chrlist
