
class ParameterError(Exception):

    def __init__(self, param, wrong, good):
        self.param = param
        self.wrong = wrong
        self.good = good

    def __str__(self):
        return repr("Value {} for parameter {} is wrong! It should be {}.".format(self.wrong, self.param, self.good))


class DataTypeError(Exception):

    def __init__(self, dirs, wrong):
        self.dirs = dirs
        self.wrong = wrong

    def __str__(self):
        return repr('Data from {} have inconsistent types: {}.'.format(self.dirs, self.wrong))


class RepeatedFileError(Exception):

    def __init__(self, name, prevdir, currdir):
        self.name = name
        self.prevdir = prevdir
        self.currdir = currdir

    def __str__(self):
        return repr('Name {} is repeated in {} and {}'.format(self.name, self.prevdir, self.currdir))
