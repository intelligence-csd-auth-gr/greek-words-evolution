import pexpect

class NearestNeighbours:
    """Class for using the command-line interface to fasttext nn to lookup neighbours.
    It's rather fiddly and depends on exact text strings. But it is at least short and simple."""
    def __init__(self, fasttextPath, modelPath, neighborsCount):
        self.nn_process = pexpect.spawn('%s nn %s %d' % (fasttextPath, modelPath, neighborsCount))
        self.nn_process.expect('Query word?')  # Flush the first prompt out.

    def getNeighboursForWord(self, word):
        self.nn_process.sendline(word)
        self.nn_process.expect('Query word?')
        output = self.nn_process.before
        return [word] + [line.strip().split()[0] for line in output.decode('utf8').strip().split('\n')[1:]]