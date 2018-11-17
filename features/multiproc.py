from subprocess import Popen
import sys


if __name__ == "__main__":
    script = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])

    for i in range(start, end):
        print(i)
        p = Popen(['python', script, str(i)])
