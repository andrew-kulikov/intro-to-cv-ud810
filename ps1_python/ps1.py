from ps1_python import *
import importlib

if __name__ == '__main__':
    #importlib.reload(ps0_1)
    task = -1
    while True:
        task = int(input())
        if task == 0:
            break
        ps0_1.run()
        