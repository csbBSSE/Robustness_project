import multiprocessing
import subprocess
from time import sleep

p = []
for i in range(num_threads):
    p.append(multiprocessing.Process(target = run_racipe, args = (racipe_command,)))
    p[i].start()

process_completion_flag = 0
while process_completion_flag == 0:
    process_completion_flag = 1
    for i in p:
        if i.is_alive():
            process_completion_flag = 0
            break
    sleep(1)
