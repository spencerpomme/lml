import threading
import time

def wait_long_func(sec: int):
    time.sleep(sec)
    print("take_long_func: {}s".format(sec))


def cpu_bound_long(round: int):
    res = 0
    for i in range(round):
        res += i
    print("iterations: {}".format(round))


def instant():
    print("instant call.")


if __name__ == "__main__":

    # serial
    start = time.time()
    print("Serial call:")
    cpu_bound_long(500000)
    cpu_bound_long(300000)
    cpu_bound_long(100000)
    instant()
    print("Total time: {}s\n".format(time.time() - start))

    # concurrent
    start2 = time.time()
    print("Concurrent call:")
    t1 = threading.Thread(target=cpu_bound_long, args=(500000,), daemon=False)
    t2 = threading.Thread(target=cpu_bound_long, args=(300000,), daemon=False)
    t3 = threading.Thread(target=cpu_bound_long, args=(100000,), daemon=False)
    t4 = threading.Thread(target=instant)
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    print("Total time: {}s\n".format(time.time() - start2))
    

