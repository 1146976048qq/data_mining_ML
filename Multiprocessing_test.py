import multiprocessing as mp

def job(q):
    res = 0
    for i in range(1000):
        res += i + i**2 + i ** 3
    q.put(res)  # 定义一个被多线程调用的函数，q 就像一个队列，用来保存每次函数运行的结果

if __name__ == '__main__':
    q = mp.Queue()

# 定义2个线程函数
p1 = mp.Process(target=job, args=(q,))
p2 = mp.Process(target=job, args=(q,))

# 分别启动、连接两个进程
p1.start()
p2.start()
p1.join()
p2.join()

res1 = q.get()
res2 = q.get()

print("res1 = ", res1)
print("res2 = ", res2)

print("res1 + res2 = ", res1+res2)

### 效率对比
import threading as td
import time
print("multithread\multiprocessing\\normal 效率对比:")

def job(q):
    res = 0
    for i in range(10000000):
        res += i + i ** 2 + i ** 3
    q.put(res)

def multiprocessing():
    q = mp.Queue()
    p1 = mp.Process(target=job, args=(q,))
    p2 = mp.Process(target=job, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print("\nmultiprocessing res1+res2 :", res1+res2)

def multithread():
    q = mp.Queue()
    t1 = td.Thread(target=job, args=(q,))
    t2 = td.Thread(target=job, args=(q,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 = q.get()
    res2 = q.get()
    print("\nmultithread res1+res2 :", res1 + res2)

def normal():
    res = 0
    for _ in range(2):
        for i in range(1000000):
            res += i + i ** 2 + i ** 3
    print("\nnormal res : ", res)

if __name__ == '__main__':
    print("**run-time**"*5)
    st = time.time()
    normal()
    st1 = time.time()
    print("normal_Runtime :", st1-st)

    multiprocessing()
    st2 = time.time()
    print("multiprocessig_Runtime :", st2-st1)

    multithread()
    st3 = time.time()
    print("multithread_Runtime :", st3-st2)


### 进程池
print("进程池 Pool() & map()")

def job(x):
    return x**2

def multiprocessing():
    pool = mp.Pool()
    res = pool.map(job, range(100))
    print(res)

if __name__ == '__main__':
    multiprocessing()

