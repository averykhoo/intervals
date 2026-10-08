# two processes hammer one DirectoryBasedExampleDatabase with save/fetch/delete of overlapping values
import os, sys, random
from hypothesis.database import DirectoryBasedExampleDatabase
db = DirectoryBasedExampleDatabase(sys.argv[1])
rnd = random.Random(int(sys.argv[2]))
errors = 0
keys = [b'k%d' % i for i in range(4)]
vals = [bytes([i]) * 20 for i in range(30)]
for n in range(4000):
    k, v = rnd.choice(keys), rnd.choice(vals)
    op = rnd.random()
    try:
        if op < 0.5: db.save(k, v)
        elif op < 0.8: list(db.fetch(k))
        else: db.delete(k, v)
    except Exception as e:
        errors += 1
        print(type(e).__name__, e)
print('pid', os.getpid(), 'errors', errors)
