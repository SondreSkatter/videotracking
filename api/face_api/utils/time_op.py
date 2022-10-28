from face_api import app
import time

def time_op(name, f):
    start_time = time.time()
    x = f()
    end_time = time.time()
    app.logger.info('$$ %s -- %s' % (name, end_time - start_time))
    return x