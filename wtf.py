import tensorflow as tf

def main():
    W = tf.Variable([3,3,3,3])
    W_assign = W[0:2].assign(W[0:2]+[2,2])


    with tf.Session() as sess:
        sess.run(W.initializer)
        sess.run(W_assign)

        print W.eval()
        print W[-2:].eval()
        print W[:2].eval()


if __name__ == '__main__':
    main()