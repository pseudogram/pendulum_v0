import tensorflow as tf

# def main():
#     W = tf.Variable([3,3,3,3])
#     W_assign = W[0:2].assign(W[0:2]+[2,2])
#
#
#     with tf.Session() as sess:
#         sess.run(W.initializer)
#         sess.run(W_assign)
#
#         print W.eval()
#         print W[-2:].eval()
#         print W[:2].eval()


if __name__ == '__main__':
    # main()

    x = [1, 2, 3]
    y = [4, 5, 6]
    z = list(range(100))
    print(z[1::2])
    print(z[::3])
    print(z[::4])
    print(z[::5])
    zipped = zip(x, y)
    list(zipped)
    print(zipped)
    x2, y2 = zip(*zip(x, y))
    print(x == list(x2) and y == list(y2))
    print(x2)
    print(y2)