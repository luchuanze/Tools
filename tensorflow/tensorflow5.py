import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#MNIST 数据集相关的常数
INPUT_NODE = 784        #输入层的节点数
OUTPUT_NODE = 10        #输出层的节点数

# 配置神经网络的参数
LAYER1_NODE = 500      #隐藏层节点数
BATCH_SIZE = 100        #一个训练batch中的训练数据个数

LEARNING_RATE_BASE = 0.8    #基础的学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率

REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000      #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        layer1 = tf.nn.relu( tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1))+avg_class.average(biases1)
        )
        return tf.matmul(layer1, avg_class.average(weights2))+avg_class.average(biases2)

#训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    #生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1)
    )
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    #生成输出层的参数
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1)
    )
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    #计算在当前参数下神经网络前向传播的结果
    Y = inference(x, None, weights1, biases1, weights2, biases2)

    #定义存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False)

    #给定滑动平均衰减率和训练轮数的变量，初始化滑动
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    #在所有代表神经网络参数的变量上使用平均滑动。其他辅助变量（比如global_step）就不需要了。
    # tf.trainable_variables() 返回的就是图上集合

    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())

    #计算使用了滑动平均之后的前向传播结果
    average_Y = inference(x, variable_averages, weights1,biases1, weights2, biases2)

    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=Y, labels=tf.argmax(y_, 1))
    #计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    #使用tf.train.GradientDescentOptimizer 优化算法来优化损失函数。注意这里损失函数
    #包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

    #在训练神经网络模型时，每过一遍数据需要通过方向传播来更新神经网络中的参数
    #又要更新每一个参数的滑动平均值。
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    #检验使用了滑动平均模型的神经网络前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_Y,1), tf.argmax(y_,1))
    #
    #这个运算首先将一个布尔型的数值转为实数型，然后计算平均值。
    #这个平均值就是模型在这个一组数据上的正确率。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        #准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的
        #条件和评判训练结果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        #准备测试数据。在真实的应用中，这部分数据在训练时是不可见的，这个数据只是
        #作为模型优劣的最后评价标准。
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        #迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出一次在验证数据集上的测试结果
            if i %1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accureacy"
                      "using average model is %g " %(i,validate_acc))

                #产生这一轮使用的一个batch 的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        #在训练结束之后，在测试数据上检测神经网络模型的最终正确率。
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average"
              "model is %g" %( TRAINING_STEPS, test_acc))


#主程序入口
def main(argv=None):
    #声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()
