import tensorflow as tf

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)
    y_train, y_test = tf.one_hot(y_train, 10).numpy(), tf.one_hot(y_test, 10).numpy()
    return x_train, y_train, x_test, y_test

def main():
    train_image,train_labels,test_image,test_labels = load_data()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'] )
    model.fit(train_image, train_labels,batch_size=64, epochs=10)

    loss,acc=model.evaluate(test_image,test_labels)
    print('损失函数：{}，精确率：{:.2f}%'.format(loss,acc*100))

main()
