from train import emotion_analysis, reshape_dataset
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from model import build_model

if __name__ == '__main__':
    num_classes = 7

    # x_train, y_train, x_test, y_test = reshape_dataset(path, num_classes)

    model = build_model(num_classes)
    model.load_weights('C:/Users/54532/Desktop/facial_expression_detection/weights.h5')

    img = image.load_img("C:/Users/54532/Desktop/facial_expression_detection/dataset/9.jpg", grayscale=True, target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x /= 255

    custom = model.predict(x)
    t1 = emotion_analysis(custom[0])

    x = np.array(x, 'float32')
    x = x.reshape([48, 48])
    plt.gray()

    plt.imshow(x)
    plt.show()
