{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "ASLwithCNN.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Arshad221b/Sign-Language-Recognition-/blob/master/ASLwithCNN.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "metadata": {
        "id": "VdNgPyvDWEUS",
        "colab_type": "code",
        "outputId": "32524036-570e-4299-a379-9fbc9594e524",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        }
      },
      "cell_type": "code",
      "source": [
        "import keras\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import cv2\n",
        "from matplotlib import pyplot as plt\n",
        "from keras.models import Sequential \n",
        "from keras.layers import Conv2D,MaxPooling2D, Dense,Flatten, Dropout\n",
        "from keras.datasets import mnist \n",
        "import matplotlib.pyplot as plt\n",
        "from keras.utils import np_utils\n",
        "from keras.optimizers import SGD"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Using TensorFlow backend.\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "metadata": {
        "id": "GlR1JOKvWHsH",
        "colab_type": "code",
        "outputId": "d5fc426a-68f5-4e75-cadc-e355268e7214",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 413
        }
      },
      "cell_type": "code",
      "source": [
        "!pip install PyDrive"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting PyDrive\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/52/e0/0e64788e5dd58ce2d6934549676243dc69d982f198524be9b99e9c2a4fd5/PyDrive-1.3.1.tar.gz (987kB)\n",
            "\u001b[K    100% |████████████████████████████████| 993kB 21.4MB/s \n",
            "\u001b[?25hRequirement already satisfied: google-api-python-client>=1.2 in /usr/local/lib/python3.6/dist-packages (from PyDrive) (1.6.7)\n",
            "Requirement already satisfied: oauth2client>=4.0.0 in /usr/local/lib/python3.6/dist-packages (from PyDrive) (4.1.3)\n",
            "Requirement already satisfied: PyYAML>=3.0 in /usr/local/lib/python3.6/dist-packages (from PyDrive) (3.13)\n",
            "Requirement already satisfied: six<2dev,>=1.6.1 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client>=1.2->PyDrive) (1.11.0)\n",
            "Requirement already satisfied: httplib2<1dev,>=0.9.2 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client>=1.2->PyDrive) (0.11.3)\n",
            "Requirement already satisfied: uritemplate<4dev,>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client>=1.2->PyDrive) (3.0.0)\n",
            "Requirement already satisfied: pyasn1-modules>=0.0.5 in /usr/local/lib/python3.6/dist-packages (from oauth2client>=4.0.0->PyDrive) (0.2.4)\n",
            "Requirement already satisfied: rsa>=3.1.4 in /usr/local/lib/python3.6/dist-packages (from oauth2client>=4.0.0->PyDrive) (4.0)\n",
            "Requirement already satisfied: pyasn1>=0.1.7 in /usr/local/lib/python3.6/dist-packages (from oauth2client>=4.0.0->PyDrive) (0.4.5)\n",
            "Building wheels for collected packages: PyDrive\n",
            "  Building wheel for PyDrive (setup.py) ... \u001b[?25ldone\n",
            "\u001b[?25h  Stored in directory: /root/.cache/pip/wheels/fa/d2/9a/d3b6b506c2da98289e5d417215ce34b696db856643bad779f4\n",
            "Successfully built PyDrive\n",
            "Installing collected packages: PyDrive\n",
            "Successfully installed PyDrive-1.3.1\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "HNnnwPjXWRuf",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "import os\n",
        "from pydrive.auth import GoogleAuth\n",
        "from pydrive.drive import GoogleDrive\n",
        "from google.colab import auth\n",
        "from oauth2client.client import GoogleCredentials"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "9LqcRmtLWbtW",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "auth.authenticate_user()\n",
        "gauth = GoogleAuth()\n",
        "gauth.credentials = GoogleCredentials.get_application_default()\n",
        "drive = GoogleDrive(gauth)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "cbSWnRL-Weng",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "download = drive.CreateFile({'id': '1wG0gS-bqjV6yz1YveuxkvHT5_2DOuT05'})\n",
        "download.GetContentFile('train.csv')\n",
        "train = pd.read_csv('train.csv')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "xLgX8B76Wpz4",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "download = drive.CreateFile({'id': '1q_Zwlu3RncjKq1YpiVtkiMPxIIueGRYB'})\n",
        "download.GetContentFile('test.csv')\n",
        "test = pd.read_csv('test.csv')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "APs14-zkaRwl",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "y_train = train['label'].values\n",
        "y_test = test['label'].values\n",
        "\n",
        "X_train = train.drop(['label'],axis=1)\n",
        "X_test = test.drop(['label'], axis=1)\n",
        "\n",
        "X_train = np.array(X_train.iloc[:,:])\n",
        "X_train = np.array([np.reshape(i, (28,28)) for i in X_train])\n",
        "\n",
        "X_test = np.array(X_test.iloc[:,:])\n",
        "X_test = np.array([np.reshape(i, (28,28)) for i in X_test])\n",
        "\n",
        "num_classes = 26\n",
        "y_train = np.array(y_train).reshape(-1)\n",
        "y_test = np.array(y_test).reshape(-1)\n",
        "\n",
        "y_train = np.eye(num_classes)[y_train]\n",
        "y_test = np.eye(num_classes)[y_test]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "VG7bDv9xkS1h",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "X_train = X_train.reshape((27455, 28, 28, 1))\n",
        "X_test = X_test.reshape((7172, 28, 28, 1))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "pX7X-hS9W1PB",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 179
        },
        "outputId": "94dc0a57-37ee-4cb2-8985-0150678d09f1"
      },
      "cell_type": "code",
      "source": [
        "\n",
        "classifier = Sequential()\n",
        "classifier.add(Conv2D(filters=8, kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(28,28,1),activation='relu', data_format='channels_last'))\n",
        "classifier.add(MaxPooling2D(pool_size=(2,2)))\n",
        "classifier.add(Conv2D(filters=16, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))\n",
        "classifier.add(Dropout(0.5))\n",
        "classifier.add(MaxPooling2D(pool_size=(4,4)))\n",
        "classifier.add(Dense(128, activation='relu'))\n",
        "classifier.add(Flatten())\n",
        "classifier.add(Dense(26, activation='softmax'))\n",
        "classifier.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Colocations handled automatically by placer.\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "Y20PZzLxW-eO",
        "colab_type": "code",
        "outputId": "2062a984-4ec7-4a8f-acc8-344ff47d98c1",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1756
        }
      },
      "cell_type": "code",
      "source": [
        "classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
        "classifier.fit(X_train, y_train, epochs=50, batch_size=100)\n"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Use tf.cast instead.\n",
            "Epoch 1/50\n",
            "27455/27455 [==============================] - 7s 242us/step - loss: 7.6854 - acc: 0.2546\n",
            "Epoch 2/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.6890 - acc: 0.7661\n",
            "Epoch 3/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.3952 - acc: 0.8629\n",
            "Epoch 4/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.2676 - acc: 0.9060\n",
            "Epoch 5/50\n",
            "27455/27455 [==============================] - 2s 88us/step - loss: 0.2072 - acc: 0.9270\n",
            "Epoch 6/50\n",
            "27455/27455 [==============================] - 2s 88us/step - loss: 0.1647 - acc: 0.9430\n",
            "Epoch 7/50\n",
            "27455/27455 [==============================] - 2s 88us/step - loss: 0.1333 - acc: 0.9542\n",
            "Epoch 8/50\n",
            "27455/27455 [==============================] - 2s 89us/step - loss: 0.1289 - acc: 0.9549\n",
            "Epoch 9/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.1016 - acc: 0.9652\n",
            "Epoch 10/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0914 - acc: 0.9681\n",
            "Epoch 11/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0865 - acc: 0.9696\n",
            "Epoch 12/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0849 - acc: 0.9708\n",
            "Epoch 13/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0761 - acc: 0.9746\n",
            "Epoch 14/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0677 - acc: 0.9773\n",
            "Epoch 15/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0706 - acc: 0.9755\n",
            "Epoch 16/50\n",
            "27455/27455 [==============================] - 2s 88us/step - loss: 0.0586 - acc: 0.9803\n",
            "Epoch 17/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0664 - acc: 0.9775\n",
            "Epoch 18/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0519 - acc: 0.9826\n",
            "Epoch 19/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0560 - acc: 0.9811\n",
            "Epoch 20/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0501 - acc: 0.9830\n",
            "Epoch 21/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0476 - acc: 0.9836\n",
            "Epoch 22/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0502 - acc: 0.9828\n",
            "Epoch 23/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0475 - acc: 0.9844\n",
            "Epoch 24/50\n",
            "27455/27455 [==============================] - 2s 88us/step - loss: 0.0453 - acc: 0.9843\n",
            "Epoch 25/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0397 - acc: 0.9870\n",
            "Epoch 26/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0526 - acc: 0.9828\n",
            "Epoch 27/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0405 - acc: 0.9866\n",
            "Epoch 28/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0427 - acc: 0.9855\n",
            "Epoch 29/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0404 - acc: 0.9867\n",
            "Epoch 30/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0445 - acc: 0.9846\n",
            "Epoch 31/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0339 - acc: 0.9891\n",
            "Epoch 32/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0478 - acc: 0.9848\n",
            "Epoch 33/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0309 - acc: 0.9901\n",
            "Epoch 34/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0389 - acc: 0.9877\n",
            "Epoch 35/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0325 - acc: 0.9894\n",
            "Epoch 36/50\n",
            "27455/27455 [==============================] - 2s 88us/step - loss: 0.0367 - acc: 0.9881\n",
            "Epoch 37/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0405 - acc: 0.9879\n",
            "Epoch 38/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0359 - acc: 0.9887\n",
            "Epoch 39/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0333 - acc: 0.9898\n",
            "Epoch 40/50\n",
            "27455/27455 [==============================] - 2s 88us/step - loss: 0.0311 - acc: 0.9901\n",
            "Epoch 41/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0342 - acc: 0.9895\n",
            "Epoch 42/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0258 - acc: 0.9913\n",
            "Epoch 43/50\n",
            "27455/27455 [==============================] - 2s 86us/step - loss: 0.0319 - acc: 0.9886\n",
            "Epoch 44/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0296 - acc: 0.9906\n",
            "Epoch 45/50\n",
            "27455/27455 [==============================] - 2s 88us/step - loss: 0.0294 - acc: 0.9908\n",
            "Epoch 46/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0345 - acc: 0.9895\n",
            "Epoch 47/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0431 - acc: 0.9857\n",
            "Epoch 48/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0326 - acc: 0.9902\n",
            "Epoch 49/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0343 - acc: 0.9890\n",
            "Epoch 50/50\n",
            "27455/27455 [==============================] - 2s 87us/step - loss: 0.0300 - acc: 0.9903\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7feba3527f28>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 10
        }
      ]
    },
    {
      "metadata": {
        "id": "Q9dEumbZckZ1",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 92
        },
        "outputId": "24bd96d4-2b0d-4a35-e5ab-32f445811512"
      },
      "cell_type": "code",
      "source": [
        "accuracy = classifier.evaluate(x=X_test,y=y_test,batch_size=32)\n",
        "print(\"Accuracy: \",accuracy[1])"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "7172/7172 [==============================] - 0s 68us/step\n",
            "Accuracy:  0.9408812046848857\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "Y3ugo3XtmKoZ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 439
        },
        "outputId": "cc60e2c4-8a49-440e-ee02-ddea623bd740"
      },
      "cell_type": "code",
      "source": [
        "classifier.summary()"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "conv2d_1 (Conv2D)            (None, 28, 28, 8)         80        \n",
            "_________________________________________________________________\n",
            "max_pooling2d_1 (MaxPooling2 (None, 14, 14, 8)         0         \n",
            "_________________________________________________________________\n",
            "conv2d_2 (Conv2D)            (None, 14, 14, 16)        1168      \n",
            "_________________________________________________________________\n",
            "dropout_1 (Dropout)          (None, 14, 14, 16)        0         \n",
            "_________________________________________________________________\n",
            "max_pooling2d_2 (MaxPooling2 (None, 3, 3, 16)          0         \n",
            "_________________________________________________________________\n",
            "dense_1 (Dense)              (None, 3, 3, 128)         2176      \n",
            "_________________________________________________________________\n",
            "flatten_1 (Flatten)          (None, 1152)              0         \n",
            "_________________________________________________________________\n",
            "dense_2 (Dense)              (None, 26)                29978     \n",
            "=================================================================\n",
            "Total params: 33,402\n",
            "Trainable params: 33,402\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "-r9GxkiFRyl7",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "from keras.utils.vis_utils import plot_model"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "8UGALrzbR4i2",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "JTikifp6R9h9",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 387
        },
        "outputId": "ac42259d-7b04-47ea-c65e-cea5c32dc888"
      },
      "cell_type": "code",
      "source": [
        "!apt install graphviz\n",
        "!pip install pydot pydot-ng\n",
        "!echo \"Double check with Python 3\"\n",
        "!python -c \"import pydot\""
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Reading package lists... Done\n",
            "Building dependency tree       \n",
            "Reading state information... Done\n",
            "graphviz is already the newest version (2.40.1-2).\n",
            "The following package was automatically installed and is no longer required:\n",
            "  libnvidia-common-410\n",
            "Use 'apt autoremove' to remove it.\n",
            "0 upgraded, 0 newly installed, 0 to remove and 10 not upgraded.\n",
            "Requirement already satisfied: pydot in /usr/local/lib/python3.6/dist-packages (1.3.0)\n",
            "Requirement already satisfied: pydot-ng in /usr/local/lib/python3.6/dist-packages (2.0.0)\n",
            "Requirement already satisfied: pyparsing>=2.1.4 in /usr/local/lib/python3.6/dist-packages (from pydot) (2.3.1)\n",
            "Double check with Python 3\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "T1b-rpZ0SH61",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 529
        },
        "outputId": "20a3ea54-c6ab-4d44-c415-29345642b01c"
      },
      "cell_type": "code",
      "source": [
        "plot_model(classifier, show_shapes=True, show_layer_names=True, to_file='model.png')\n",
        "from IPython.display import Image\n",
        "Image(retina=True, filename='model.png')"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiUAAAOxCAIAAAClj0BdAAAABmJLR0QA/wD/AP+gvaeTAAAgAElE\nQVR4nOzdaVgUV9o38FNNN90szSYKyA7tjkriMgI6BJ1oEsYFFSVRE02MJC4dXIgiyiCihmCE0UAS\n0SEZVxQNOkY0jybooMaYKFHxCSJGxQUB2RoaBJp6P9SVevsB7IVuqmn4/z7MNbXdfVeVqZs6VXUO\nRdM0AQAA6GQ8QycAAAA9AuoNAABwAfUGAAC4gHoDAABc4Bs6ATA+oaGhhk4BDG/FihV+fn6GzgKM\nCe5vQGuZmZkPHz40dBZgSJmZmcXFxYbOAowM7m+gI5YvXz5r1ixDZwEGQ1GUoVMA44P7GwAA4ALq\nDQAAcAH1BgAAuIB6AwAAXEC9AQAALqDeAAAAF1BvAACAC6g3AADABdQbAADgAuoNAABwAfUGAAC4\ngHoDAABcQL0BAAAuoN4AAAAXUG+gE7W0tCQlJfn7+6tYp6GhYeDAgevWrVOemZubGxAQYG5u7uTk\ntHr16ufPnysvbWpq2rx5s0QiMTU1tbGx8fHxuXfvHrMoPj6e+r98fHw0j6xiaUJCwsCBA83MzCws\nLAYOHLh+/fqamhq9LCWE7N+/f9SoUWKx2N3dfcGCBSUlJRrur9rjrHp/We2eBQA9owG0RAjJyMhQ\nu9rt27cDAgIIIcOHD1ex2ooVKwgh0dHR7JybN2+amZmtX7++trb24sWL9vb2CxYsUN4kJCRkwIAB\nP/30U1NT0+PHj6dMmXLjxg1m0caNG1v9Cx8yZIiGkVUvDQ4O3rp1a2lpqUwmO3TokEAgePXVV/Wy\n9ODBg4SQhISEqqqqa9eueXl5+fr6NjU1abK/qo+z2iOp4iyopuG/AQBlqDegNU2uNXl5edOnT9+7\nd6+vr6+KenPhwoWJEye2utLNnj3b09OzpaWFmUxMTKQo6n//93+ZyQMHDlAUdf369XYDbty4cc+e\nPS/6OdWRVS8NCQmpr69nQzGDaj9+/Fj3pUFBQX379mV/9/PPPyeE5ObmarK/qo+z6j1itXsWVEO9\ngQ5Aexp0iuHDhx85cmTOnDlCofBF69TX10dGRiYnJyvPbG5u/u677wIDA9kRJF9//XWapo8dO8ZM\nfvHFFy+//PLQoUO1TUl1ZLW/e/ToUZFIxEZzdnYmhNTW1uq+tLi42MnJif1dV1dXQsj9+/c12V8V\nx1ntHjHaPQsAnQH1BgwmOjp6yZIlvXv3Vp559+7d2tpaNzc3do63tzch5Pr164SQxsbGn376ydfX\ntwM/pzqy6qVtFRYW2tjYuLu7677Uy8urtLSUXco8vPHy8iKdub+sds8CQGdAvQHDuHDhQlFR0Vtv\nvdVqPnO1FYvF7ByRSGRmZvb06VNCyOPHjxsbG3/99degoCAnJyeRSDRo0KCUlBSaptn1o6KibG1t\nTU1NPT09p02bduXKFU0iq17KampqevTo0eeff37mzJkdO3aYmprqvnTt2rUlJSU7duyQyWT5+fnJ\nycmTJk0aM2aMhvv7Iprs0YvOAkBn4Bs6AeiJ6uvrIyIisrKy2i5iXqAyMTFRnikQCOrr68mfbVC9\ne/eOjY0dOHCgiYnJJ598snTpUhsbmzlz5hBC3nnnneDg4H79+pmaml69enXx4sWBgYFXrlwZMmSI\n6siql7JcXV2fPn3aq1evTz/9dPbs2a2S79jSwMDA1atXS6VSqVRKCHFxcdm1axezSO3+qqB2j1Sc\nBYDOgPsbMIC1a9cuWrSIeYzRCvOco7m5WXlmY2OjmZkZIYR5SjFkyBB/f387Oztra+sNGzZYW1vv\n3LmTWdPV1fWll16ytLQ0NTUdM2ZMenp6fX19SkqK2siql7KKi4tLS0v379//zTffvPTSS8rtYB1e\nGh0dvXPnzrNnz9bW1t69e9ff39/Pz6+4uFiT/VVB7R6pOAsAnQH1BriWm5t748aNhQsXtrvU0dGR\nEKL8eYpcLm9oaHByciKEMP9bXl7OLjU1NXV3dy8qKmo32tChQ01MTG7fvq02suqlLIFA0Lt374kT\nJx48eDA/P3/z5s06Ln3y5ElCQsKiRYvGjx9vYWHh6emZlpb2+PHjxMTEDuyv5kdS9VkA6AyoN8C1\n3bt3nz17lsfjMd9jMk+qN23aRFHUL7/84unpKRaL2bezCCF37twhhAwbNowQYmlp2a9fv1u3bikH\nbG5utra2bve3WlpaWlpamLsE1ZFVL21LIpGYmJjk5+fruLSwsFChUPTt25ddamVlZWdnxyzVdn+V\nqd4j1WdBbXCADkC9Aa6lp6crv5JfVlZG/vzyY+TIkXw+/4033jh//nxLSwuzfnZ2NkVRU6ZMYSZn\nz5597dq1u3fvMpNyufz+/fvs68KTJk1S/q0rV67QNO3n50cIUR1Z9dJnz561eqjO1Anm3WVdlrq4\n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\n5OQkEon8/f0vX77MLJVKpaampo6OjszkkiVLLCwsKIoqLy8nhERERKxcubKoqIiiKIlEQgg5deqU\nlZXVpk2bOmnXtm/fTtP0lClT2i6Kj4/v37//rl27zpw50+62NE1v27Zt0KBBQqHQ1tZ22rRpv//+\nO7NI9SEihCgUipiYGDc3NzMzs2HDhnWsq8NHjx6ZmZl5enoyk15eXsqlmnnE4uXl1aUit2VraxsY\nGJicnEz3wJ63DHx/pVdoT4O2SIfa04qLiwkhO3bsYCajo6MJIWfPnq2uri4tLR03bpyFhUVjYyOz\nNDw83MLC4tatWw0NDfn5+cxzZrZxZs6cOQ4ODmzkxMREQkhZWRkzOWPGDG9vb3bpiRMnxGJxXFyc\ntglr2Pbi5eU1ePDgVjO9vb3/+OMPmqYvXrzI4/E8PDxqa2vpNu1pMTExpqame/bsqaqqun79+ssv\nv2xvb19SUsIsVX2IVq1aJRQKMzMzKysr165dy+Pxrly5otUO1tXVicViqVTKzsnJyREIBNu3b6+p\nqbl58+agQYMmTZqkVcxOjfyXv/yl3fY0mqajoqIIIdeuXVMdAe1pAD2Xv7+/lZVV7969w8LC6urq\nHjx4wC7i8/nMH/6DBw9OTU2VyWTp6ekd+Ing4OCampr169frL+v/r66u7o8//vD29n7RCn5+fsuX\nL793796aNWtaLaqvr9+2bdv06dPnzp1rbW09dOjQL7/8sry8fOfOncqrtXuIGhoaUlNTQ0JCZsyY\nYWNjs27dOoFAoO3x2bx5s5OTU3x8PDsnMDBw9erVUqnUysrKx8dHJpPt2rVLq5idHflF+vXrRwi5\nceOGHmMaBdQbAK0xnXU2NTW1u3TkyJHm5uZsW1PXUVpaStO0ubm5inXi4+MHDBiQkpKSm5urPD8/\nP7+2tnbkyJHsnFGjRpmamrIth60oH6KCggK5XO7j48MsMjMzc3R01Or4HD169NChQ6dPnxaLxezM\n6OjonTt3nj17tra29u7du/7+/n5+fsyNaVeIrAJzCp4+faqvgMYC9QZA/4RCYVlZmaGzaK2hoYEQ\n8qLn2AyRSJSenk5R1LvvvltfX8/Or6qqIoRYWloqr2xjYyOTydT+bl1dHSFk3bp11J/u37/PPpxX\n6+DBg5988klOTg4z0h3jyZMnCQkJixYtGj9+vIWFhaenZ1pa2uPHj5nmSoNHVs3MzIz8eTp6FNQb\n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\nCQ4Orq6u5qA/tPr6etV/TXPgk08+OXjw4KFDh8RiMTtz+/btPB4vPDy8urragLm18ttvv61Zs+bD\nDz/09fVVsdrFixdv3rzZRSJ/9dVXffv2jYyMtLa29vX1XbFiRV5e3uXLl5mlH3300fDhw9944w2m\nAvU0qDcA3Nm9e3dpaakBE7hz58769es3bNjQ6hMif3//iIiIR48erVq1ylC5tTV8+PAjR47MmTNH\nKBS+aJ36+vrIyEhtP7zvvMjFxcVOTk7sOLCurq6EkPv377MrxMbG5uXlGXtPAR2DegM9XW5urpub\nG0VRn3/+OSEkNTXVwsLC3Nz82LFjr7/+upWVlYuLy4EDB5iVt2/fLhKJ+vTp88EHHzg5OYlEIn9/\nf/avV6lUampq6ujoyEwuWbLEwsKCoqjy8nJCSERExMqVK4uKiiiKkkgkhJBTp05ZWVlt2rSJs53d\nvn07TdNTpkxpuyg+Pr5///67du06c+ZMu9vSNL1t27ZBgwYJhUJbW9tp06b9/vvvzCLVB40QolAo\nYmJi3NzczMzMhg0bpscOh6Kjo5csWdK7d299BdQxspeXl/KfFMzDGy8vL3aOra1tYGBgcnJyT2vF\nJag3AGPHjr148SI7uXjx4uXLl9fX14vF4oyMjKKiIi8vr/fff7+pqYkQIpVK58+fL5fLP/roo3v3\n7l29erW5ufnVV18tLi4mhGzfvl25g5mUlJQNGzawk8nJyZMnT/b29qZp+s6dO4QQ5tFxS0sLZzv7\n3XffDRgwwNzcvO0iMzOzr7/+msfjvf/++3V1dW1XiI2NjYqKio6OLi0tPX/+fHFx8bhx454+fUrU\nHTRCyJo1az799NOkpKQnT55Mnjz5rbfe+uWXX3TfnQsXLhQVFb311lu6h9JX5LVr15aUlOzYsUMm\nk+Xn5ycnJ0+aNGnMmDHK67z00kuPHj367bff9JevcUC9AWifv7+/lZVV7969w8LC6urqHjx4wC7i\n8/nMn/mDBw9OTU2VyWTp6ekd+Ing4OCampr169frL2tV6urq/vjjD29v7xet4Ofnt3z58nv37q1Z\ns6bVovr6+m3btk2fPn3u3LnW1tZDhw798ssvy8vLd+7cqbxauwetoaEhNTU1JCRkxowZNjY269at\nEwgEHTtirVKKiIhITU3VMY5+IwcGBq5evVoqlVpZWfn4+Mhksl27drVap1+/foSQGzdu6CFXo4J6\n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\nVCqVslisiRMnEpOffPIJj8fDMKylpQUhFBUVtW3bturqagzDxGIxQujbb78VCAQHDhww0K6lp6fj\nOB4YGDh4UUJCwrRp006ePHn16tUht8VxPDU1dcaMGWw2WygULlu27N69e8Qi9YcIIdTf3x8bG+vi\n4mJhYfHKK68QF50jVVlZaW1tPWXKFIq3Va++vt7CwsLV1ZWYdHNzU/4Dgnh44+bmRs4RCoV+fn5p\naWn4uKxIqS2ar69ogsbr9eyYodv9tLq6OoTQ0aNHicndu3cjhL7//vvnz583NTUtXLiQx+P19PQQ\nSyMjI3k83t27d7u7uysqKohnxeQNljVr1jg4OJAtJycnI4Sam5uJyZCQEJFIRC69dOkSn8+Pj4/X\nYU+1OVfd3Nw8PT1VZopEoocPH+I4/sMPP5iZmU2dOrWjowMfdD8tNjaWxWKdPn26vb399u3bc+bM\nsbW1bWxsJJaqP0Tbt29ns9n5+fltbW27du0yMzP76aeftNyvnp6ex48fHz16lM1mnz59WsutRr8t\njuNvvPHG4Ptpyjo7O/l8vlQqJeeUlJQwmcz09HSZTHbnzp0ZM2a8/fbbKlvFxMQghEpLSzUGMG5/\nf+D6Box3EolEIBDY2dmFh4d3dnbW1taSixgMBvGHv6enZ2Zmplwuz8rK0uEjAgICZDLZ3r179Rf1\nv3R2dj58+FAkEg23gq+v75YtW2pqanbu3KmySKFQpKamLl++fO3atVZWVl5eXseOHWtpaTl+/Ljy\nakMeou7u7szMzODg4JCQEGtr6z179jCZTO2Pz+TJk52dnePi4g4fPhwWFjaiXR7NttpITEx0dHRM\nSEgg5/j5+UVHR0ulUoFAMGvWLLlcfvLkSZWt3N3dEULl5eV6j2fMgHwDwP/HYrEQQr29vUMu9fHx\n4XK55L0m49HU1ITjOJfLVbNOQkLC9OnTMzIyrl+/rjy/oqKio6PDx8eHnDN37lwWi0XeOVShfIju\n37/f1dU1a9YsYpGFhcXEiRO1Pz51dXVNTU1fffXVl19+OXv27BE97hrNthoVFBTk5eVduXKFz+eT\nM3fv3n38+PHvv/++o6Pj119/lUgkvr6+xOUyifgKiLFfwZAg3wCgLTab3dzcTHcUqrq7uxFCQz4D\nJ3E4nKysLAzD1q9fr1AoyPnt7e0IIUtLS+WVra2t5XK5xs/t7OxECO3Zswd76dGjR+QDdo2YTKad\nnZ2/v39OTk5FRUViYqKWG45yW/VycnIOHTpUUlIydepUcuaTJ0+SkpI++uijN998k8fjubq6njhx\noqGhgbiJSrKwsEAvvw4wJMg3AGilt7e3vb3dCMdkJH7mNL5v6Ovru3Xr1srKyv3795Mzra2tEUIq\n2UXL3bSzs0MIHTlyRPkG/Y0bN0Yav1gsNjc3r6ioGOmGo9x2sKNHj2ZnZxcXF0+aNEl5fmVlZX9/\nv/JMgUBgY2Oj8rk9PT3o5dcBhgT5BgCtlJSU4Dg+b948YpLBYAx3541i9vb2GIY9f/5c45r79+/3\n8PAoLS0l58yaNcvS0vLnn38m59y8ebOnp+e1117T2NrkyZM5HE5ZWdmIom1tbV29erXyHOLXnOhh\nbLht1cNxPDo6ury8vLCwUOVqDyFEZN8nT56Qc+Ry+bNnz1Q+l/gKHBwcRhnMGAb5BoBhDQwMtLW1\n9fX13b59OyoqysXFJSIiglgkFoufPXtWWFjY29vb3Nys/CoGQsjGxqahoaGmpkYul/f29hYVFRmu\nPzSXy3Vzc3v8+LHGNYm7aspvmXA4nG3bthUUFGRnZ8tksvLy8o0bNzo6OkZGRmrT2vvvv3/27NnM\nzEyZTNbf3//48WPiRzk8PNzBwWHIejk8Hu+7774rLi6WyWS9vb2lpaXvvfcej8fbunUrsYKBtlXv\n7t27hw8fPnHiBJPJxJSkpKQghFxdXRcvXnzixIm//e1vCoWirq6OOD5//OMflRshvgIvL6+Rfvo4\nQkuvONqh8dofcczQoT/00aNHiTdmuFxuYGBgRkYG8YDX3d29urr6+PHjAoEAITRlypQHDx7gOB4Z\nGclkMp2cnBgMhkAgWLZsWXV1Ndlaa2vr4sWLORyOq6vrp59+umPHDoSQWCwmOkzfunVrypQpFhYW\nCxYsaGxsvHz5Mp/PT0hI0GFPtTlXpVIpk8ns6uoiJgsKCojuara2tps2bVJZeceOHcr9oQcGBpKT\nk93d3ZlMplAoDA4Ovn//PrFI4yF68eJFdHS0i4sLg8Gws7MLCQmpqKjAcTw4OBghFBsbO2S0gYGB\nrq6ulpaWbDZbJBKFh4eXl5eTSw237Y0bN+bPn+/o6Ej89E2cOFEikVy7dg3H8eE6lSUnJxPbtrS0\nREVFicViNpttaWk5f/78v/zlLyrtBwQEODk5kTUI1Bi3vz+Qb4BJoqCeTWRkpI2NjUE/QhvanKuV\n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\nTkSuQmCC\n",
            "text/plain": [
              "<IPython.core.display.Image object>"
            ]
          },
          "metadata": {
            "tags": [],
            "image/png": {
              "width": 274,
              "height": 472
            }
          },
          "execution_count": 16
        }
      ]
    },
    {
      "metadata": {
        "id": "yRpbDys4TH5U",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "outputId": "1bdc91e3-b643-45d3-b554-66f1feea7bd3"
      },
      "cell_type": "code",
      "source": [
        "classifier.save('CNNmodel.h5')\n",
        "weights_file = drive.CreateFile({'title' : 'CNNmodel.h5'})\n",
        "weights_file.SetContentFile('CNNmodel.h5')\n",
        "weights_file.Upload()\n",
        "drive.CreateFile({'id': weights_file.get('id')})"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "GoogleDriveFile({'id': '1cCWUVBKuM069e8UGzh2y95aPIOguAVee'})"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 17
        }
      ]
    },
    {
      "metadata": {
        "id": "XeEDOSACyGLG",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}