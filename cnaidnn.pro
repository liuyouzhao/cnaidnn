QT -= core
QT -= gui

CONFIG += c++11

TARGET = cnaidnn
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES +=  \
    core/sgd/cnai_dnn_node.cpp \
    core/sgd/cnai_dnn_layer.cpp \
    core/sgd/cnai_dnn_net.cpp \
    core/minibgd/cnai_dnn_net_ext.cpp \
    core/cnai_dnn.cpp \
    core/cnai_dnn_weight.cpp \
    tests/test01.cpp \
    core/minibgd/cnai_dnn_node_ext.cpp \
    core/minibgd/cnai_dnn_layer_ext.cpp

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
#DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += ./thirdparty/tf-out
INCLUDEPATH += ./thirdparty/tf-out/bazel-out/host/genfiles
INCLUDEPATH += ./thirdparty/tf-ext/eigen_archive
INCLUDEPATH += ./thirdparty/tf-ext/com_google_absl
INCLUDEPATH += ./thirdparty/tf-ext/protobuf_archive
INCLUDEPATH += ./core
INCLUDEPATH += ./core/sgd

QMAKE_CXXFLAGS += -std=c++11

# Your cuda
LIBS += -L/usr/local/cuda/lib64

# gnu path
LIBS += -L/usr/lib/x86_64-linux-gnu

# tensorflow_cc.so path
LIBS += -L$$PWD/thirdparty/tf-bin

LIBS += -ldl -lm -lpthread -lstdc++
LIBS += -ltensorflow_cc -ltensorflow_framework
LIBS += -lcublas -lcufft -lcusolver -lcudart -lcurand

HEADERS += \
    core/cnai_inc.h \
    core/cnai_dnn.h \
    core/cnai_dnn_weight.h \
    core/sgd/cnai_dnn_node.h \
    core/sgd/cnai_dnn_layer.h \
    core/sgd/cnai_dnn_net.h \
    core/minibgd/cnai_dnn_net_ext.h \
    core/minibgd/cnai_dnn_node_ext.h \
    core/minibgd/cnai_dnn_layer_ext.h

