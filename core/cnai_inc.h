#ifndef CNAI_INC_H
#define CNAI_INC_H

#include <string>
#include <vector>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using tensorflow::string;
using tensorflow::int32;
using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

#ifndef __CN_INNER
#define __CN_INNER std::string("ir_")
#endif

#define __ACT_CB Output (*activation)(Scope &scope, Output sigmaWeights)
#define __DER_CB Output (*derivative)(Scope &scope, Output act)


template <class T>
struct CNAIList {
    std::vector<T> v;
    CNAIList(std::initializer_list<T> l) : v(l) {
         std::cout << "constructed with a " << l.size() << "-element list\n";
    }
    void append(std::initializer_list<T> l) {
        v.insert(v.end(), l.begin(), l.end());
    }
    std::pair<const T*, std::size_t> c_arr() const {
        return {&v[0], v.size()};  // copy list-initialization in return statement
                                   // this is NOT a use of std::initializer_list
    }
};

/// target values (TAGS)
#define __CNAI_TARGET_Y(i) std::string("cn_target_y_") + i

/// intput values(TRAIN DATA INPUT)
#define __CNAI_INPUT_X(i) std::string("cn_input_x_") + i

#define __CNAI_TRAIN_STEP std::string("cn_train_step")
#define __CNAI_WEIGHT_FILE std::string("cn_train_weight_file")

#define __CNAI_WF_DEFAULT std::string("/tmp/tensorflow/")

#define __CNAI_WINPUT std::string("WI")
#define __CNAI_WOUTPUT std::string("WO")

#define __CNAI_KEYWORD_INPUT std::string("_i_")
#define __CNAI_KEYWORD_OUTPUT std::string("_o_")
#define __CNAI_KEYWORD_DERIVATIVE std::string("_derivative_")

#endif // CNAI_INC_H
