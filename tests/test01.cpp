/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <stdlib.h>
#include <iostream>

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

#include "cnai_dnn_net.h"

using tensorflow::string;
using tensorflow::int32;
using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

static std::vector< std::pair<std::string, Tensor> > s_trainData;
static std::vector< std::pair<std::string, Tensor> > s_runData;
static std::vector<string> s_outNames;
static std::vector<string> s_weightInNames;
static float __frand(float from, float to);

#define TRAIN_TIMES 300
#define TRAIN_INPUT_NODES 4
#define TRAIN_LAYER1_NODES 16
#define TRAIN_LAYER2_NODES 32
#define TRAIN_LAYER3_NODES 32
#define TRAIN_LAYER4_NODES 0
#define TRAIN_LAYER5_NODES 0
#define TRAIN_OUTPUT_NODES 4

#define TRAIN_FOLDER "../data/graphs"
#define TRAIN_NAME "comp4x4"
#define TRAIN_FILE "../data/graphs/comp4x4"

static int s_weight_from = 0;
static int s_weight_num = 0;

static int s_debug = 1;

static void mockInputData(float *data, float *expect, int max, int emax, int numd, int nume)
{
    data[max] = __frand(0.0f, 1000.0f);

    for(int i = 0; i < nume; i ++)
    {
        expect[i] = 0.0f;
    }
    expect[emax] = 1.0f;

    for(int i = 0; i < numd; i ++)
    {
        if(i != max)
        {
            data[i] = data[max] - 100.0f - __frand(0.0f, 500.0f);
        }
    }

    for(int i = 0; i < numd; i ++)
    {
        printf("%f ", data[i]);
    }
    printf("\n");
    for(int i = 0; i < nume; i ++)
    {
        printf("%f ", expect[i]);
    }
    printf("\n");
}

static void generateTestData(std::vector< std::pair<std::string, Tensor> > &data,
                             float *inputs, int ni,
                             float *expects, int no)
{
    data.clear();


    for(int i = 0; i < ni; i ++)
    {
        Tensor tx(DT_FLOAT, {1, 1});
        auto x = tx.flat<float>();
        x.setConstant(inputs[i]);

        char n[8] = {0};
        sprintf(n, "%d", i);
        data.push_back({__CNAI_INPUT_X(n), tx});
    }

    for(int i = 0; i < no; i ++)
    {
        Tensor tx(DT_FLOAT, {1, 1});
        auto x = tx.flat<float>();
        x.setConstant(expects[i]);

        char n[8] = {0};
        sprintf(n, "%d", i);
        data.push_back({__CNAI_TARGET_Y(n), tx});
    }

    Tensor b(DT_FLOAT, {1, 1});
    auto b_flat = b.flat<float>();
    b_flat.setConstant(0.0f);

    data.push_back({__CNAI_INPUT_B, b});
}

static void generateTrainData(std::vector< std::pair<std::string, Tensor> > &data,
                              float *inputs, int ni,
                              float *expects, int no)
{
    data.clear();

    for(int i = 0; i < ni; i ++)
    {
        Tensor tx(DT_FLOAT, {1, 1});
        auto x = tx.flat<float>();
        x.setConstant(inputs[i]);

        char n[8] = {0};
        sprintf(n, "%d", i);
        data.push_back({__CNAI_INPUT_X(n), tx});
    }

    for(int i = 0; i < no; i ++)
    {
        Tensor tx(DT_FLOAT, {1, 1});
        auto x = tx.flat<float>();
        x.setConstant(expects[i]);

        char n[8] = {0};
        sprintf(n, "%d", i);
        data.push_back({__CNAI_TARGET_Y(n), tx});
    }

    Tensor b(DT_FLOAT, {1, 1});
    auto b_flat = b.flat<float>();
    b_flat.setConstant(0.0f);

    data.push_back({__CNAI_INPUT_B, b});

    Tensor step1(DT_FLOAT, {1, 1});
    auto s_flat1 = step1.flat<float>();
    s_flat1.setConstant(0.01f);

    Tensor step2(DT_FLOAT, {1, 1});
    auto s_flat2 = step2.flat<float>();
    s_flat2.setConstant(0.01f);

    Tensor step3(DT_FLOAT, {1, 1});
    auto s_flat3 = step3.flat<float>();
    s_flat3.setConstant(0.01f);

    Tensor step4(DT_FLOAT, {1, 1});
    auto s_flat4 = step4.flat<float>();
    s_flat4.setConstant(0.001f);

    Tensor step5(DT_FLOAT, {1, 1});
    auto s_flat5 = step5.flat<float>();
    s_flat5.setConstant(0.001f);

    Tensor step6(DT_FLOAT, {1, 1});
    auto s_flat6 = step6.flat<float>();
    s_flat6.setConstant(0.001f);

    data.push_back({__CNAI_TRAIN_STEP + "_1", step1});
    data.push_back({__CNAI_TRAIN_STEP + "_2", step2});
    data.push_back({__CNAI_TRAIN_STEP + "_3", step3});
    data.push_back({__CNAI_TRAIN_STEP + "_4", step4});
}

static float __frand(float from, float to)
{
    float r01 = (rand() % 999999) / 999999.0f;
    r01 = (to - from) * r01;
    r01 = r01 + from;
    return r01;
}

static void printOutputs(std::vector< std::pair<string, Tensor> > outputs)
{
    if(s_debug == 0)
    {
        return;
    }
    std::printf("***************************************\n");
    std::printf("Output DebugPrint: %d\n", outputs.size());
    for(int i = 0; i < outputs.size(); i ++)
    {
        string name = outputs[i].first;
        Tensor &t = outputs[i].second;
        auto f = t.flat<float>();
        std::printf("[%d] %s: %f\n", i, name.c_str(), f(0));
    }
    std::printf("***************************************\n");
}

static void statisticsOutputs(std::vector< std::pair<string, Tensor> > outputs, int largest)
{
    static int correct = 0;
    float max = 0.0f;
    int which = -1;
    for(int i = 0; i < outputs.size(); i ++)
    {
        if(outputs[i].first.find(__CNAI_KEYWORD_OUTPUT) != std::string::npos)
        {
            Tensor t = outputs[i].second;
            auto flat = t.flat<float>();
            if(flat(0) > max)
            {
                which = i;
                max = flat(0);
            }
        }
    }
    if(which == largest)
    {
        correct ++;
    }
    else
    {
        printf("[NORMAL] Failed classification!!!!!!!!!!!!!\n");
    }
    printf("correct: %d\n", correct);
}

static Tensor __genWeight(int layer, int nleft, int nright)
{
    float f = 1.0f / sqrt((float)nleft);

    Tensor t(DT_FLOAT, {1, 1});
    auto f_flat = t.flat<float>();
    f_flat.setConstant(__frand(-f, f));
    return t;
}

static void __train()
{
    CNAIDNNNet net("cnet");

    std::vector< std::pair<string, Tensor> > outputs;

    Scope scope = net.begin();

    /// input layer
    net.addLayer(scope, "_LInput_", CNAI_DNN_TYPE_NORMAL, TRAIN_INPUT_NODES, CN_AT_Non);
    net.addLayer(scope, "_LDnn1_", CNAI_DNN_TYPE_NORMAL, TRAIN_LAYER1_NODES, CN_AT_SIGMOD);
    net.addLayer(scope, "_LDnn2_", CNAI_DNN_TYPE_NORMAL, TRAIN_LAYER2_NODES, CN_AT_SIGMOD);
    net.addLayer(scope, "_LDnn3_", CNAI_DNN_TYPE_NORMAL, TRAIN_LAYER3_NODES, CN_AT_SIGMOD);
    net.addLayer(scope, "_LOut_", CNAI_DNN_TYPE_NORMAL, TRAIN_OUTPUT_NODES, CN_AT_SIGMOD);

    net.end(scope);

    net.saveGraph(TRAIN_FOLDER, TRAIN_NAME);
    net.saveWeightNames(TRAIN_FOLDER, TRAIN_NAME);

    net.addOutputFilter({"_LOut_", __CNAI_KEYWORD_OUTPUT});
    net.initWeights(__genWeight);

    for(int t = 0; t < 500; t ++)
    {
        for(int m = 0; m < TRAIN_INPUT_NODES; m ++)
        {
            for(int i = 0; i < TRAIN_TIMES; i ++)
            {
                printf("[t->m->i: %d->%d->%d]\n", t, m, i);
                float data[TRAIN_INPUT_NODES] = {0.0f};
                float exps[TRAIN_OUTPUT_NODES] = {0.0f};
                mockInputData(data, exps, m, m, TRAIN_INPUT_NODES, TRAIN_OUTPUT_NODES);

                generateTrainData(s_trainData, data, TRAIN_INPUT_NODES, exps, TRAIN_OUTPUT_NODES);
                outputs.clear();

                outputs.clear();
                outputs = net.train(s_trainData);
                printOutputs(outputs);
            }
        }
    }
    net.save(TRAIN_FOLDER, TRAIN_NAME);
}

static void __test()
{
    std::vector< std::pair<string, Tensor> > outputs;

    CNAIDNNNet net("cnet");

    Status ret = net.load(TRAIN_FILE);

    if(ret != Status::OK())
    {
        printf("[ERR] Load graph failed: %s\n", ret.error_message().c_str());
        exit(-1);
    }

    s_outNames.clear();

    net.addOutputFilter({"_LOut_", __CNAI_KEYWORD_OUTPUT});

    for(int i = 0; i < 1000; i ++)
    {
        float data[TRAIN_INPUT_NODES] = {0.0f};
        float exps[TRAIN_OUTPUT_NODES] = {0.0f};

        int m = rand() % TRAIN_INPUT_NODES;

        mockInputData(data, exps,
                      m, m,
                      TRAIN_INPUT_NODES, TRAIN_OUTPUT_NODES);

        generateTrainData(s_runData, data, TRAIN_INPUT_NODES, exps, TRAIN_OUTPUT_NODES);

        outputs.clear();
        outputs = net.run(s_runData);
        printOutputs(outputs);

        statisticsOutputs(outputs, m);
    }
}

int main(int argc, char* argv[])
{
    static int train = 1;
    if(train)
        __train();
    else
        __test();

    return 0;
}
