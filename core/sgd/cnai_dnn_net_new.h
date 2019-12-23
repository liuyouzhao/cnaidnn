#ifndef CNAI_DNN_NET_H
#define CNAI_DNN_NET_H

#include "cnai_inc.h"
#include "cnai_dnn_layer.h"

/// target values (TAGS)
#define __CNAI_TARGET_Y(i) std::string("cn_target_y_") + i

/// intput values(TRAIN DATA INPUT)
#define __CNAI_INPUT_X(i) std::string("cn_input_x_") + i

#define __CNAI_TRAIN_STEP std::string("cn_train_step")

class CNAIDNNNetNew
{
public:
    CNAIDNNNetNew(std::string net);

    void addLayer(Scope &scope, string name, CNAIDNN_LAYERTYPE type, int nodes, CNAIDNN_ACT_TYPE act = CN_AT_Softplus);
    Scope &initFrontGraph();
    Scope &initBackGraph();

    /**
     * @brief trainFront
     * @param outputs
     */
    void trainFront();
    void trainBack();

    void initFFInputs(std::vector<Tensor> &inputs);
    void initFFWeights(std::vector<Tensor> &weights);

    void setBPLoss(std::vector<Tensor> &losses);

    CNAIDNNLayer *getLayer(int i) { return m_layers[i]; }
    int layerNumber()   {   return m_layers.size(); }

    void setUserOutputsNames(std::vector<string> names) {   m_userOutputsNames = names; }


    std::vector<Tensor> &getOutputLayerOuts()   {  return m_innerOutputLayerOuts;  }
    std::vector<Tensor> &getWeights() { return m_innerWeights;  }
    std::vector<Tensor> &userOutputs() {    return m_userOutputs;   }

protected:
    void __initBPDifferences();
    void __initBPWeights();


    void __updateCachedWeights(std::vector<Tensor> outputs);
    void __updateCachedDiffes(std::vector<Tensor> outputs);
    void __updateCachedLayerOutputs(std::vector<Tensor> outputs);
    void __updateBPOutputs(std::vector<Tensor> outputs);
    void __updateUserOutputs(std::vector<Tensor> outputs);
private:
    int m_outlayerNodeNum;

    std::string m_name;

    std::vector<string> m_userOutputsNames;

    std::vector<CNAIDNNLayer*> m_layers;
    GraphDef m_graphdefFront;
    GraphDef m_graphdefBack;

    SessionOptions m_options;
    std::unique_ptr<Session> m_sessionFF;
    std::unique_ptr<Session> m_sessionBP;

    std::vector< std::pair<std::string, Tensor> > m_ffInputs;
    std::vector< std::pair<std::string, Tensor> > m_bpInputs;

    std::vector<string> m_innerFFOutNames;

    std::vector<Tensor> m_innerOutputLayerOuts;
    std::vector<Tensor> m_innerDifference;
    std::vector<Tensor> m_innerWeights;

    std::vector<Tensor> m_userOutputs;
};

#endif // CNAI_DNN_NET_H
