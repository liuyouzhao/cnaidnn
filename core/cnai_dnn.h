#ifndef CNAI_DNN_H
#define CNAI_DNN_H

#include "cnai_dnn_net.h"

class CNAIDNN
{
public:
    CNAIDNN(std::string name);

    void begin();

    void netAddLayer(string name, CNAIDNN_LAYERTYPE type, int nodes, CNAIDNN_ACT_TYPE act = CN_AT_Softplus);
    void netAddLayer(string name, CNAIDNN_LAYERTYPE type, int nodes, __ACT_CB, __DER_CB);

    void end();

    Status save(const char *folder, const char *filename);
    Status load(const char *fullname);

    void setInputs(std::vector< std::pair<string, Tensor> > &inputData);
    void setOutputsTags(std::vector<string> &tags);

    std::vector<Tensor> train();
    std::vector<Tensor> run();
private:
    CNAIDNNNet m_netTrain;
    CNAIDNNNet m_netTest;

    Scope m_scopeTrain;
    Scope m_scopeTest;
};

#endif // CNAI_DNN_H
