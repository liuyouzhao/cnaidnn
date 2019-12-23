#include "cnai_dnn.h"

CNAIDNN::CNAIDNN(string name):
    m_netTrain(name),
    m_netTest(name),
    m_scopeTrain(Scope::NewRootScope()),
    m_scopeTest(Scope::NewRootScope())
{
}

void CNAIDNN::begin()
{
    m_scopeTrain = (m_netTrain.begin());
    m_scopeTest = (m_netTest.begin());
}

void CNAIDNN::end()
{
    m_netTrain.end((m_scopeTrain));
    m_netTest.end((m_scopeTest));
}

void CNAIDNN::netAddLayer(string name, CNAIDNN_LAYERTYPE type, int nodes, CNAIDNN_ACT_TYPE act)
{
    m_netTrain.addLayer((m_scopeTrain), name, type, nodes, act);
    m_netTest.addLayer((m_scopeTest), name, type, nodes, act);
}

void CNAIDNN::netAddLayer(string name, CNAIDNN_LAYERTYPE type, int nodes, __ACT_CB, __DER_CB)
{
    m_netTrain.addLayer((m_scopeTrain), name, type, nodes, activation, derivative);
    m_netTest.addLayer((m_scopeTest), name, type, nodes, activation, derivative);
}

Status CNAIDNN::save(const char *folder, const char *filename)
{
    char ch[256] = {0};
    sprintf(ch, "%s_train", filename);
    m_netTrain.save(folder, ch);
    m_netTest.save(folder, filename);
}

Status CNAIDNN::load(const char *fullname)
{
    char ch[512] = {0};
    sprintf(ch, "%s_train", fullname);
    m_netTrain.load(ch);
    m_netTest.load(fullname);
}

std::vector<Tensor> CNAIDNN::train()
{
    std::vector<string> outNames;
    CNAIDNNLayer *l = m_netTrain.getLayer(m_netTrain.layerNumber() - 1);
    m_netTrain.fillOutputNames(l->name(), outNames);
}

std::vector<Tensor> CNAIDNN::run()
{

}
