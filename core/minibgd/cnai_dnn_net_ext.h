#ifndef CNAI_DNN_NET_EXT_H
#define CNAI_DNN_NET_EXT_H

#include "cnai_dnn.h"
#include "cnai_dnn_net.h"

class CNAIDNNNetExt : public CNAIDNNNet
{
public:
    CNAIDNNNetExt(std::string net);

    /* Override pass function */
    virtual GraphDef end(Scope &scope);

    void setBatch(int batch) {  m_numBatch = batch; }

private:
    /* Batch value */
    int m_numBatch;
};

#endif // CNAI_DNN_NET_EXT_H
