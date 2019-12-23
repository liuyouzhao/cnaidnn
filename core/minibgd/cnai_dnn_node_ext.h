#ifndef CNAI_DNN_NODE_EXT_H
#define CNAI_DNN_NODE_EXT_H

#include "cnai_dnn_node.h"

class CNAIDNNNodeExt : public CNAIDNNNode
{
public:


    /** Override pass function for mini-bgd
     * @brief pass
     * @param scope
     * @param ip
     * @return
     */
    Output pass(Scope &scope, Output ip);
    Output pass(Scope &scope, Output ip,
                __ACT_CB,
                __DER_CB);

    /** Override loss function for mini-bgd
     * @brief loss
     * @param scope
     * @param err
     * @return
     */
    Output loss(Scope &scope, Output err);

friend class CNAIDNNLayerExt;
protected:
    CNAIDNNNodeExt(Scope &scope,
                   std::string net,
                   std::string layer,
                   std::string name,
                   int r = 1, int c = 1,
                   CNAIDNN_ACT_TYPE at = CN_AT_Softplus);

    std::vector<Output> m_actDiffCache;
};

#endif // CNAI_DNN_NODE_EXT_H
