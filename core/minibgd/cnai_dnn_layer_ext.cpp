#include "cnai_dnn_layer_ext.h"
#include "cnai_dnn_node_ext.h"

CNAIDNNLayerExt::CNAIDNNLayerExt(Scope &scope,
                                 std::string net,
                                 std::string layer,
                                 int nodes,
                                 CNAIDNN_LAYERTYPE type,
                                 CNAIDNN_ACT_TYPE ntype):
    CNAIDNNLayer(scope, net, layer, nodes, type, ntype)
{
}

int CNAIDNNLayerExt::constructNodes(Scope &scope)
{
    for(int i = 0; i < m_numNodes; i ++)
    {
        char nm[32] = {0};
        sprintf(nm, "_n%d_", i);

        CNAIDNNNode *node = new CNAIDNNNodeExt(scope, m_netName, m_name, std::string(nm), 1, 1, m_nActType);
        m_nodes.push_back(node);
    }
}
