#include "cnai_dnn_node_ext.h"

CNAIDNNNodeExt::CNAIDNNNodeExt(Scope &scope,
                               std::string net,
                               std::string layer,
                               std::string name,
                               int r, int c,
                               CNAIDNN_ACT_TYPE at):
    CNAIDNNNode(scope, net, layer, name, r, c, at)
{

}

Output CNAIDNNNodeExt::pass(Scope &scope, Output ip)
{
    CNAIDNNNode::pass(scope, ip);
    m_actDiffCache.push_back(m_activationDiff);
}

Output CNAIDNNNodeExt::pass(Scope &scope, Output ip,
                            __ACT_CB,
                            __DER_CB)
{
    CNAIDNNNode::pass(scope, ip, activation, derivative);
    m_actDiffCache.push_back(m_activationDiff);
}

Output CNAIDNNNodeExt::loss(Scope &scope, Output err)
{
    m_error = Sum(scope.WithOpName(nameNode() + "_error"), err, 0);

    AddN diffs = AddN(scope, m_actDiffCache);
    Output numDiff = Const<float>(scope, (float)(m_actDiffCache.size()), {1, 1});

    m_loss = Div(scope.WithOpName(nameNode() + "_loss"), diffs, numDiff);

    m_actDiffCache.clear();

    return m_loss;
}
