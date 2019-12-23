#include "cnai_dnn_weight.h"
#include "cnai_dnn_node.h"

CNAIDNNWeight::CNAIDNNWeight(Scope &scope, CNAIDNNNode *left, CNAIDNNNode *right, int r, int c)
{
    m_pLeftNode = left;
    m_pRightNode = right;

    Tensor x(DT_FLOAT, TensorShape({1, 1}));
    auto x_flat = x.flat<float>();
    x_flat.setConstant(1.0f);

    m_nameInput = m_pLeftNode->nameNode() + __CNAI_WINPUT + m_pRightNode->nameNode();
    m_nameOutput = m_pLeftNode->nameNode() + __CNAI_WOUTPUT + m_pRightNode->nameNode();
    m_weightInput = Const<float>(scope.WithOpName(m_nameInput), x_flat(0), {r, c});
}


void CNAIDNNWeight::set(Scope &scope, Output w)
{
    m_weightOutput = Identity(scope.WithOpName(m_nameOutput), w);
    //m_weightOutput = Add(scope.WithOpName(m_nameOutput), m_weightInput, 0.0f);
}
