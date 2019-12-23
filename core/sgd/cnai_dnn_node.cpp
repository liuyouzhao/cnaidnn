#include "cnai_dnn_node.h"
#include "cnai_dnn_der_base.h"

std::string CNAIDNNNode::s_inputName = __CNAI_KEYWORD_INPUT;

std::string CNAIDNNNode::s_outputName = __CNAI_KEYWORD_OUTPUT;

CNAIDNNNode::CNAIDNNNode(Scope &scope, std::string net, std::string layer, std::string name, int r, int c, CNAIDNN_ACT_TYPE t)
{
    m_netName = net;
    m_layerName = layer;
    m_nodeName = name;
    m_at = t;

    std::printf("[%s] ", nameNode().c_str());
}

CNAIDNNNode::~CNAIDNNNode()
{
}

Output CNAIDNNNode::pass(Scope &scope, Output ip)
{
    /* Use Sum only for making m_weightsSumInput accessable */
    m_weightsSumInput = ip;

    switch(m_at)
    {
    case CN_AT_Softplus:
    {
        m_activationOutput = Softplus(scope.WithOpName(nameOutput()), ip);
        {
            Sub sub = Sub(scope, 0.0f, ip);
            Exp exp = Exp(scope, sub);
            Add add = Add(scope, 1.0f, exp);
            m_activationDiff = Div(scope.WithOpName(nameNode() + __CNAI_KEYWORD_DERIVATIVE), 1.0f, add);
        }
    } break;
    case CN_AT_Softmax:
    {
        Softmax sp = Softmax(scope.WithOpName(__CN_INNER + nameOutput()), ip);
        m_activationOutput = Sum(scope.WithOpName(nameOutput()), sp, 0);
        {
            /* NOT Implemented */
            //m_activationDiff = div;
        }
    } break;
    case CN_AT_Relu:
    {
        Relu sp = Relu(scope.WithOpName(__CN_INNER + nameOutput()), ip);
        m_activationOutput = Sum(scope.WithOpName(nameOutput()), sp, 0);
        {
            Less less = Less(scope, ip, {0.0f});
            if(less.z == Const<bool>(scope, 0))
            {
                m_activationOutput = Const<float>(scope, 0.0f, {1, 1});
            }
            else
            {
                m_activationOutput = Const<float>(scope, 1.0f, {1, 1});
            }
        }
    } break;
    case CN_AT_Tan:
    {
        Tan sp = Tan(scope.WithOpName(__CN_INNER + nameOutput()), ip);
        m_activationOutput = Sum(scope.WithOpName(nameOutput()), sp, 0);
    } break;
    case CN_AT_SIGMOD:
    {
        Sub sub = Sub(scope, 0.0f, ip);
        Exp exp = Exp(scope, sub);
        Add add = Add(scope, exp, 1.0f);
        m_activationOutput = Div(scope.WithOpName(nameOutput()), 1.0f, add);
        {
            Sub _s = Sub(scope, 1.0f, m_activationOutput);
            m_activationDiff = Mul(scope.WithOpName(nameNode() + __CNAI_KEYWORD_DERIVATIVE), m_activationOutput, _s);
        }
    } break;
    case CN_AT_Non:
    {
        m_activationOutput = Sum(scope.WithOpName(nameOutput()), ip, 0);
        Output diff = Const<float>(scope, 0.0f, {1, 1});
        m_activationDiff = Sum(scope.WithOpName(nameNode() + __CNAI_KEYWORD_DERIVATIVE), diff, 0);
    } break;
    }

    return m_activationOutput;
}

Output CNAIDNNNode::pass(Scope &scope, Output ip,
                        __ACT_CB,
                        __DER_CB)
{
    /* Use Sum only for making m_weightsSumInput accessable */
    m_weightsSumInput = ip;
    m_activationOutput = activation(scope, ip);
    {

        m_activationDiff = derivative(scope, ip);
    }
    return m_activationOutput;
}

/**
 * @brief CNAIDNNNode::back
 * @param scope
 * @param err
 * @return
 */
Output CNAIDNNNode::loss(Scope &scope, Output err)
{
    /*
                |------------|
         loss---| Net  | Out | <----err
                |------------|
      d(ET)    d(ET)     d(Out)
      ----- == ----- *  --------
      d(Loss)  d(Out)    d(Net)
    */
    m_error = Sum(scope.WithOpName(nameNode() + "_error"), err, 0);
    m_loss = Mul(scope.WithOpName(nameNode() + "_loss"), err, m_activationDiff);
    return m_loss;
}
