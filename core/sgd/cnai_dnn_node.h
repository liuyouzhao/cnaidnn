#ifndef CNAI_DNN_NODE_H
#define CNAI_DNN_NODE_H

#include "cnai_inc.h"

enum CNAIDNN_ACT_TYPE
{
    CN_AT_Softplus = 0,
    CN_AT_Softmax = 1,
    CN_AT_Relu = 2,
    CN_AT_Tan = 3,
    CN_AT_SIGMOD = 4,
    CN_AT_Non = 100
};

class cnai_dnn_weight;

/*
            |------------|
        w---| Net  | Out | <----err
            |------------|

  d(ET)    d(ET)     d(Out)    d(Net)
  ----- == ----- *  ------ *  ------
  d(w)     d(Out)    d(Net)    d(w)

  d(ET)    d(E1+E2+E3...+En)     d(Out)    d(Net)
  ----- == ------------------ *  ------ *  ------
  d(w)     d(Out)                d(Net)    d(w)


  d(ET)      d(E1)     d(O-1)    d(N-1)         d(En)    d(O-n)   d(N-n)       d(Out)    d(Net)
  ----- == [ ------ * ------- * ------- + ... + ------ * ------ * ------ ] *   ------ *  ------
  d(w)       d(O-1)    d(N-1)    d(Out)         d(O-n)   d(N-n)   d(Out)       d(Net)    d(w)


*/
class CNAIDNNWeight;
class CNAIDNNLayer;

class CNAIDNNNode
{
public:
    ~CNAIDNNNode();

    std::string nameOutput()   {   return m_netName + m_layerName + m_nodeName + s_outputName;  }
    std::string nameInput()   {   return m_netName + m_layerName + m_nodeName + s_inputName;  }
    std::string nameNode()     {   return m_netName + m_layerName + m_nodeName;  }

    Output activatiOutput() {   return m_activationOutput;  }
    Output getInput()   {   return m_weightsSumInput;   }
    Output getLoss()    {   return m_loss;  }
    Output getError()   {   return m_error; }

    Output pass(Scope &scope, Output ip);
    Output pass(Scope &scope, Output ip,
                __ACT_CB,
                __DER_CB);


    Output loss(Scope &scope, Output err);

friend class CNAIDNNLayer;

protected:
    CNAIDNNNode(Scope &scope,
                std::string net,
                std::string layer,
                std::string name,
                int r = 1, int c = 1,
                CNAIDNN_ACT_TYPE at = CN_AT_Softplus);

protected:
    std::string m_netName;
    std::string m_layerName;
    std::string m_nodeName;
    static std::string s_inputName;
    static std::string s_outputName;

    /**
     * @brief m_activationOutput
     * Activation function output value
     */
    Output        m_activationOutput;

    /**
     * @brief m_weightsSumInput
     * Weight Sumarize value addon to output
     */
    Output        m_weightsSumInput;

    /**
     * @brief m_activationDiff
     * Difference for backpropagation
     */
    Output        m_activationDiff;

    /**
     * @brief m_error
     * Error the first step in BP
     *
     * loss<--[Net|Out]<--error
     *
     *
     */
    Output        m_error;

    /**
     * @brief m_loss
     */
    Output        m_loss;

    CNAIDNN_ACT_TYPE m_at;
};

#endif // CNAI_DNN_NODE_H
