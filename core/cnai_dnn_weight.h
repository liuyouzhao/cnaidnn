#ifndef CNAI_DNN_WEIGHT_H
#define CNAI_DNN_WEIGHT_H

#include "cnai_inc.h"

class CNAIDNNNode;

class CNAIDNNWeight
{
public:
friend class CNAIDNNLayer;
friend class CNAIDNNNet;
    std::string outputName() {  return m_nameOutput;    }
    std::string inputName() {   return m_nameInput; }

    CNAIDNNNode *left() {   return m_pLeftNode; }
    CNAIDNNNode *right() {   return m_pRightNode; }

protected:
    CNAIDNNWeight(Scope &scope, CNAIDNNNode *left, CNAIDNNNode *right, int r = 1, int c = 1);

    void set(Scope &scope, Output getInputWeights);
    Output getInputWeights() {    return m_weightInput;    }
    Output getOutputWeights() {   return m_weightOutput;  }


private:
    CNAIDNNNode *m_pLeftNode;
    CNAIDNNNode *m_pRightNode;

    Output m_weightInput;
    Output m_weightOutput;

    std::string m_nameInput;
    std::string m_nameOutput;
};

#endif // CNAI_DNN_WEIGHT_H
