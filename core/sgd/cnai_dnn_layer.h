#ifndef CNAI_DNN_LAYER_H
#define CNAI_DNN_LAYER_H

#include "cnai_inc.h"
#include "cnai_dnn_node.h"
#include "cnai_dnn_weight.h"

enum CNAIDNN_LAYERTYPE
{
    CNAI_DNN_TYPE_NORMAL,
    CNAI_DNN_TYPE_CONV2D,
    CNAI_DNN_TYPE_CONV3D
};

#define __CNAI_INPUT_B std::string("cn_input_b")

class CNAIDNNLayer
{
public:
    ~CNAIDNNLayer();
    int numNodes() {    return m_nodes.size();  }
    CNAIDNNNode *getNode(int i) {   return m_nodes[i];  }

    int numWeights() {  return m_wIn.size();    }
    CNAIDNNWeight *weight(int i) {  return m_wIn[i];    }
    CNAIDNNLayer *left() {   return m_pLeft; }
    CNAIDNNLayer *right() {   return m_pRight; }
friend class CNAIDNNNet;
friend class CNAIDNNNetNew;
friend class CNAIDNNNetExt;
friend class CNAIDNN;
protected:
    CNAIDNNLayer(Scope &scope,
                 std::string net,
                 std::string layer,
                 int nodes,
                 CNAIDNN_LAYERTYPE type,
                 CNAIDNN_ACT_TYPE ntype);

    /** Being overrided by subsidiaries!
     * @brief constructNodes
     * @param scope
     * @return
     */
    virtual int constructNodes(Scope &scope);

    std::string name()  {   return m_netName + m_name;  }


    /**
     * @brief connectTo
     * @param scope
     * @param left
     *
     * Connect to upper layer from right size!
     */
    void connectTo(Scope &scope, CNAIDNNLayer *left);
    void setRight(CNAIDNNLayer *right) {    m_pRight = right;    }




    /** KEY FUNCTION
     * @brief passes
     * @param scope
     */
    virtual void passes(Scope &scope);

    /**
     * @brief passesExt
     * @param scope
     * Using custimized activation and derivative funtions
     * given by user
     */
    virtual void passesExt(Scope &scope);

    /** KEY FUNCTION
     * @brief losses
     * @param scope
     *
     * Be sure of that Right neibor layer
     * already done the losses
     *
     */
    virtual void losses(Scope &scope);

    /** KEY FUNCTION
     * @brief revises
     * @param scope
     */
    virtual void revises(Scope &scope, Output step);

    /**
     * @brief setActivationFunction
     * Set custimized activation function
     */
    void setActivationFunction(__ACT_CB)
    {   this->activation = activation;  }

    /**
     * @brief setDerivativeFuntion
     * Set custimized delivative function
     */
    void setDerivativeFuntion(__DER_CB)
    {   this->derivative = derivative;  }



protected:

    std::vector<CNAIDNNNode*> m_nodes;
    /**
     * @brief m_wIn
     * Only save input weights
     */
    std::vector<CNAIDNNWeight*> m_wIn;

    std::string m_netName;
    std::string m_name;

    CNAIDNNLayer *m_pLeft;
    CNAIDNNLayer *m_pRight;

    CNAIDNN_LAYERTYPE m_type;
    CNAIDNN_ACT_TYPE m_nActType;

    int m_numNodes;

    __ACT_CB;
    __DER_CB;
};

#endif // CNAI_DNN_LAYER_H
