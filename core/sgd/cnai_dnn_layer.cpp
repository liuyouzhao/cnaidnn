#include "cnai_dnn_layer.h"

CNAIDNNLayer::CNAIDNNLayer( Scope &scope,
                            std::string net,
                            std::string layer,
                            int nodes,
                            CNAIDNN_LAYERTYPE type,
                            CNAIDNN_ACT_TYPE ntype)
{
    m_pLeft = NULL;
    m_pRight = NULL;
    activation = NULL;
    derivative = NULL;
    m_netName = net;
    m_name = layer;
    m_type = type;
    m_nActType = ntype;
    m_numNodes = nodes;

    constructNodes(scope);
}

CNAIDNNLayer::~CNAIDNNLayer()
{
}

int CNAIDNNLayer::constructNodes(Scope &scope)
{
    for(int i = 0; i < m_numNodes; i ++)
    {
        char nm[32] = {0};
        sprintf(nm, "_n%d_", i);

        CNAIDNNNode *node = new CNAIDNNNode(scope, m_netName, m_name, std::string(nm), 1, 1, m_nActType);
        m_nodes.push_back(node);
    }
}

void CNAIDNNLayer::connectTo(Scope &scope, CNAIDNNLayer *left)
{
    if(m_pLeft)
    {
        std::printf("[ERROR] %s %s %d\n", __FILE__, __FUNCTION__, __LINE__);
        std::printf("m_pLeft is not NULL\n");
        return;
    }
    m_pLeft = left;

    m_pLeft->setRight(this);



    switch(m_type)
    {
    case CNAI_DNN_TYPE_NORMAL:
    {
        int n = m_pLeft->numNodes();

        for(int i = 0; i < m_numNodes; i ++)
        {
            CNAIDNNNode *cn = getNode(i);
            for(int j = 0; j < n; j ++)
            {
                CNAIDNNNode *ln = m_pLeft->getNode(j);
                CNAIDNNWeight *pw = new CNAIDNNWeight(scope, ln, cn);
                m_wIn.push_back(pw);
            }
        }
    } break;
    case CNAI_DNN_TYPE_CONV2D:
    case CNAI_DNN_TYPE_CONV3D:
        /// Not implemented
        std::printf("NOT Implemented!\n");
        break;
    }
}

void CNAIDNNLayer::passesExt(Scope &scope)
{
    if(m_pLeft == NULL)
    {
        for(int i = 0; i < m_numNodes; i ++)
        {
            CNAIDNNNode *n = getNode(i);
            n->pass(scope, n->getInput(), activation, derivative);
        }
        return;
    }

    for(int i = 0; i < m_numNodes; i ++)
    {
        CNAIDNNNode *cn = getNode(i);

        //Output sum = Const<float>(scope, 0.0f, {1, 1});
        std::vector<Output> sums;

        for(int j = 0; j < m_pLeft->numNodes(); j ++)
        {
            CNAIDNNNode *ln = m_pLeft->getNode(j);
            Output lo = ln->activatiOutput();

            /* group by current layer! */
            CNAIDNNWeight *w = weight(i * m_pLeft->numNodes() + j);

            Mul m = Mul(scope, lo, w->getInputWeights());

            sums.push_back(m);
        }

        Output b = Const<float>(scope.WithOpName(__CNAI_INPUT_B), 0.0f, {1, 1});

        AddN addn = AddN(scope, sums);

        Add add = Add(scope.WithOpName(cn->nameNode() + "_suminput"), b, addn);

        cn->pass(scope, add, activation, derivative);
    }
}

void CNAIDNNLayer::passes(Scope &scope)
{
    if(m_pLeft == NULL)
    {
        for(int i = 0; i < m_numNodes; i ++)
        {
            CNAIDNNNode *n = getNode(i);
            n->pass(scope, n->getInput());
        }
        return;
    }

    switch(m_type)
    {
    case CNAI_DNN_TYPE_NORMAL:
        for(int i = 0; i < m_numNodes; i ++)
        {
            CNAIDNNNode *cn = getNode(i);

            //Output sum = Const<float>(scope, 0.0f, {1, 1});
            std::vector<Output> sums;

            for(int j = 0; j < m_pLeft->numNodes(); j ++)
            {
                CNAIDNNNode *ln = m_pLeft->getNode(j);
                Output lo = ln->activatiOutput();

                /* group by current layer! */
                CNAIDNNWeight *w = weight(i * m_pLeft->numNodes() + j);

                Mul m = Mul(scope, lo, w->getInputWeights());

                sums.push_back(m);
            }

            Output b = Const<float>(scope.WithOpName(__CNAI_INPUT_B), 0.0f, {1, 1});

            AddN addn = AddN(scope, sums);

            Add add = Add(scope.WithOpName(cn->nameNode() + "_suminput"), b, addn);

            //Div div = Div(scope.WithOpName(cn->nameNode() + "_nodeinput"), add, (float)(m_pLeft->numNodes() + 1));

            cn->pass(scope, add);
        }
        break;
    case CNAI_DNN_TYPE_CONV2D:
    case CNAI_DNN_TYPE_CONV3D:
        /// Not implemented
        std::printf("NOT Implemented!\n");
        break;
    }
}

void CNAIDNNLayer::losses(Scope &scope)
{
    /**Caution! Make sule Output Layer's Error has been inited*/
    if(m_pRight == NULL)
    {
        for(int i = 0; i < m_numNodes; i ++)
        {
            CNAIDNNNode *n = getNode(i);
            n->loss(scope, n->getError());
            Output sumerr = Identity(scope.WithOpName(n->nameNode() + "_sumerr"), n->getError());
        }
        return;
    }

    switch(m_type)
    {
    case CNAI_DNN_TYPE_NORMAL:
        for(int i = 0; i < m_numNodes; i ++)
        {
            CNAIDNNNode *ln = getNode(i);

            std::vector<Output> all;
            for(int j = 0; j < m_pRight->numNodes(); j ++)
            {
                CNAIDNNNode *rn = m_pRight->getNode(j);
                Output rl = rn->getError();

                /* group by right layer! */
                CNAIDNNWeight *w = m_pRight->weight(j * numNodes() + i);

                Mul m = Mul(scope, rl, w->getInputWeights());
                all.push_back(m);
            }
            AddN addn = AddN(scope.WithOpName(ln->nameNode() + "_sumerr"), all);
            ln->loss(scope, addn);
        }
        break;
    case CNAI_DNN_TYPE_CONV2D:
    case CNAI_DNN_TYPE_CONV3D:
        /// Not implemented
        std::printf("NOT Implemented!\n");
        break;
    }
}

void CNAIDNNLayer::revises(Scope &scope, Output step)
{
    if(m_pLeft == NULL)
    {
        return;
    }
    switch(m_type)
    {
    case CNAI_DNN_TYPE_NORMAL:

        for(int i = 0; i < m_numNodes; i ++)
        {
            CNAIDNNNode *cn = getNode(i);
            Output loss = cn->getLoss();

            for(int j = 0; j < m_pLeft->numNodes(); j ++)
            {
                int index = i * m_pLeft->numNodes() + j;

                CNAIDNNNode *ln = m_pLeft->getNode(j);


                Output out = ln->activatiOutput();

                CNAIDNNWeight *w = weight(index);
                Output currentWeight = w->getInputWeights();

                if(ln != w->left() || cn != w->right())
                {
                    printf("\nERROR Steven\n");
                    exit(-1);
                }



                Mul m = Mul(scope, out, loss);
                Mul m2 = Mul(scope.WithOpName(cn->nameNode() + "_m2"), m, step);

                /* Gradient Descent, +W = W - s*(Loss)    */
                Add newWeight = Add(scope, currentWeight, m2);
                w->set(scope, newWeight);
            }
        }
        break;
    case CNAI_DNN_TYPE_CONV2D:
    case CNAI_DNN_TYPE_CONV3D:
        /// Not implemented
        std::printf("NOT Implemented!\n");
        break;
    }
}
