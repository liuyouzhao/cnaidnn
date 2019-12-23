#include "cnai_dnn_net_ext.h"

CNAIDNNNetExt::CNAIDNNNetExt(std::string net):
    CNAIDNNNet(net)
{
    m_numBatch = 1;
}

GraphDef CNAIDNNNetExt::end(Scope &scope)
{
    for(int batch = 0; batch < m_numBatch; batch ++)
    {
        /**
          At least input + output layers
          */
        if(m_layers.size() < 2)
        {
            exit(-1);
        }

        /* Init Total Input for the input layer */
        CNAIDNNLayer *inputLayer = m_layers[0];
        for(int i = 0; i < inputLayer->numNodes(); i ++)
        {
            char ch[32] = {0};
            char bh[16] = {0};
            sprintf(ch, "%d", i);
            sprintf(bh, "%d", batch);
            Output input = Const<float>(scope.WithOpName(__CNAI_INPUT_X(ch)), 99999.0f, {1, 1});
            CNAIDNNNode *n = inputLayer->getNode(i);

            n->pass(scope, input);
        }

        /* [1] Pass front */
        for(int i = 0; i < m_layers.size(); i ++)
        {
            m_layers[i]->passes(scope);
        }
    }

    /* Init Total Errors for the output layer loss */
    CNAIDNNLayer *outLayer = m_layers[m_layers.size() - 1];
    std::vector<Output> adds;
    for(int i = 0; i < outLayer->numNodes(); i ++)
    {
        char ch[32] = {0};
        sprintf(ch, "%d", i);

        Output target = Const<float>(scope.WithOpName(__CNAI_TARGET_Y(ch)), 10000.0f, {1, 1});
        CNAIDNNNode *n = outLayer->getNode(i);

        Sub err = Sub(scope, target, n->activatiOutput());

        n->loss(scope, err);
    }

    /* [2] Loss back */
    for(int i = m_layers.size() - 1; i >= 0; i --)
    {
        m_layers[i]->losses(scope);
    }

    /* [3] Revise weights */
    for(int i = m_layers.size() - 1; i > 0; i --)
    {
        char ch[32] = {0};
        sprintf(ch, "_%d", i);
        Output step = Const<float>(scope.WithOpName(__CNAI_TRAIN_STEP + ch), 9999999.0f, {1, 1});
        m_layers[i]->revises(scope, step);
    }


    /* [4] Save checkpoint */
    /* TODO: Is there a accessable way to Save all weights from GPU? The Save() method dows not work in GTX1070*/

    /* [5] Update inner tags */
    __updateInnerWeightNames();

    GraphDef def;
    TF_CHECK_OK(scope.ToGraphDef(&def));
    m_graphdef = def;

    if (m_options.target.empty())
    {
        graph::SetDefaultDevice( "/device:GPU:0" , &m_graphdef);
    }
    TF_CHECK_OK(m_session->Create(m_graphdef));

    return m_graphdef;
}
