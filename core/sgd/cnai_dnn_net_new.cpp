#include "cnai_dnn_net_new.h"

CNAIDNNNetNew::CNAIDNNNetNew(std::string net):
    m_sessionFF(NewSession(m_options)),
    m_sessionBP(NewSession(m_options))
{
    m_name = net;
    m_outlayerNodeNum = 0;
}

Scope &CNAIDNNNetNew::initFrontGraph()
{
    Scope scope = Scope::NewRootScope();
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
        sprintf(ch, "%d", i);

        Output input = Const<float>(scope.WithOpName(__CNAI_INPUT_X(ch)), 0.0f, {1, 1});
        CNAIDNNNode *n = inputLayer->getNode(i);

        n->pass(scope, input);
    }

    /* [1] Pass front */
    for(int i = 0; i < m_layers.size(); i ++)
    {
        m_layers[i]->passes(scope);
    }

    GraphDef def;
    TF_CHECK_OK(scope.ToGraphDef(&def));
    m_graphdefFront = def;
    return scope;
}

Scope &CNAIDNNNetNew::initBackGraph()
{
    Scope scope = Scope::NewRootScope();

    /* [1] Loss back */
    for(int i = m_layers.size() - 1; i >= 0; i --)
    {
        m_layers[i]->losses(scope);
    }

    /* [2] Revise weights */
    for(int i = m_layers.size() - 1; i > 0; i --)
    {
        char ch[8] = {0};
        sprintf(ch, "_%d", i);
        Output step = Const<float>(scope.WithOpName(__CNAI_TRAIN_STEP + string(ch)), 0.0f, {1, 1});
        m_layers[i]->revises(scope, step);
    }

    GraphDef def;
    TF_CHECK_OK(scope.ToGraphDef(&def));
    m_graphdefBack = def;
    return scope;
}

void CNAIDNNNetNew::addLayer(Scope &scope, string name, CNAIDNN_LAYERTYPE type, int nodes, CNAIDNN_ACT_TYPE act)
{

    CNAIDNNLayer *layer = new CNAIDNNLayer(scope, m_name, name, nodes, type, act);

    if(m_layers.size() > 0)
    {
        layer->connectTo(scope, m_layers[m_layers.size() - 1]);
    }

    m_layers.push_back(layer);
    m_outlayerNodeNum = layer->numNodes();
}

void CNAIDNNNetNew::initFFInputs(std::vector<Tensor> &inputs)
{
    for(int i = 0; i < inputs.size(); i ++)
    {
        char num[8] = {0};
        sprintf(num, "%d", i);
        m_ffInputs.push_back({__CNAI_INPUT_X(num), inputs[i]});
    }
}

void CNAIDNNNetNew::initFFWeights(std::vector<Tensor> &weights)
{
    int all = 0;
    for(int i = 0; i < layerNumber(); i ++)
    {
        CNAIDNNLayer *l = getLayer(i);
        for(int j = 0; j < l->numWeights(); j ++)
        {
            CNAIDNNWeight *w = l->weight(j);
            m_ffInputs.push_back({w->inputName(), weights[all]});
            all ++;
        }
    }
    if(all != weights.size())
    {
        exit(-1);
    }
    m_innerWeights = weights;
}

void CNAIDNNNetNew::trainFront()
{
    static int first = 1;
    std::vector<Tensor> outputs;

    if(first == 1)
    {
        first = 0;
        if (m_options.target.empty())
        {
            graph::SetDefaultDevice( "/device:GPU:0" , &m_graphdefFront);
        }
        TF_CHECK_OK(m_sessionFF->Create(m_graphdefFront));
    }

    Status stat = m_sessionFF->Run(m_ffInputs, m_innerFFOutNames, {}, &outputs);

    if(stat == Status::OK())
    {
    }
    else
    {
        std::printf("Run Error! %s\n", stat.error_message().c_str());
    }
    if(m_innerFFOutNames.size() != outputs.size())
    {
        std::printf("ERROR!!!   %d %d\n", m_innerFFOutNames.size(), outputs.size());
        return;
    }

    __updateCachedWeights(outputs);
    __updateCachedDiffes(outputs);
    __updateCachedLayerOutputs(outputs);
    __updateUserOutputs(outputs);

    m_ffInputs.clear();
}

void CNAIDNNNetNew::trainBack()
{
    static int first = 1;
    std::vector<Tensor> outputs;
    if(first == 1)
    {
        first = 0;
        if (m_options.target.empty())
        {
            graph::SetDefaultDevice( "/device:GPU:0" , &m_graphdefBack);
        }
        TF_CHECK_OK(m_sessionBP->Create(m_graphdefBack));
    }

    __initBPWeights();
    __initBPDifferences();

    Status stat = m_sessionBP->Run(m_bpInputs, m_userOutputsNames, {}, &outputs);

    if(stat == Status::OK())
    {
        //std::printf("Run OK!\n");
    }
    else
    {
        std::printf("Run Error! %s\n", stat.error_message().c_str());
    }
    if(m_userOutputsNames.size() != outputs.size())
    {
        std::printf("ERROR!!!   %d %d\n", m_userOutputsNames.size(), outputs.size());
        return;
    }
    __updateUserOutputs(outputs);

    m_bpInputs.clear();
}

void CNAIDNNNetNew::__initBPWeights()
{
    int all = 0;
    for(int i = 0; i < layerNumber(); i ++)
    {
        CNAIDNNLayer *l = getLayer(i);
        for(int j = 0; j < l->numWeights(); j ++)
        {
            CNAIDNNWeight *w = l->weight(j);
            m_bpInputs.push_back({w->inputName(), m_innerWeights[all]});
            all ++;
        }
    }
    if(all != m_innerWeights.size())
    {
        exit(-1);
    }
}

void CNAIDNNNetNew::__initBPDifferences()
{
    int all = 0;
    for(int i = 0; i < layerNumber(); i ++)
    {
        CNAIDNNLayer *l = getLayer(i);
        for(int j = 0; j < l->numNodes(); j ++)
        {
            CNAIDNNNode *n = l->getNode(j);
            m_bpInputs.push_back({n->nameNode() + __CNAI_KEYWORD_DERIVATIVE, m_innerDifference[all]});
            all ++;
        }
    }
}

void CNAIDNNNetNew::__updateCachedWeights(std::vector<Tensor> outputs)
{
    m_innerWeights.clear();
    for(int i = m_outlayerNodeNum; i < m_innerWeights.size() + m_outlayerNodeNum; i ++)
    {
        Tensor t = outputs[i];
        m_innerWeights.push_back(t);
    }
}

void CNAIDNNNetNew::__updateCachedDiffes(std::vector<Tensor> outputs)
{
    m_innerDifference.clear();
    for(int i = m_outlayerNodeNum + m_innerWeights.size(); i < outputs.size() - m_userOutputsNames.size(); i ++)
    {
        Tensor t = outputs[i];
        m_innerDifference.push_back(t);
    }
}

void CNAIDNNNetNew::__updateCachedLayerOutputs(std::vector<Tensor> outputs)
{
    m_innerOutputLayerOuts.clear();
    for(int i = 0; i < m_outlayerNodeNum; i ++)
    {
        Tensor t = outputs[i];
        m_innerOutputLayerOuts.push_back(t);
    }
}

void CNAIDNNNetNew::__updateUserOutputs(std::vector<Tensor> outputs)
{

}
