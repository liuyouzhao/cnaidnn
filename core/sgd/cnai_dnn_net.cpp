#include "cnai_dnn_net.h"

/* TODO: Use placeholder instead of Output for all weights values */

/**
 * @brief CNAIDNNNet::CNAIDNNNet
 * @param net
 */
CNAIDNNNet::CNAIDNNNet(std::string net):
    m_session(NewSession(m_options))
{
    m_name = net;
    m_numAllWeights = 0;
}

Scope CNAIDNNNet::begin()
{
    Scope scope = Scope::NewRootScope();
    return scope;
}

GraphDef CNAIDNNNet::end(Scope &scope)
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
        sprintf(ch, "%d", i);

        Output input = Const<float>(scope.WithOpName(__CNAI_INPUT_X(ch)), 99999.0f, {1, 1});
        CNAIDNNNode *n = inputLayer->getNode(i);

        n->pass(scope, input);
    }

    /* [1] Pass front */
    for(int i = 0; i < m_layers.size(); i ++)
    {
        m_layers[i]->passes(scope);
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

void CNAIDNNNet::addLayer(Scope &scope, string name, CNAIDNN_LAYERTYPE type, int nodes, CNAIDNN_ACT_TYPE act)
{

    CNAIDNNLayer *layer = new CNAIDNNLayer(scope, m_name, name, nodes, type, act);

    if(m_layers.size() > 0)
    {
        layer->connectTo(scope, m_layers[m_layers.size() - 1]);
    }

    m_layers.push_back(layer);
    m_numAllWeights += layer->numWeights();
}

void CNAIDNNNet::addLayer(Scope &scope, string name, CNAIDNN_LAYERTYPE type, int nodes,
                          __ACT_CB,
                          __DER_CB)
{
    CNAIDNNLayer *layer = new CNAIDNNLayer(scope, m_name, name, nodes, type, CN_AT_Non);

    layer->setActivationFunction(activation);
    layer->setDerivativeFuntion(derivative);

    if(m_layers.size() > 0)
    {
        layer->connectTo(scope, m_layers[m_layers.size() - 1]);
    }

    m_layers.push_back(layer);
    m_numAllWeights += layer->numWeights();
}


void CNAIDNNNet::train(std::vector< std::pair<std::string, Tensor> > &train_data,
                       std::vector<std::string> &output_names,
                       std::vector<Tensor> *outputs)
{
    Status stat = m_session->Run(train_data, output_names, {}, outputs);

    if(stat != Status::OK())
    {
        std::printf("Run Error! %s\n", stat.error_message().c_str());
    }
    if(output_names.size() != outputs->size())
    {
        std::printf("ERROR!!!   %d %d\n", output_names.size(), outputs->size());
        return;
    }
}

void CNAIDNNNet::run(std::vector< std::pair<string, Tensor> > &run_data,
                     std::vector<string> &output_names,
                     std::vector<Tensor> *outputs)
{

    Status stat = m_session->Run(run_data, output_names, {}, outputs);

    if(stat != Status::OK())
    {
        std::printf("Run Error! %s\n", stat.error_message().c_str());
    }
    if(output_names.size() != outputs->size())
    {
        std::printf("ERROR!!!   %d %d\n", output_names.size(), outputs->size());
        return;
    }
}

std::vector< std::pair<string, Tensor> > CNAIDNNNet::train(std::vector< std::pair<string, Tensor> > &train_data)
{
    std::vector<string> outputNames;
    std::vector<Tensor> outputs;
    std::vector< std::pair<string, Tensor> > results;


    for(int i = 0; i < m_innerWeightNamesIn.size(); i ++)
    {
        train_data.push_back({m_innerWeightNamesIn[i], m_innerWeights[i]});
    }
    for(int i = 0; i < m_innerWeightNamesOut.size(); i ++)
    {
        outputNames.push_back(m_innerWeightNamesOut[i]);
    }
    for(int i = 0; i < m_outputUserNames.size(); i ++)
    {
        outputNames.push_back(m_outputUserNames[i]);
    }

    Status stat = m_session->Run(train_data, outputNames, {}, &outputs);

    if(stat != Status::OK())
    {
        std::printf("Run Error! %s\n", stat.error_message().c_str());
    }
    if(outputNames.size() != outputs.size())
    {
        std::printf("ERROR!!!   %d %d\n", outputNames.size(), outputs.size());
        return results;
    }

    updateInnerWeight(outputNames, outputs);
    for(int i = m_innerWeights.size(); i < outputs.size(); i ++)
    {
        results.push_back({outputNames[i], outputs[i]});
    }

    return results;
}

std::vector< std::pair<string, Tensor> > CNAIDNNNet::run(std::vector< std::pair<string, Tensor> > &run_data)
{
    std::vector<Tensor> outputs;
    std::vector< std::pair<string, Tensor> > results;

    for(int i = 0; i < m_innerWeightNamesIn.size(); i ++)
    {
        run_data.push_back({m_innerWeightNamesIn[i], m_innerWeights[i]});
    }

    Status stat = m_session->Run(run_data, m_outputUserNames, {}, &outputs);

    if(stat != Status::OK())
    {
        std::printf("Run Error! %s\n", stat.error_message().c_str());
    }
    if(m_outputUserNames.size() != outputs.size())
    {
        std::printf("ERROR!!!   %d %d\n", m_outputUserNames.size(), outputs.size());
        return results;
    }

    for(int i = 0; i < outputs.size(); i ++)
    {
        results.push_back({m_outputUserNames[i], outputs[i]});
    }

    return results;
}

Status CNAIDNNNet::saveWeightNames(const char *folder, const char *filename)
{
    std::string sf(folder);
    std::string f(filename);
    std::string fullpath = sf + "/" + f + ".wns";
    /* Save all weight names */
    FILE *file = fopen(fullpath.c_str(), "w");

    for(int i = 0; i < m_innerWeightNamesIn.size(); i ++)
    {
        fwrite((m_innerWeightNamesIn[i] + "\n").c_str(), m_innerWeightNamesIn[i].length() + 1, 1, file);
    }
    fflush(file);
    fclose(file);

    return Status::OK();
}

Status CNAIDNNNet::saveGraph(const char *folder, const char *filename)
{
    std::string sf(folder);
    std::string f(filename);
    std::string fgraph = sf + "/" + f + ".pb";
    Status status = WriteBinaryProto(tensorflow::Env::Default(), fgraph, m_graphdef);
    if(status != Status::OK())
    {
        return status;
    }

}

Status CNAIDNNNet::save(const char *folder, const char *filename)
{
    if(m_innerWeights.size() == 0)
    {
        printf("[ERR] Save weight impossible, there is no inner weight here\n");
        exit(-1);
    }

    std::string sf(folder);
    std::string f(filename);
    std::string fgraph = sf + "/" + f + ".pb";
    std::string fullpath = sf + "/" + f + ".ckp";
    Status status = WriteBinaryProto(tensorflow::Env::Default(), fgraph, m_graphdef);
    if(status != Status::OK())
    {
        printf("[ERR] Save Graph file : %s\n", status.error_message());
        return status;
    }

    /* Save all weights */
    FILE *file = fopen(fullpath.c_str(), "w");
    if(file == NULL)
    {
        printf("[ERR] Save weight file is NULL: %s\n", fullpath.c_str());
        exit(-1);
    }

    int len = m_innerWeights.size();
    fwrite(&len, sizeof(int), 1, file);

    for(int i = 0; i < len; i ++)
    {
        Tensor &t = m_innerWeights[i];
        auto flat = t.flat<float>();

        int dims = t.dims();
        fwrite(&(dims), sizeof(int), 1, file);

        int alllen = 1;
        for(int j = 0; j < dims; j ++)
        {
            int d = t.dim_size(j);
            alllen = alllen * d;
            fwrite(&d, sizeof(int), 1, file);
        }

        for(int j = 0; j < alllen; j ++)
        {
            float value = flat(j);
            fwrite(&value, sizeof(float), 1, file);
        }
    }
    fflush(file);
    fclose(file);

    return status;
}

Status CNAIDNNNet::load(const char *fullname)
{
    Status status = ReadBinaryProto(tensorflow::Env::Default(), string(fullname) + string(".pb"), &m_graphdef);

    if (m_options.target.empty())
    {
        graph::SetDefaultDevice( "/device:GPU:0" , &m_graphdef);
    }
    TF_CHECK_OK(m_session->Create(m_graphdef));

    /* Load all weights names */
    m_innerWeightNamesIn.clear();
    FILE *fw = fopen((std::string(fullname) + string(".wns")).c_str(), "r");

    /* Load all weights */
    FILE *file = fopen((std::string(fullname) + string(".ckp")).c_str(), "r");
    if(file == NULL)
    {
        printf("[ERR] Load weight file is NULL: %s\n", fullname);
        exit(-1);
    }

    m_innerWeights.clear();

    int len = 0;
    fread(&len, sizeof(int), 1, file);



    for(int i = 0; i < len; i ++)
    {
        char line[256] = {0};
        char name[255] = {0};
        fgets(line, 256, fw);
        memcpy(name, line, strlen(line) - 1);
        m_innerWeightNamesIn.push_back(string(name));

        TensorShape ts;
        int dims = 0;
        fread(&dims, sizeof(int), 1, file);
        int dim_sizes[dims];

        int alllen = 1;
        for(int j = 0; j < dims; j ++)
        {
            int ds = 0;
            fread(&ds, sizeof(int), 1, file);
            dim_sizes[j] = ds;
            ts.AddDim(ds);
            alllen = alllen * ds;
        }

        Tensor t(DT_FLOAT, ts);
        auto flat = t.flat<float>();

        for(int j = 0; j < alllen; j ++)
        {
            float value = 0.0f;
            fread(&value, sizeof(float), 1, file);
            flat(j) = value;
        }
        m_innerWeights.push_back(t);
    }

    fclose(file);
    fclose(fw);
    return status;
}

void CNAIDNNNet::addOutputFilter(std::string keyword)
{
    std::vector<std::string> oNames;
    fillOutputNames(keyword, oNames);
    appendOutputNames(oNames);
}
void CNAIDNNNet::addOutputFilter(std::vector<std::string> keys)
{
    std::vector<std::string> oNames;
    fillOutputNames(keys, oNames);
    appendOutputNames(oNames);
}

int CNAIDNNNet::fillOutputNames(std::vector<std::string> keys, std::vector<std::string> &outputNames)
{
    int siz = outputNames.size();
    int nc = m_graphdef.node_size();
    for (int i = 0; i < nc; i++) {
        auto n = m_graphdef.node(i);

        int find = 1;
        for(int j = 0; j < keys.size(); j ++)
        {
            if (n.name().find(keys[j].c_str()) != std::string::npos) {
                find = 1;
            }
            else
            {
                find = 0;
                break;
            }
        }
        if(find == 1)
        {
            outputNames.push_back(n.name());
        }

    }
    return outputNames.size() - siz;
}

int CNAIDNNNet::fillOutputNames(std::string keyword,
                                std::vector<std::string> &outputNames)
{
    int siz = outputNames.size();
    int nc = m_graphdef.node_size();
    for (int i = 0; i < nc; i++) {
        auto n = m_graphdef.node(i);

        if (n.name().find(keyword.c_str()) != std::string::npos) {
            outputNames.push_back(n.name());
        }
    }
    return outputNames.size() - siz;
}

void CNAIDNNNet::updateInnerWeight(std::vector<string> names, std::vector<Tensor> outputs)
{
    m_innerWeights.clear();
    for(int i = 0; i < names.size(); i ++)
    {
        string str = __CNAI_WOUTPUT;
        if(names[i].find(str) != std::string::npos)
        {
            m_innerWeights.push_back(outputs[i]);
        }
    }
}

void CNAIDNNNet::initWeights(Tensor (*genWeight)(int, int, int) )
{
    m_innerWeights.clear();
    for(int i = 0; i < layerNumber(); i ++)
    {
        CNAIDNNLayer *l = getLayer(i);
        for(int j = 0; j < l->numWeights(); j ++)
        {
            int leftNodes = l->left()->numNodes();
            Tensor t = genWeight(i, leftNodes, l->numNodes());
            m_innerWeights.push_back(t);
        }
    }
}

void CNAIDNNNet::appendOutputNames(std::vector<std::string> outputNames)
{
    for(int i = 0; i < outputNames.size(); i ++)
    {
        m_outputUserNames.push_back(outputNames[i]);
    }
}
void CNAIDNNNet::__updateInnerWeightNames()
{
    m_innerWeightNamesIn.clear();
    for(int i = 0; i < layerNumber(); i ++)
    {
        CNAIDNNLayer *l = getLayer(i);
        for(int j = 0; j < l->numWeights(); j ++)
        {
            CNAIDNNWeight *w = l->weight(j);
            m_innerWeightNamesIn.push_back(w->inputName());
            m_innerWeightNamesOut.push_back(w->outputName());
        }
    }
}
