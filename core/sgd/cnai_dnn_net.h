#ifndef CNAI_DNN_NET_H
#define CNAI_DNN_NET_H

#include "cnai_inc.h"
#include "cnai_dnn_layer.h"

class CNAIDNNNet
{
public:
    CNAIDNNNet(std::string net);

    /**
     * @brief begin
     *
     * Must be called before addLayer
     *
     * @return
     */
    Scope begin();

    /**
     * @brief addLayer
     * @param scope scope the same object return by begin() function
     * @param name layer name
     * @param type layer type
     * @param nodes how many node inside the layer
     * @param act
     *
     * Append layer to the network, from left one to right direction one by one
     */
    void addLayer(Scope &scope, string name, CNAIDNN_LAYERTYPE type, int nodes, CNAIDNN_ACT_TYPE act = CN_AT_Softplus);
    void addLayer(Scope &scope, string name, CNAIDNN_LAYERTYPE type, int nodes, __ACT_CB, __DER_CB);

    virtual GraphDef end(Scope &scope);

    void initWeights( Tensor (*genWeight)(int, int, int) );

    void train(std::vector< std::pair<string, Tensor> > &train_data,
               std::vector<string> &output_names,
               std::vector<Tensor> *outputs);

    void run(std::vector< std::pair<string, Tensor> > &run_data,
             std::vector<string> &output_names,
             std::vector<Tensor> *outputs);

    std::vector< std::pair<string, Tensor> > train(std::vector< std::pair<string, Tensor> > &train_data);
    std::vector< std::pair<string, Tensor> > run(std::vector< std::pair<string, Tensor> > &run_data);

    int fillOutputNames(std::vector<std::string> keys, std::vector<std::string> &outputNames);
    int fillOutputNames(std::string keyword, std::vector<std::string> &outputNames);
    void appendOutputNames(std::vector<std::string> outputNames);

    void addOutputFilter(std::string keyword);
    void addOutputFilter(std::vector<std::string> keys);

    Status saveWeightNames(const char *folder, const char *filename);
    Status saveGraph(const char *folder, const char *filename);

    Status save(const char *folder, const char *filename);
    Status load(const char *fullname);


    CNAIDNNLayer *getLayer(int i) { return m_layers[i]; }
    int layerNumber()   {   return m_layers.size(); }
    std::vector<Tensor> getWeights()  {     return m_innerWeights;  }

protected:
    /* For Override */
    virtual void updateInnerWeight(std::vector<tensorflow::string> names, std::vector<Tensor> outputs);

    void __updateInnerWeightNames();

    std::vector<Tensor> m_innerWeights;
    std::vector<string> m_innerWeightNamesIn;
    std::vector<string> m_innerWeightNamesOut;
    std::vector<string> m_outputUserNames;

    std::string m_name;
    std::vector<CNAIDNNLayer*> m_layers;
    GraphDef m_graphdef;

    SessionOptions m_options;
    std::unique_ptr<Session> m_session;

    int m_numAllWeights;
};

#endif // CNAI_DNN_NET_H
