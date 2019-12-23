#ifndef CNAI_DNN_LAYER_EXT_H
#define CNAI_DNN_LAYER_EXT_H

#include "cnai_dnn_layer.h"

class CNAIDNNLayerExt : public CNAIDNNLayer
{
public:
    CNAIDNNLayerExt(Scope &scope,
                    std::string net,
                    std::string layer,
                    int nodes,
                    CNAIDNN_LAYERTYPE type,
                    CNAIDNN_ACT_TYPE ntype);

protected:
    /**
     * @Override
     * Override constructNodes function
     */
    int constructNodes(Scope &scope);
};

#endif // CNAI_DNN_LAYER_EXT_H
