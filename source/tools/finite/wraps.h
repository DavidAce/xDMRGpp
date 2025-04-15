#pragma once

#include <memory>
#include <string_view>

template<typename Scalar> class StateFinite;
template<typename Scalar> class ModelFinite;
template<typename Scalar> class EdgesFinite;
class AlgorithmStatus;
namespace h5pp {
    class File;
}

namespace tools::finite {
    class mwrap {
        // Mutable wrapper
        public:
        StateFinite     &state;
        ModelFinite     &model;
        EdgesFinite     &edges;
        AlgorithmStatus &status;
        h5pp::File      &file;
        mwrap(StateFinite &state, ModelFinite &model, EdgesFinite &edges, AlgorithmStatus &status, h5pp::File &file)
            : state(state), model(model), edges(edges), status(status), file(file){};
    };

    class cwrap {
        // Const wrapper
        public:
        const StateFinite     &state;
        const ModelFinite     &model;
        const EdgesFinite     &edges;
        const AlgorithmStatus &status;
        const h5pp::File      &file;
        cwrap(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges, const AlgorithmStatus &status, const h5pp::File &file)
            : state(state), model(model), edges(edges), status(status), file(file){};
    };

    class wwrap {
        // Writeable wrapper
        public:
        const StateFinite     &state;
        const ModelFinite     &model;
        const EdgesFinite     &edges;
        const AlgorithmStatus &status;
        h5pp::File            &file;
        wwrap(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges, const AlgorithmStatus &status, h5pp::File &file)
            : state(state), model(model), edges(edges), status(status), file(file){};
    };

}
