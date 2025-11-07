
namespace Eigen {
    template<typename Scalar_, int NumIndices_, int Options_, typename IndexType_> class Tensor;

}

namespace tenx {
    template<typename Scalar_, int NumIndices_, int Options_ = 0, typename IndexType_ = long> using Tensor =
        Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_>;
}
