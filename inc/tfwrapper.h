#ifndef TFWRAPPER_H
#define TFWRAPPER_H

#include <fstream>
#include <functional>
#include <memory>

#include <tensorflow/c/c_api.h>

namespace tfwrapper
{

//------------------------------------------------------------------------------
// TYPEDEFS
//------------------------------------------------------------------------------

template<typename T>
using ref_vector = std::vector<std::reference_wrapper<T>>;

//------------------------------------------------------------------------------
// Wrapper classes for various structs and operations of the tensorflow C-API
//------------------------------------------------------------------------------

template<typename T, typename Constructor = std::function<T*(void)>, typename Destructor = void(*)(T*)>
class TFWrapper
{
public:
    TFWrapper(Constructor constructor, Destructor destructor) : c_obj_(constructor(), destructor) {}
    ~TFWrapper() = default;

    TFWrapper(const TFWrapper&) = delete;
    TFWrapper& operator=(const TFWrapper&) = delete;

    const T* TFObj() const
    {
        return c_obj_.get();
    }
    
    T* TFObj()
    {
        return c_obj_.get();
    }

private:
    std::unique_ptr<T, Destructor> c_obj_;
};

//------------------------------------------------------------------------------

class Status : public TFWrapper<TF_Status>
{
public:
    Status() : TFWrapper<TF_Status>(TF_NewStatus, TF_DeleteStatus) {}

    bool IsOk() const
    {
        return TF_GetCode(TFObj()) == TF_Code::TF_OK;
    }

    std::string Message() const
    {
        return std::string(TF_Message(TFObj()));
    }

    void ThrowRuntimeErrorIfNotOk() const throw(std::runtime_error)
    {
        if (!IsOk())
        {
            throw std::runtime_error(Message());
        }
    }
};

//------------------------------------------------------------------------------

class Buffer : public TFWrapper<TF_Buffer>
{
public:
    Buffer(uint8_t* bytes, size_t length, bool copy) : TFWrapper<TF_Buffer>(TF_NewBuffer, TF_DeleteBuffer), copy_(copy)
    {
        TFObj()->length = length;
        if (copy_)
        {
            uint8_t* copied = new uint8_t[length];
            std::copy_n(bytes, length, copied);
            TFObj()->data = copied;
        }
        else
        {
            TFObj()->data = bytes;
        }
    }

    Buffer(const std::string& filename) : TFWrapper<TF_Buffer>(TF_NewBuffer, TF_DeleteBuffer), copy_(true)
    {
        std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
        std::ifstream::pos_type pos = ifs.tellg();
        TFObj()->length = pos;
        char* bytes = new char[TFObj()->length];
        ifs.seekg(0, std::ios::beg);
        ifs.read(bytes, TFObj()->length);
        ifs.close();
        TFObj()->data = bytes;
    }

    virtual ~Buffer()
    {
        if (copy_)
        {
            delete[] static_cast<const uint8_t*>(TFObj()->data);
        }
    }

private:
    bool copy_;
};

//------------------------------------------------------------------------------

class Operation
{
public:
    Operation(TF_Operation* op) : op_(op) {}

    void Output(int index, TF_Output& output) const
    {
        output.index = index;
        output.oper = op_;
    }
    
    const TF_Operation* TFObj() const
    {
        return op_;
    }
    
    TF_Operation* TFObj()
    {
        return op_;
    }

private:
    TF_Operation* op_;
};

//------------------------------------------------------------------------------

class ImportGraphDefOptions : public TFWrapper<TF_ImportGraphDefOptions>
{
public:
    ImportGraphDefOptions() : TFWrapper<TF_ImportGraphDefOptions>(TF_NewImportGraphDefOptions, TF_DeleteImportGraphDefOptions) {}
    
    void AddControlDependency(Operation& operation)
    {
        TF_ImportGraphDefOptionsAddControlDependency(TFObj(), operation.TFObj());
    }
    
    void RemapControlDependency(const std::string& src_name, Operation& operation)
    {
        TF_ImportGraphDefOptionsRemapControlDependency(TFObj(), src_name.c_str(), operation.TFObj());
    }
    
    void AddInputMapping(const std::string& src_name, size_t src_index, const TF_Output& dst)
    {
        TF_ImportGraphDefOptionsAddInputMapping(TFObj(), src_name.c_str(), src_index, dst);
    }
    
    void AddReturnOutput(const std::string& src_name, size_t index)
    {
        TF_ImportGraphDefOptionsAddReturnOutput(TFObj(), src_name.c_str(), index);
    }
    
    size_t NumReturnOutputs() const 
    {
        return TF_ImportGraphDefOptionsNumReturnOutputs(TFObj());
    }
    
    void SetPrefix(const std::string& prefix)
    {
        TF_ImportGraphDefOptionsSetPrefix(TFObj(), prefix.c_str());
    }
    
};

//------------------------------------------------------------------------------

class Graph : public TFWrapper<TF_Graph>
{
public:
    Graph() : TFWrapper<TF_Graph>(TF_NewGraph, TF_DeleteGraph) {}

    void ImportGraphDef(Buffer &buffer, const ImportGraphDefOptions &opts) throw(std::exception)
    {
        Status status;
        TF_GraphImportGraphDef(TFObj(), buffer.TFObj(), opts.TFObj(), status.TFObj());
        status.ThrowRuntimeErrorIfNotOk();
    }

    Operation GetOperation(const std::string &name)
    {
        return Operation(TF_GraphOperationByName(TFObj(), name.c_str()));
    }
};

//------------------------------------------------------------------------------

namespace
{
// Construct a tensor from a single cv::Mat.
// Opencv will still be in charge of the deallocation process. This tensor will
// be valid as long as the data of the cv::Mat stays at the same location.
TF_Tensor* construct_tensor_from_cv_mat(const cv::Mat& input_image) throw (std::runtime_error)
{
    if (!input_image.isContinuous())
    {
        throw std::runtime_error("Can only work with continuous images!");
    }

    TF_DataType dtype;
    switch (input_image.depth())
    {
        case CV_8U:
            dtype = TF_DataType::TF_UINT8;
            break;
        case CV_8S:
            dtype = TF_DataType::TF_INT8;
            break;
        case CV_32F:
            dtype = TF_DataType::TF_FLOAT;
            break;
        default:
            throw std::runtime_error("Unsupported datatype!");
    }

    const int64_t dims[] = {1, input_image.rows, input_image.cols, input_image.channels()};
    size_t num_bytes = input_image.total() * input_image.elemSize();
    return TF_NewTensor(dtype,
                        dims, 4,
                        input_image.data, num_bytes,
                        [](void*, size_t, void*){}, &num_bytes);
}
}

class Tensor : public TFWrapper<TF_Tensor>
{
public:

    Tensor(const cv::Mat& input_image) throw(std::runtime_error)
        : TFWrapper<TF_Tensor>(std::bind(construct_tensor_from_cv_mat, input_image), TF_DeleteTensor)
    {}

    Tensor(TF_Tensor* tensor) : TFWrapper<TF_Tensor>([tensor]{return tensor;}, TF_DeleteTensor) {}
    
    // inner class for convenient access to tensors' data
    template<typename DType, size_t D>
    class TensorView
    {
    public:
        TensorView(Tensor& tensor) throw(std::runtime_error)
        {
            if (tensor.NumDims() != D)
            {
                throw std::runtime_error("Number of dimensions do not match!");
            }

            num_el_ = 1;
            for (size_t i = 0; i < D; ++i)
            {
                dims_[i] = tensor.Dim(i);
                num_el_ *= dims_[i];
            }

            if (tensor.NumBytes() != (num_el_ * sizeof(DType)))
            {
                throw std::runtime_error("Wrong TensorView!");
            }

            data_ = static_cast<DType*>(tensor.Bytes());
        }

        const DType& operator()(std::array<size_t, D> n) const
        {
            return data_[ComputeOffset(n)];
        }
        
        DType& operator()(std::array<size_t, D> n)
        {
            return data_[ComputeOffset(n)];
        }

        size_t NumElements() const
        {
            return num_el_;
        }

    private:
        DType* data_;
        std::array<size_t, D> dims_;
        size_t num_el_;
        
        size_t ComputeOffset(std::array<size_t, D> n) const
        {
            size_t offset = 0;
            for (size_t i = 0; i < D; ++i)
            {
                size_t N = 1;
                for (size_t j = i + 1; j < D; ++j)
                {
                    N *= dims_[j];
                }
                offset += N * n[i];
            }
            return offset;
        }
    };

    template<typename DType, size_t D>
    TensorView<DType, D> View()
    {
        return TensorView<DType, D>(*this);
    }

    size_t NumDims() const
    {
        return TF_NumDims(TFObj());
    }

    int64_t Dim(size_t dim_index) const
    {
        return TF_Dim(TFObj(), dim_index);
    }

    TF_DataType Type() const
    {
        return TF_TensorType(TFObj());
    }

    size_t NumBytes() const
    {
        return TF_TensorByteSize(TFObj());
    }

protected:
    const void* Bytes() const
    {
        return TF_TensorData(TFObj());
    }
    
    void* Bytes()
    {
        return TF_TensorData(TFObj());
    }
};


//------------------------------------------------------------------------------

class SessionOptions : public TFWrapper<TF_SessionOptions>
{
public:
    SessionOptions() : TFWrapper<TF_SessionOptions>(TF_NewSessionOptions, TF_DeleteSessionOptions) {}
};

//------------------------------------------------------------------------------

namespace
{
// construct a session with default (empty) options.
TF_Session* construct_session(TF_Graph* graph)
{
    // TODO: check status
    Status status;
    SessionOptions opts;
    return TF_NewSession(graph, opts.TFObj(), status.TFObj());
}

// delete a session ignoring the status
void delete_session(TF_Session *session)
{
    Status status;
    TF_DeleteSession(session, status.TFObj());
}
}

class Session : public TFWrapper<TF_Session>
{
public:
    Session(Graph &graph) : TFWrapper<TF_Session>(std::bind(construct_session, graph.TFObj()), delete_session) {}

    void Close() throw(std::exception)
    {
        Status status;
        TF_CloseSession(TFObj(), status.TFObj());
        status.ThrowRuntimeErrorIfNotOk();
    }

    void Run(const std::vector<TF_Output> input_names, const ref_vector<Tensor> input_tensors,
             const std::vector<TF_Output> output_names,
             std::vector<std::shared_ptr<Tensor>>& results) throw(std::exception)
    {
        if (input_names.size() != input_tensors.size())
        {
            std::runtime_error("input_names must be of same length as input_tensors!");
        }

        Status status;

        std::vector<TF_Tensor*> tf_input_tensors;
        tf_input_tensors.resize(input_tensors.size());
        std::transform(input_tensors.begin(), input_tensors.end(), tf_input_tensors.begin(),
                       [](Tensor& tensor) { return tensor.TFObj(); });


        std::vector<TF_Tensor*> tf_output_tensors;
        tf_output_tensors.resize(output_names.size());

        TF_SessionRun(TFObj(), nullptr,
                input_names.data(), tf_input_tensors.data(), input_names.size(),
                output_names.data(), tf_output_tensors.data(), output_names.size(),
                nullptr, 0,
                nullptr,
                status.TFObj());

        results.resize(tf_output_tensors.size());
        std::transform(tf_output_tensors.begin(), tf_output_tensors.end(), results.begin(),
                       [](TF_Tensor* tf_tensor) { return std::shared_ptr<Tensor>(new Tensor(tf_tensor)); });

        status.ThrowRuntimeErrorIfNotOk();
    }
};

} // namespace tfwrapper

#endif /* TFWRAPPER_H */
