#include "tfdetect.h"
#include "tfwrapper.h"

#include <memory>
#include <functional>


//------------------------------------------------------------------------------

namespace tfdetect
{
namespace
{

//namespace tf = tensorflow;

class GraphProtoDetector : public Detector
{
public:
  GraphProtoDetector(const std::string &path_to_graph_proto) : graph_(new tfwrapper::Graph())
  {
      tfwrapper::Buffer graph_buffer(path_to_graph_proto);
      tfwrapper::ImportGraphDefOptions opts;
      graph_->ImportGraphDef(graph_buffer, opts);
      session_ = std::move(std::unique_ptr<tfwrapper::Session>(new tfwrapper::Session(*graph_)));
  }

  virtual ~GraphProtoDetector()
  {
    if (session_)
    {
      session_->Close();
    }
  }

  virtual void detect(const cv::Mat &input_image, std::vector<Detection> &results) const throw(std::exception) override
  {
    cv::Mat converted_image;
    input_image.convertTo(converted_image, CV_8UC3);

    std::vector<TF_Output> input_names(1);
    graph_->GetOperation("image_tensor").Output(0, input_names.at(0));

    tfwrapper::Tensor image_tensor(input_image);
    tfwrapper::ref_vector<tfwrapper::Tensor> input_tensors{image_tensor};

    std::vector<TF_Output> output_names(3);
    graph_->GetOperation("detection_scores").Output(0, output_names.at(0));
    graph_->GetOperation("detection_boxes").Output(0, output_names.at(1));
    graph_->GetOperation("detection_classes").Output(0, output_names.at(2));

    std::vector<std::shared_ptr<tfwrapper::Tensor>> result_tensors;
    session_->Run(input_names, input_tensors, output_names, result_tensors);

    const auto output_scores = result_tensors[0]->View<float, 2>();
    const auto output_boxes = result_tensors[1]->View<float, 3>();
    const auto output_classes = result_tensors[2]->View<float, 2>();

    results.clear();
    for (size_t i = 0; i < output_scores.NumElements(); ++i)
    {
      if (output_scores({0, i}) > 0.)
      {
        results.emplace_back(output_classes({0, i}),
                             output_scores({0, i}),
                             output_boxes({0, i, 1}),
                             output_boxes({0, i, 0}),
                             output_boxes({0, i, 3}),
                             output_boxes({0, i, 2}));
      }
    }
  }


private:
//  std::unique_ptr<tf::Session> session_;
    std::unique_ptr<tfwrapper::Graph> graph_;
    std::unique_ptr<tfwrapper::Session> session_;
};

} // namespace

std::unique_ptr<Detector> CreateDetectorFromGraph(const std::string &path_to_graph_proto)
{
  return std::unique_ptr<GraphProtoDetector>(new GraphProtoDetector(path_to_graph_proto));
}

} // namespace tf_detector
