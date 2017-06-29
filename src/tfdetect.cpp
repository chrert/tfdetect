#include "tfdetect.h"
#include "tfwrapper.h"

#include <memory>
#include <functional>


//------------------------------------------------------------------------------

namespace tfdetect
{
namespace
{

class GraphProtoDetector : public Detector
{
public:
  GraphProtoDetector(const std::string &path_to_graph_proto) :
    graph_(new tfwrapper::Graph()),
    input_names_(1),
    output_names_(3)
  {
      // import graphdef and open session
      tfwrapper::Buffer graph_buffer(path_to_graph_proto);
      tfwrapper::ImportGraphDefOptions opts;
      graph_->ImportGraphDef(graph_buffer, opts);
      session_ = std::move(std::unique_ptr<tfwrapper::Session>(new tfwrapper::Session(*graph_)));

      // find the input placeholder
      graph_->GetOperation("image_tensor").Output(0, input_names_.at(0));

      // find the tensors we want to compute
      graph_->GetOperation("detection_scores").Output(0, output_names_.at(0));
      graph_->GetOperation("detection_boxes").Output(0, output_names_.at(1));
      graph_->GetOperation("detection_classes").Output(0, output_names_.at(2));
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
    // the graph expects images of type uint8
    cv::Mat im_to_use;
    if (input_image.depth() != CV_8U)
    {
        input_image.convertTo(im_to_use, CV_8UC3);
    }
    else
    {
        im_to_use = input_image;
    }

    // construct a tensor from the opencv mat (only a view)
    tfwrapper::Tensor image_tensor(im_to_use);
    tfwrapper::ref_vector<tfwrapper::Tensor> input_tensors{image_tensor};

    // execute the graph
    std::vector<std::shared_ptr<tfwrapper::Tensor>> result_tensors;
    session_->Run(input_names_, input_tensors, output_names_, result_tensors);

    // get views into the output tensors
    const auto output_scores = result_tensors[0]->View<float, 2>();
    const auto output_boxes = result_tensors[1]->View<float, 3>();
    const auto output_classes = result_tensors[2]->View<float, 2>();

    // copy detections to the results vector
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
    std::unique_ptr<tfwrapper::Graph> graph_;
    std::unique_ptr<tfwrapper::Session> session_;

    std::vector<TF_Output> input_names_;
    std::vector<TF_Output> output_names_;
};

} // namespace

std::unique_ptr<Detector> CreateDetectorFromGraph(const std::string &path_to_graph_proto)
{
  return std::unique_ptr<GraphProtoDetector>(new GraphProtoDetector(path_to_graph_proto));
}

} // namespace tf_detector
