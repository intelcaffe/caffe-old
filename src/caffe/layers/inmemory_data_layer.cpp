#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
InMemoryDataLayer<Dtype>::~InMemoryDataLayer<Dtype>() {
	memDb_.clear();
}

template <typename Dtype>
void InMemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  prefetchOn_ = this->layer_param_.data_param().prefetch(); //true;
  shuffle_ = this->layer_param_.data_param().shuffle();    //false;
  //  ----  copy DB into vector memDB_ --------------------------
  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
  db_->Open(this->layer_param_.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
  LOG(INFO) << "Copy DB into memory: start... ";
  CPUTimer timer;
  timer.Start();
  memDb_.clear();
  num_entries_ = 0;
  while ( cursor_->valid()) {
     // get datum
     string entry=cursor_->value();
     memDb_.push_back(entry);
     cursor_->Next();
     num_entries_ ++;
   }
  timer.Stop();
  LOG(INFO) << "Copy DB into memory completed in " << timer.MilliSeconds()
		    << " ms.";
  LOG(INFO) << "Number of entries in DB " << num_entries_;
  if (shuffle_) Shuffle();
  //db_->Close();

  // -- init buffers --------------------------------------------------

  current_ = 0;
  // Check if we should randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    current_= skip % num_entries_;
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(memDb_[current_]);

  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if ((force_color && DecodeDatum(&datum, true)) ||
      DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }
  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    top[0]->Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
    this->transformed_data_.Reshape(1, datum.channels(), crop_size, crop_size);
  } else {
    top[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
    this->transformed_data_.Reshape(1, datum.channels(),
      datum.height(), datum.width());
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, this->layer_param_.data_param().batch_size());
    top[1]->Reshape(label_shape);
    this->prefetch_label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void InMemoryDataLayer<Dtype>::Shuffle() {
	std::srand( unsigned(std::time(0)) );
	std::random_shuffle( memDb_.begin(), memDb_.end() );
}

template <typename Dtype>
void InMemoryDataLayer<Dtype>::SetCursor(size_t cursor){
  if (num_entries_> 0) {
	current_ = cursor % num_entries_;
  }  else {
	current_ = 0;
	LOG(WARNING) << "InMemoryDataLayer::SetCursor Error";
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void InMemoryDataLayer<Dtype>::InternalThreadEntry() {
  if (!prefetchOn_)
	return;

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if (batch_size == 1 && crop_size == 0) {
    Datum datum;
    datum.ParseFromString(memDb_[current_]);
    if (datum.encoded()) {
      if (force_color) {
        DecodeDatum(&datum, true);
      } else {
        DecodeDatumNative(&datum);
      }
    }
    this->prefetch_data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
    this->transformed_data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
  }

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a blob
    Datum datum;
    datum.ParseFromString(memDb_[current_]);

    cv::Mat cv_img;
    if (datum.encoded()) {
      if (force_color) {
        cv_img = DecodeDatumToCVMat(datum, true);
      } else {
        cv_img = DecodeDatumToCVMatNative(datum);
      }
      if (cv_img.channels() != this->transformed_data_.channels()) {
        LOG(WARNING) << "Your dataset contains encoded images with mixed "
        << "channel sizes. Consider adding a 'force_color' flag to the "
        << "model definition, or rebuild your dataset using "
        << "convert_imageset.";
      }
    }
    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply data transformations (mirror, scale, crop...)
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (datum.encoded()) {
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    } else {
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();
    // go to the next iter
    current_++;
    if (current_ >= num_entries_ ) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      current_ = 0;
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void InMemoryDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  if (prefetchOn_){
    BasePrefetchingDataLayer<Dtype>::Forward_cpu(bottom, top);
  } else {
	// -------- no prefetching thread ----------------------------------
    CPUTimer batch_timer;
    batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;

	const int batch_size = this->layer_param_.data_param().batch_size();
	const int crop_size = this->layer_param_.transform_param().crop_size();
	bool force_color = this->layer_param_.data_param().force_encoded_color();

	if (batch_size == 1 && crop_size == 0) {
	  Datum datum;
	  datum.ParseFromString(memDb_[current_]);
	  if (datum.encoded()) {
	    if (force_color) {
	      DecodeDatum(&datum, true);
	    } else {
	      DecodeDatumNative(&datum);
	    }
	  }
	  this->prefetch_data_.Reshape(1, datum.channels(),
	      datum.height(), datum.width());
	  this->transformed_data_.Reshape(1, datum.channels(),
	      datum.height(), datum.width());
    }

	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
    if (this->output_labels_) {
	  top_label = this->prefetch_label_.mutable_cpu_data();
	}
	for (int item_id = 0; item_id < batch_size; ++item_id) {
	  timer.Start();
	  // get a blob
	  Datum datum;
	  datum.ParseFromString(memDb_[current_]);

	  cv::Mat cv_img;
	  if (datum.encoded()) {
	    if (force_color) {
	      cv_img = DecodeDatumToCVMat(datum, true);
	    } else {
	      cv_img = DecodeDatumToCVMatNative(datum);
        }
	    if (cv_img.channels() != this->transformed_data_.channels()) {
	      LOG(WARNING) << "Your dataset contains encoded images with mixed "
	        << "channel sizes. Consider adding a 'force_color' flag to the "
	        << "model definition, or rebuild your dataset using "
	        << "convert_imageset.";
	    }
	  }
	  read_time += timer.MicroSeconds();
	  timer.Start();

	  // Apply data transformations (mirror, scale, crop...)
	  int offset = this->prefetch_data_.offset(item_id);
	  this->transformed_data_.set_cpu_data(top_data + offset);
	  if (datum.encoded()) {
	    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
	  } else {
	    this->data_transformer_->Transform(datum, &(this->transformed_data_));
	  }
	  if (this->output_labels_) {
	    top_label[item_id] = datum.label();
	  }
	  trans_time += timer.MicroSeconds();
	  // go to the next iter
	  current_++;
	  if (current_ >= num_entries_ ) {
	    DLOG(INFO) << "Restarting data prefetching from start.";
	    current_ = 0;
	  }
	}

	batch_timer.Stop();
	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

	top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
	           this->prefetch_data_.height(), this->prefetch_data_.width());
	// Copy the data
    caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
					 top[0]->mutable_cpu_data());
	DLOG(INFO) << "Prefetch copied";
	if (this->output_labels_) {
	    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
				   top[1]->mutable_cpu_data());
	}
  }

}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(InMemoryDataLayer, Forward);
#else
template <typename Dtype>
void InMemoryDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetchOn_){
    BasePrefetchingDataLayer<Dtype>::Forward_gpu(bottom, top);
  } else {

    Forward_cpu(bottom, top);
    top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
    // Copy the data
    caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
    if (this->output_labels_) {
      caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
    }
  }
}
INSTANTIATE_LAYER_GPU_FORWARD(InMemoryDataLayer);
#endif

INSTANTIATE_CLASS(InMemoryDataLayer);
REGISTER_LAYER_CLASS(InMemoryData);

}  // namespace caffe
