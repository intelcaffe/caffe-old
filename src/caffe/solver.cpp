#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>
#include<iomanip>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"


namespace caffe {

// ------------------------------------------------------------------
// quadratic approximation y[i] ~ a*x[i]^2 + b*x[i] + c
// Rsquare measures how good is approximation

template <typename Dtype>
void Solver<Dtype>::QuadraticEstimate(vector<Dtype> & x, vector<Dtype> & y,
		Dtype & a, Dtype & b, Dtype & c , Dtype & Rsquare) {
  if ( (x.size()!=y.size()) or (x.size()==0) )	{
	LOG(ERROR) << " ERROR: QuadraticEstimate: input size = 0" ;
	return ;
  }
  // auxiliary  calculations
  Dtype x_4 = .0;
  Dtype x_3 = .0;
  Dtype x_2 = .0;
  Dtype x_1 = .0;
  Dtype x_0 = x.size();
  Dtype y_1 = .0;
  Dtype y_x_1 = .0;
  Dtype y_x_2 = .0;
  for (int i = 0; i < x.size(); ++i) {
    x_4 += x[i] * x[i] * x[i] * x[i];
    x_3 += x[i] * x[i] * x[i];
    x_2 += x[i] * x[i];
    x_1 += x[i];
    y_1 += y[i];
    y_x_1 += y[i] * x[i];
    y_x_2 += y[i] * x[i] * x[i];
  }

  Dtype det = x_4*(x_2*x_0 - x_1*x_1) - x_3*(x_3*x_0 - x_1*x_2) + x_2*(x_3*x_1 - x_2*x_2);
  Dtype d0 = y_x_2*(x_2*x_0 - x_1*x_1) - y_x_1*(x_3*x_0 - x_1*x_2) + y_1*(x_3*x_1 - x_2*x_2);
  Dtype d1 = x_4*(y_x_1*x_0 - y_1*x_1) - x_3 * (y_x_2*x_0 - y_1*x_2) + x_2*(y_x_2*x_1 - y_x_1*x_2);
  Dtype d2 = x_4*(x_2*y_1 - x_1*y_x_1) - x_3 * (x_3*y_1 - x_1*y_x_2) + x_2*(x_3*y_x_1 - x_2*y_x_2);
  a = b = c  =.0;
  if (det != 0.0) {
    a = d0/det;
    b = d1/det;
    c = d2/det;
  }
  else
	 LOG(INFO) << "LINE SEARCH ERROR: det = 0";

  // R^2 calculation
  Dtype mean = 0.0;
  for (int i = 0; i < x.size(); ++i) {
	mean += y[i];
  }
  mean = mean/x.size();
  Dtype error = 0.0;
  Dtype distance_from_mean = 0.0;
  for (int i = 0; i < x.size(); ++i) {
	Dtype tmp = y[i] - a * x[i] * x[i] - b * x[i] - c ;
	Dtype tmp2 = mean - y[i];
    error += tmp*tmp;
	distance_from_mean += tmp2*tmp2;
  }
  Rsquare = 0.0;
  if (distance_from_mean!=0.)
   Rsquare = 1.0 - (error/distance_from_mean);
//   LOG(INFO) << "LINE SEARCH: a = " << a << " b = " << b << " c = " << c
//            << "Rsquare = " << Rsquare;
}

//----------------------------------------------------------------------------
template <typename Dtype>
void Solver<Dtype>::InitLineSearch() {
  Dtype alpha_min_ = param_.ls_param().alpha_min();
  Dtype alpha_max_ = param_.ls_param().alpha_max();
  Dtype alpha_step_ = param_.ls_param().alpha_step();

  CHECK((alpha_min_ <  alpha_max_) && (alpha_step_ >0.))
	 << "Wrong line search alpha parameters";
  ls_alphas_.clear();
  ls_loss_.clear();
  for (Dtype alpha = alpha_min_; alpha <= alpha_max_ + 0.0001 ; alpha +=alpha_step_) {
	ls_alphas_.push_back(alpha);
	ls_loss_.push_back(0.);
  }
  //prepare ls buffers
  const vector<shared_ptr<Blob<Dtype> > >& net_params = net_->params(); //this->net_->params();
  ls_history_.clear();
  ls_temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    ls_history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    ls_temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
  SaveWeights(ls_history_);
  ls_lr_ = param_.base_lr() ;
  LOG(INFO) << "LINE SEARCH initialized at "<< iter_ ;
}

//=============================================================================
template <typename Dtype>
void Solver<Dtype>::ClearLineSearch() {
  ls_alphas_.clear();
  ls_loss_.clear();
  for (int i=0; i < ls_history_.size(); i++) {
	  ls_history_[i].reset();
	  ls_temp_[i].reset();
  }
  ls_history_.clear();
  ls_temp_.clear();
}

//-------------------------------------------------------------------
//  1. combine weights0 and weights1
//  2. load combination into solver
//  3. test loss functions
//-------------------------------------------------------------------

template <typename Dtype>
void Solver<Dtype>::TestLineInterval(
		const vector<shared_ptr<Blob<Dtype> > >& weights0,
		const vector<shared_ptr<Blob<Dtype> > >& weights1)  {
  //  SaveWeights(ls_temp_);
  LOG(INFO) << "Starting linear combination ";
  const vector<shared_ptr<Blob<Dtype> > >& net_params = net_->params();
  CHECK(weights0.size() == weights1.size() &&
		net_params.size() == weights0.size() ) << "Weights size mismatch";
  ls_loss_.clear();
  int n;
  int numAlphas=ls_alphas_.size();
  for (int it=0; it < numAlphas; it ++) {
	Dtype alpha = ls_alphas_[it];
    // --- combine weights ----------------------
	for (int i = 0; i < net_params.size(); ++i) {
	  n = net_params[i]->count();
	  CHECK((weights0[i]->count() == n) && (weights1[i]->count() == n) ) << "Layer mismatch";
      switch (Caffe::mode()) {
        case Caffe::CPU:
         caffe::caffe_copy(n, weights0[i]->cpu_data(), net_params[i]->mutable_cpu_data());
          caffe::caffe_cpu_axpby(n, alpha, weights1[i]->cpu_data(),
        		           Dtype (1.0 - alpha), net_params[i]->mutable_cpu_data());
          break;
        case Caffe::GPU:
          caffe::caffe_copy(n,weights0[i]->gpu_data(), net_params[i]->mutable_gpu_data());
          caffe::caffe_gpu_axpby(n, alpha, weights1[i]->gpu_data(),
        		  Dtype (1.0 - alpha), net_params[i]->mutable_gpu_data());
          break;
      }
    }
	// ----- compute loss --------------------------------------------
	Dtype loss = LS_Loss( param_.ls_param().ls_net_id());
	
	// -----compute weights^2----------------------------------------------------------
	Dtype netNorm2 = 0.;
	Dtype layerNorm2 =0.;
	for (int i = 0; i < net_params.size(); ++i) {
    int n = net_params[i]->count();
    switch (Caffe::mode()) {
      case Caffe::CPU:
        layerNorm2 = caffe::caffe_cpu_dot(n, net_params[i]->mutable_cpu_data(),
      		  net_params[i]->mutable_cpu_data());
        break;
      case Caffe::GPU:
         caffe::caffe_gpu_dot(n, net_params[i]->mutable_gpu_data(),
      		   net_params[i]->mutable_gpu_data(), &layerNorm2);
        break;
      }
      netNorm2+=layerNorm2;
  }
  loss += 0.5 * param_.weight_decay() * netNorm2;
	//----------------------------------------------------------------
	LOG(INFO) << "Linear combination: iteration " << iter_
			  << " alpha = " << std::fixed << std::setprecision(2)<< alpha
			  << " loss = " << std::fixed << std::setprecision(8) << loss;
	ls_loss_.push_back(loss);
  }

//  LoadWeights(ls_temp_);
}

// --------------------------------------------------------

template <typename Dtype>
void Solver<Dtype>::LineSearch() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params=net_->params();
  SaveWeights(ls_temp_);
 // Dtype diffW = DiffWeights(ls_history_,ls_temp_);

  //compute LS_Loss on interval between previous heckPoint (ls_history) and current weights;
  TestLineInterval(ls_history_,ls_temp_);

  Dtype a, b, c, Rsquare;
  QuadraticEstimate(ls_alphas_,ls_loss_, a, b, c, Rsquare);

  Dtype alphaOpt  = 1.0;
  if ( (Rsquare > LS_RSQUARE_THRESHOLD) && (a > LS_A_THRESHOLD) )  {
      alphaOpt =  (-b/(2.*a));
      // if ( alphaOpt > param_.ls_param().alpha_max() )
      // 	alphaOpt = param_.ls_param().alpha_max();
      // if ( alphaOpt < param_.ls_param().alpha_min() )
      // 	alphaOpt = param_.ls_param().alpha_min();
  }

  //--- merge --------------------------------------------
  if ( param_.ls_param().merge() && (alphaOpt >1. )) {
    for (int i = 0; i < net_params.size(); ++i) {
      int n = net_params[i]->count();
      switch (Caffe::mode()) {
      case Caffe::CPU:
        caffe::caffe_cpu_axpby(n, alphaOpt, ls_temp_[i]->cpu_data(),
      		      Dtype (1.0 - alphaOpt), ls_history_[i]->mutable_cpu_data());
        break;
      case Caffe::GPU:
        caffe::caffe_gpu_axpby(n, alphaOpt, ls_temp_[i]->gpu_data(),
    		      Dtype(1.0 - alphaOpt), ls_history_[i]->mutable_gpu_data());
        break;
      }
    }
    LoadWeights(ls_history_);
  }
  else {
	  // restore current solver weights from ls_temp
    LoadWeights(ls_temp_);
	  // update ls_history
	  SaveWeights(ls_history_);
  }

  //-----------------------------------------------------
  Dtype ls_loss_after = LS_Loss( param_.ls_param().ls_net_id() );
  Dtype test_loss_after = LS_Loss(0);

  //------ update learning rate --------------------------
  float old_lr = param_.base_lr() ;
  float old_moment=param_.momentum();
  float new_lr  =  old_lr;
  float new_moment = old_moment;
  if ( param_.ls_param().alrc()) {
    
    float alrc_factor = sqrt(2.); //sqrt((sqrt(2.)));   
    
    if ( (Rsquare > LS_RSQUARE_THRESHOLD) && (a > LS_A_THRESHOLD) ){
    	// new_lr = old_lr * alphaOpt;
      if (alphaOpt < 0.77) {
        new_lr = old_lr / alrc_factor;         
      } else if ((alphaOpt > 3.41) ) {   //&& (alphaOpt < 5.0)) {
	      new_lr = old_lr * alrc_factor ;
	    }
	      // else if (alphaOpt > 5.0) {
	      // new_lr = old_lr * alrc_factor; 
	      // } 
    } else if (Rsquare < LS_RSQUARE_THRESHOLD) {
	    //new_lr = old_lr / alrc_factor;
    } else if (a < LS_A_THRESHOLD) {
	    //  new_lr = old_lr * alrc_factor;
    }
    new_lr = std::max((float)LS_LR_MIN, new_lr);
    new_lr = std::min((float)LS_LR_MAX, new_lr);
    param_.set_base_lr(new_lr);
    //------ update moment --------------------------
    if ( param_.ls_param().amc()) {
      if (new_lr < 1.) {
       new_moment = (1-sqrt(new_lr))*(1-sqrt(new_lr));
      }
      param_.set_momentum(new_moment);
    }
  }

  LOG(INFO) << "LINE SEARCH: iteration " << iter_
		    << " a = " << std::fixed << std::setprecision(8) << a
		    << " b = " << std::fixed << std::setprecision(8) << b
		    << " c = " << std::fixed << std::setprecision(8) << c
		    << " Rsquare = " << std::fixed << std::setprecision(8) << Rsquare
		    << " alpha_opt = " << std::fixed << std::setprecision(8) << alphaOpt
		    << " test loss after = " << std::fixed << std::setprecision(8) << test_loss_after
		    << " ls loss after = "  << std::fixed << std::setprecision(8) << ls_loss_after
		    << " new_lr = " << std::fixed << std::setprecision(8) << new_lr
		    << " new_moment = " << std::fixed << std::setprecision(8) << new_moment;
}

//---------------------------------------------------------

template <typename Dtype>
void Solver<Dtype>::SaveWeights(vector<shared_ptr<Blob<Dtype> > > & weights) {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = net_->params();
  //   netWeights.clear();
  //  for (int i = 0; i < net_params.size(); ++i) {
  //    const vector<int>& shape = net_params[i]->shape();
  //    netWeights.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  //  }

  if ( net_params.size()!= weights.size()) {
	LOG(ERROR) << "ERROR in Solver::SaveWeigths " ;
    return;
  } else {
  for (int i = 0; i < net_params.size(); ++i) {
    int n = net_params[i]->count();
    if (weights[i]->count()!=n){
      LOG(ERROR) << "ERROR in Solver::SaveWeigths " ;
      return ;
    } else {
    if (Caffe::mode()==Caffe::CPU)
      caffe::caffe_copy(n, net_params[i]->cpu_data(),
    		            weights[i]->mutable_cpu_data());
    else // Caffe::mode()==Caffe::GPU
      caffe::caffe_copy(n, net_params[i]->gpu_data(),
    		            weights[i]->mutable_gpu_data());
    }
  } // end of for
 }
}


//---------------------------------------------------------

template <typename Dtype>
void Solver<Dtype>::LoadWeights(
		const vector<shared_ptr<Blob<Dtype> > >   & weights) {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = net_->params();
  int n;
  if ( net_params.size()!= weights.size()) {
    LOG(ERROR) << "ERROR in Solver::SaveWeigths " ;
	return;
  } else {
  for (int i = 0; i < net_params.size(); ++i) {
    n = net_params[i]->count();
    if (weights[i]->count()!=n){
      LOG(ERROR) << "ERROR in Solver::SaveWeigths " ;
      return ;
    } else {
    if (Caffe::mode()==Caffe::CPU)
      caffe::caffe_copy(n, weights[i]-> cpu_data(),
	                    net_params[i]-> mutable_cpu_data());
    else // Caffe::mode()==Caffe::GPU
      caffe::caffe_copy(n, weights[i]->gpu_data(),
            	        net_params[i]-> mutable_gpu_data());
    }
  } // end of for
 }
}

// ----------------------------------------------------------------------------

template <typename Dtype>
Dtype Solver<Dtype>::DiffWeights(
		const vector<shared_ptr<Blob<Dtype> > >& weights0,
		const vector<shared_ptr<Blob<Dtype> > >& weights1)  {
  Dtype diff =  0.;
    // --- combine weights ----------------------
  for (int i = 0; i < weights0.size(); ++i) {
	int n0 = weights0[i]->count();
	//int n1 = weights1[i]->count();
	CHECK(weights0[i]->count() == weights1[i]->count()) << "Layer mismatch";
    for ( int k = 0; k < n0; k++) {
   	  diff += (weights0[i]->cpu_data()[k] - weights1[i]->cpu_data()[k])*
    	      (weights0[i]->cpu_data()[k] - weights1[i]->cpu_data()[k]);
    }
    /*
    switch (Caffe::mode()) {
      case Caffe::CPU:
        for ( int k=0; k<n; k++) {
       	  diff += (weights0[i]->cpu_data()[k] - weights1[i]->cpu_data()[k])*
        	      (weights0[i]->cpu_data()[k] - weights1[i]->cpu_data()[k]);
        }
        break;
      case Caffe::GPU:
        for ( int k=0; k<n; k++) {
          diff += (weights0[i]->gpu_data()[k] - weights1[i]->gpu_data()[k])*
        	      (weights0[i]->gpu_data()[k] - weights1[i]->gpu_data()[k]);
        }
        break;
    }
    */
  }
  return(sqrt(diff));
}

//======== END OF LINE SEARCH ================================================

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_() {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_() {
  SolverParameter param;
  ReadProtoFromTextFileOrDie(param_file, &param);
  Init(param);
 }

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG(INFO) << "Initializing solver from parameters: " << std::endl
            << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  if (param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  InitTestNets();
  LOG(INFO) << "Solver scaffolding done.";
  iter_ = 0;
  current_step_ = 0;
  //-----------------------------------------------------
  // if (param_.ls_on())  InitLineSearch();

}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG(INFO) << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG(INFO) << "Creating training net from train_net file: "
              << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG(INFO) << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG(INFO) << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  net_.reset(new Net<Dtype>(net_param));
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  vector<Blob<Dtype>*> bottom_vec;
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;

  while (iter_ < stop_iter) {
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      TestAll();
    }

    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    Dtype loss = net_->ForwardBackward(bottom_vec);
    if (losses.size() < average_loss) {
      losses.push_back(loss);
      int size = losses.size();
      smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
    } else {
      int idx = (iter_ - start_iter) % average_loss;
      smoothed_loss += (loss - losses[idx]) / average_loss;
      losses[idx] = loss;
    }
    if (display) {
      LOG(INFO) << "Iteration " << iter_ << ", train loss = " << smoothed_loss;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG(INFO) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    ComputeUpdateValue();
    net_->Update();

    if (param_.ls_on()){
    	if (iter_ == param_.ls_param().ls_start()){
    		InitLineSearch();
    	}
    	else if ( (iter_ > param_.ls_param().ls_start() )
    			  && (iter_ % param_.ls_param().ls_interval() == 0 ) ) {
    		LineSearch();
    	}
    }

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    // Save a snapshot if needed.
    if (param_.snapshot() && iter_ % param_.snapshot() == 0) {
      Snapshot();
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    Dtype loss;
    net_->ForwardPrefilled(&loss);
    LOG(INFO) << "Iteration " << iter_ << ", train loss = " << loss;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}


template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
    Test(test_net_id);
  }
}


template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0.;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) <<"  Test net #" << test_net_id
        << " output #" << i << ": " << output_name << " = "
        << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
Dtype Solver<Dtype>::LS_Loss(const int test_net_id) {
  //LOG(INFO) << "LS_loss started" ;
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0.;
  Dtype retLoss =0.;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    Dtype iter_loss;
    //const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
/*    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
    */
  }

  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
//    LOG(INFO) << "Test loss: " << loss;
    retLoss = loss;
  }
  /*
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
      retLoss = mean_score;
    }
//    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
//        << mean_score << loss_msg_stream.str();
  }
  */
  return retLoss;
}


template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  NetParameter net_param;
  // For intermediate results, we will also dump the gradient values.
  net_->ToProto(&net_param, param_.snapshot_diff());
  string filename(param_.snapshot_prefix());
  string model_filename, snapshot_filename;
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d", iter_);
  filename += iter_str_buffer;
  model_filename = filename + ".caffemodel";
  LOG(INFO) << "Snapshotting to " << model_filename;
  WriteProtoToBinaryFile(net_param, model_filename.c_str());
  SolverState state;
  SnapshotSolverState(&state);
  state.set_iter(iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(current_step_);
  snapshot_filename = filename + ".solverstate";
  LOG(INFO) << "Snapshotting solver state to " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  SolverState state;
  NetParameter net_param;
  ReadProtoFromBinaryFile(state_file, &state);
  if (state.has_learned_net()) {
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    net_->CopyTrainedLayersFrom(net_param);
  }
  iter_ = state.iter();
  current_step_ = state.current_step();
  RestoreSolverState(state);
}


//------------------------------------------------------------------------

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }

  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    if (this->net_->param_owners()[i] < 0) {
      sumsq_diff += net_params[i]->sumsq_diff();
    }
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      if (this->net_->param_owners()[i] < 0) {
        net_params[i]->scale_diff(scale_factor);
      }
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  // get the learning rate
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << std::fixed << std::setprecision(8) << rate;
  }
  ClipGradients();
  Dtype momentum = this->param_.momentum();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else if (regularization_type == "L1") {
          caffe_cpu_sign(net_params[param_id]->count(),
              net_params[param_id]->cpu_data(),
              temp_[param_id]->mutable_cpu_data());
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              temp_[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->cpu_diff(), momentum,
                history_[param_id]->mutable_cpu_data());
      // copy
      caffe_copy(net_params[param_id]->count(),
          history_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else if (regularization_type == "L1") {
          caffe_gpu_sign(net_params[param_id]->count(),
              net_params[param_id]->gpu_data(),
              temp_[param_id]->mutable_gpu_data());
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              temp_[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->gpu_diff(), momentum,
                history_[param_id]->mutable_gpu_data());
      // copy
      caffe_copy(net_params[param_id]->count(),
          history_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  state->clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state->add_history();
    history_[i]->ToProto(history_blob);
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void NesterovSolver<Dtype>::ComputeUpdateValue() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  // get the learning rate
  Dtype rate = this->GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  SGDSolver<Dtype>::ClipGradients();
  Dtype momentum = this->param_.momentum();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // save history momentum for stepping back
      caffe_copy(net_params[param_id]->count(),
          this->history_[param_id]->cpu_data(),
          this->update_[param_id]->mutable_cpu_data());

      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else if (regularization_type == "L1") {
          caffe_cpu_sign(net_params[param_id]->count(),
              net_params[param_id]->cpu_data(),
              this->temp_[param_id]->mutable_cpu_data());
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              this->temp_[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      // update history
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->cpu_diff(), momentum,
                this->history_[param_id]->mutable_cpu_data());

      // compute udpate: step back then over step
      caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
          this->history_[param_id]->cpu_data(), -momentum,
          this->update_[param_id]->mutable_cpu_data());

      // copy
      caffe_copy(net_params[param_id]->count(),
          this->update_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // save history momentum for stepping back
      caffe_copy(net_params[param_id]->count(),
          this->history_[param_id]->gpu_data(),
          this->update_[param_id]->mutable_gpu_data());

      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else if (regularization_type == "L1") {
          caffe_gpu_sign(net_params[param_id]->count(),
              net_params[param_id]->gpu_data(),
              this->temp_[param_id]->mutable_gpu_data());
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              this->temp_[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      // update history
      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->gpu_diff(), momentum,
                this->history_[param_id]->mutable_gpu_data());

      // compute udpate: step back then over step
      caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
          this->history_[param_id]->gpu_data(), -momentum,
          this->update_[param_id]->mutable_gpu_data());

      // copy
      caffe_copy(net_params[param_id]->count(),
          this->update_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void AdaGradSolver<Dtype>::ComputeUpdateValue() {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  // get the learning rate
  Dtype rate = this->GetLearningRate();
  Dtype delta = this->param_.delta();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  SGDSolver<Dtype>::ClipGradients();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else if (regularization_type == "L1") {
          caffe_cpu_sign(net_params[param_id]->count(),
              net_params[param_id]->cpu_data(),
              this->temp_[param_id]->mutable_cpu_data());
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              this->temp_[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      // compute square of gradient in update
      caffe_powx(net_params[param_id]->count(),
          net_params[param_id]->cpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_cpu_data());

      // update history
      caffe_add(net_params[param_id]->count(),
          this->update_[param_id]->cpu_data(),
          this->history_[param_id]->cpu_data(),
          this->history_[param_id]->mutable_cpu_data());

      // prepare update
      caffe_powx(net_params[param_id]->count(),
                this->history_[param_id]->cpu_data(), Dtype(0.5),
                this->update_[param_id]->mutable_cpu_data());

      caffe_add_scalar(net_params[param_id]->count(),
                delta, this->update_[param_id]->mutable_cpu_data());

      caffe_div(net_params[param_id]->count(),
                net_params[param_id]->cpu_diff(),
                this->update_[param_id]->cpu_data(),
                this->update_[param_id]->mutable_cpu_data());

      // scale and copy
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
          this->update_[param_id]->cpu_data(), Dtype(0),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else if (regularization_type == "L1") {
          caffe_gpu_sign(net_params[param_id]->count(),
              net_params[param_id]->gpu_data(),
              this->temp_[param_id]->mutable_gpu_data());
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              this->temp_[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      // compute square of gradient in update
      caffe_gpu_powx(net_params[param_id]->count(),
          net_params[param_id]->gpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_gpu_data());

      // update history
      caffe_gpu_add(net_params[param_id]->count(),
          this->update_[param_id]->gpu_data(),
          this->history_[param_id]->gpu_data(),
          this->history_[param_id]->mutable_gpu_data());

      // prepare update
      caffe_gpu_powx(net_params[param_id]->count(),
                this->history_[param_id]->gpu_data(), Dtype(0.5),
                this->update_[param_id]->mutable_gpu_data());

      caffe_gpu_add_scalar(net_params[param_id]->count(),
                delta, this->update_[param_id]->mutable_gpu_data());

      caffe_gpu_div(net_params[param_id]->count(),
                net_params[param_id]->gpu_diff(),
                this->update_[param_id]->gpu_data(),
                this->update_[param_id]->mutable_gpu_data());

      // scale and copy
      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
          this->update_[param_id]->gpu_data(), Dtype(0),
          net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);
INSTANTIATE_CLASS(NesterovSolver);
INSTANTIATE_CLASS(AdaGradSolver);

}  // namespace caffe
