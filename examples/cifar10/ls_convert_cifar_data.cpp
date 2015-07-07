//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <ctime>
#include <iostream>
#include <cstdlib>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *label = label_char;
  file->read(buffer, kCIFARImageNBytes);
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder,
    const string& db_type , const string& num_samples) {
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/cifar10_test_" + db_type +"_"+ num_samples, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  // Data buffer
  int label;
  char str_buffer[kCIFARImageNBytes];
  Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);

  std::vector<std::string> trainSet;

  LOG(INFO) << "Reaing Training data";
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    snprintf(str_buffer, kCIFARImageNBytes, "/data_batch_%d.bin", fileid + 1);
    std::ifstream data_file((input_folder + str_buffer).c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
      read_image(&data_file, &label, str_buffer);
      datum.set_label(label);
      datum.set_data(str_buffer, kCIFARImageNBytes);
//      int length = snprintf(str_buffer, kCIFARImageNBytes, "%05d",
//          fileid * kCIFARBatchSize + itemid);
      string out;
      CHECK(datum.SerializeToString(&out));
      trainSet.push_back(out);
//      txn->Put(string(str_buffer, length), out);
    }
  }
  // shuffle data set
  LOG(INFO) << "Shuffle images...";
  std::srand( unsigned(std::time(0)) );
  std::random_shuffle(trainSet.begin(), trainSet.end());

  size_t  maxNumImages = (size_t) atoi(num_samples.c_str());
  maxNumImages =std::min(maxNumImages, trainSet.size());

  LOG(INFO) << "Writing Training db: " << maxNumImages << " images";
  for (int itemid = 0; itemid < maxNumImages; itemid++) {
    int length = snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
    txn->Put(string(str_buffer, length), trainSet[itemid]);
  }
  txn->Commit();
  train_db->Close();
 //===============================================
 /*
  LOG(INFO) << "Writing Testing data";
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  test_db->Open(output_folder + "/cifar10_test_" + db_type, db::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    read_image(&data_file, &label, str_buffer);
    datum.set_label(label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    int length = snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(str_buffer, length), out);
  }
  txn->Commit();
  test_db->Close();
  */
}

int main(int argc, char** argv) {
  if ((argc !=5) ) {
    printf("This script converts the CIFAR dataset to the leveldb/lmdb format\n"
           "used by caffe to perform classification.\n"
           "Usage:\n"
           "  ls_convert_cifar_data input_folder output_folder db_type num_samples\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]), string(argv[4]));
  }
  return 0;
}
