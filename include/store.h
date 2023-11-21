#pragma once
// #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <ATen/ATen.h>
#include <Python.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <omp.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <utility>
#include <vector>

#include "gather.h"
#include "spdlog/spdlog.h"

const int64_t INVALID_PART_ID = -1;
bool fileExists(const std::string &filePath) {
  std::ifstream file(filePath);
  return file.good();
}

void createFileIfNotExists(const std::string &filePath) {
  if (!fileExists(filePath)) {
    std::ofstream file(filePath);
    if (file.is_open()) {
      file.close();
    } else {
      SPDLOG_ERROR("Failed to create file: {}", filePath);
    }
  }
}
class ReadNextPartition {
 public:
  ReadNextPartition(int fd, std::vector<int64_t> offsets_in_bytes)
      : fd_(fd), offsets_in_bytes_(offsets_in_bytes) {
    thread_ = nullptr;
    lock_ = new std::mutex;
    prepared_ = true;
    done_ = false;
    max_size_ = 0;
    part_id_ = INVALID_PART_ID;
    for (size_t i = 0; i < offsets_in_bytes_.size() - 1; i++) {
      max_size_ =
          std::max(offsets_in_bytes_[i + 1] - offsets_in_bytes_[i], max_size_);
    }
  }

  ~ReadNextPartition() {
    if (mem_ != nullptr) {
      free(mem_);
    }
    stop();
    delete lock_;
  }

  void start() {
    if (thread_ == nullptr) {
      SPDLOG_INFO("start read thread");
      thread_ = new std::thread(&ReadNextPartition::run, this);
    }
  }

  void stop() {
    if (thread_ != nullptr) {
      if (thread_->joinable()) {
        SPDLOG_INFO("stopping read thread");
        done_ = true;
        prepared_ = false;
        cv_.notify_all();
        thread_->join();
        SPDLOG_INFO("read thread finished");
      }
      delete thread_;
    }
  }
  /*将预取的数据拷贝到缓存中*/
  void move_to_cache(void *cache, int64_t size_in_bytes, int64_t part_id) {
    std::unique_lock<std::mutex> lock(*lock_);
    cv_.wait(lock, [this] { return prepared_ == true; });
    memcpy(cache, mem_, size_in_bytes);
    prepared_ = false;
    part_id_ = part_id + 1;
    lock.unlock();
    cv_.notify_all();
    SPDLOG_INFO("move data(part id={}, size={} bytes) to cache done",
                part_id,
                size_in_bytes);
  }

  void start_read() {
    std::unique_lock<std::mutex> lock(*lock_);
    cv_.wait(lock, [this] {
      return prepared_ == true && part_id_ == INVALID_PART_ID;
    });

    prepared_ = false;
    part_id_ = 0;
    // 分配预取空间
    if (posix_memalign(&mem_, 4096, max_size_)) {
      SPDLOG_ERROR(
          "Unable to allocate memory: {} bytes\nError: {}", max_size_, errno);
      throw std::runtime_error("Unable to allocate memory");
    }
    lock.unlock();
    cv_.notify_all();
  }

 private:
  // 从0到n顺序加载分区数据到缓存中
  void run() {
    while (!done_) {
      // 阻塞直到数据被使用
      std::unique_lock<std::mutex> lock(*lock_);
      cv_.wait(lock, [this] {
        return (prepared_ == false && part_id_ < offsets_in_bytes_.size() - 1 &&
                part_id_ >= 0) ||
               done_;
      });
      // 读取数据
      if (!done_ && part_id_ < offsets_in_bytes_.size() - 1 && part_id_ >= 0) {
        int64_t size =
            offsets_in_bytes_[part_id_ + 1] - offsets_in_bytes_[part_id_];

        if (pread(fd_, (char *)mem_, size, offsets_in_bytes_[part_id_]) == -1) {
          SPDLOG_ERROR("pread ERROR: {}, size={}, offset={}",
                       errno,
                       size,
                       offsets_in_bytes_[part_id_]);
          throw std::runtime_error("ReadNextPartition ERROR");
        }
      }
      prepared_ = true;
      lock.unlock();
      cv_.notify_all();
    }
  }
  std::thread *thread_;
  // 存储预取的数据
  void *mem_;
  int fd_;
  int64_t max_size_;
  // 数据是否准备好
  std::atomic<bool> prepared_;
  std::mutex *lock_;
  std::condition_variable cv_;
  // 是否处理完成
  std::atomic<bool> done_;
  std::vector<int64_t> offsets_in_bytes_;
  // 当前读取数据的分区ID
  int64_t part_id_;
};

class WritePartition {
 public:
  WritePartition(int fd,
                 int dim,
                 int64_t dtype_size,
                 int num_workers = 2,
                 bool sequential = true)
      : fd_(fd),
        dim_(dim),
        dtype_size_(dtype_size),
        num_workers_(num_workers),
        sequential_(sequential) {
    queue_lock_ = new std::mutex;
  }

  ~WritePartition() {
    stop();
    delete queue_lock_;
  }

  void start() {
    if (threads_.size() == 0) {
      SPDLOG_INFO("start {} write threads", num_workers_);
      for (int i = 0; i < num_workers_; i++) {
        std::thread *t = new std::thread(&WritePartition::run, this);
        threads_.push_back(t);
      }
    }
  }

  void stop() {
    if (threads_.size() != 0) {
      SPDLOG_INFO("stopping {} write threads", num_workers_);
      done_ = true;
      cv_.notify_all();
      for (int i = 0; i < num_workers_; i++) {
        threads_[i]->join();
        delete threads_[i];
      }
      SPDLOG_INFO("{} write threads finished", num_workers_);
      threads_.clear();
    }
  }

  void async_write(torch::Tensor data, torch::Tensor batch) {
    std::unique_lock<std::mutex> lock(*queue_lock_);
    queue_.push({data, batch});
    lock.unlock();
    cv_.notify_all();
    auto sizes = data.sizes();
    SPDLOG_DEBUG("send {} data (shape=({}, {}))to queue",
                 batch.numel(),
                 sizes[0],
                 sizes[1]);
  }

  /*同步：直到队列中的所有数据写入磁盘*/
  void flush() {
    SPDLOG_INFO("flushing all data to disk");
    // 注意：只是等到队列为空，不能保证所有数据都写入到磁盘中
    std::unique_lock<std::mutex> lock(*queue_lock_);
    cv_.wait(lock, [this] { return queue_.empty() == true; });
    SPDLOG_INFO("flush all data to disk done");
  }

 private:
  void write_data_to_ssd(const torch::Tensor &data,
                         const torch::Tensor &batch) {
    int64_t n = batch.numel();
    int num_threads = atoi(getenv("SINFER_NUM_THREADS"));

    int64_t row_size = dim_ * dtype_size_;

    auto data_ptr = data.data_ptr();
    auto batch_ptr = batch.data_ptr<int64_t>();
    if (!sequential_) {
#pragma omp parallel for num_threads(num_threads)
      for (int64_t i = 0; i < n; i++) {
        int64_t data_offset = i * row_size;
        int64_t offset = batch_ptr[i] * row_size;
        if (pwrite(fd_, (char *)data_ptr + data_offset, row_size, offset) ==
            -1) {
          SPDLOG_ERROR(
              "pwrite ERROR: {}, node id={}, data offset={}, size={}, file "
              "offset={}",
              strerror(errno),
              batch_ptr[i],
              data_offset,
              row_size,
              offset);
          throw std::runtime_error("WritePartition pwrite ERROR");
        }
      }
    } else {
      int64_t start = batch_ptr[0];
      int64_t end = batch_ptr[n - 1];
      int64_t total_size = n * row_size;
      int64_t file_offset = start * row_size;
      if (pwrite(fd_, (char *)data_ptr, total_size, file_offset) == -1) {
        SPDLOG_ERROR(
            "pwrite ERROR: {}, node({}-{}), total_size={}, file offset={}",
            strerror(errno),
            start,
            end,
            total_size,
            file_offset);
        throw std::runtime_error("WritePartition pwrite ERROR");
      }
    }
    SPDLOG_DEBUG("write {} data (shape=({}, {}))to file done",
                 batch.numel(),
                 data.sizes()[0],
                 data.sizes()[1]);
  }
  void run() {
    while (!done_) {
      // 阻塞直到队列中有数据或者线程结束
      std::unique_lock<std::mutex> lock(*queue_lock_);
      cv_.wait(lock, [this] { return queue_.empty() == false || done_; });

      if (done_) {
        break;
      }
      auto data = std::move(queue_.front());
      auto feat = data.first;
      auto batch = data.second;
      queue_.pop();
      lock.unlock();
      cv_.notify_all();

      write_data_to_ssd(feat, batch);
    }
  }
  std::vector<std::thread *> threads_;
  int fd_;
  int64_t dim_;
  int64_t dtype_size_;
  int num_workers_;
  bool sequential_;
  std::mutex *queue_lock_;
  std::condition_variable cv_;
  std::atomic<bool> done_;
  std::queue<std::pair<torch::Tensor, torch::Tensor>> queue_;
};

class FeatureStore {
 public:
  FeatureStore(const std::string &file_path,
               std::vector<int64_t> offsets,
               int64_t num,
               int64_t dim,
               bool prefetch = true,
               torch::Dtype dtype = torch::kFloat32,
               int writer_workers = 2,
               bool writer_seq = true)
      : file_path_(file_path),
        offsets_(offsets),
        num_(num),
        dim_(dim),
        prefetch_(prefetch),
        dtype_(dtype) {
    createFileIfNotExists(file_path_);
    int flags = O_RDWR;
    fd_ = open(file_path_.c_str(), flags);
    if (fd_ == -1) {
      SPDLOG_ERROR("Unable to open {}\nError: {}", file_path_, errno);
      throw std::runtime_error("Unable to open file " + file_path_);
    }
    if (dtype_ == torch::kFloat64) {
      dtype_size_ = 8;
    } else if (dtype_ == torch::kFloat32) {
      dtype_size_ = 4;
    } else if (dtype_ == torch::kFloat16) {
      dtype_size_ = 2;
    } else if (dtype_ == torch::kInt64) {
      dtype_size_ = 8;
    } else if (dtype_ == torch::kInt32) {
      dtype_size_ = 4;
    }
    offsets_in_bytes_.reserve(offsets_.size());
    for (size_t i = 0; i < offsets_.size(); i++) {
      int64_t n = offsets_[i];
      offsets_in_bytes_.push_back(n * dim_ * dtype_size_);
    }
    max_size_in_bytes_ = 0;
    for (size_t i = 0; i < offsets_in_bytes_.size() - 1; i++) {
      max_size_in_bytes_ = std::max(
          offsets_in_bytes_[i + 1] - offsets_in_bytes_[i], max_size_in_bytes_);
    }

    cache_part_id_ = INVALID_PART_ID;
    reader_ = std::make_unique<ReadNextPartition>(fd_, offsets_in_bytes_);
    reader_->start();

    writer_ = std::make_unique<WritePartition>(
        fd_, dim_, dtype_size_, writer_workers, writer_seq);
    writer_->start();
  }
  ~FeatureStore() {
    if (cache_ptr_ != nullptr) {
      free(cache_ptr_);
    }
    close(fd_);
  }
  /*将part_id对应的分区特征加载到内存中*/
  void update_cache(int64_t part_id) {
    if (part_id != cache_part_id_) {
      int64_t num = offsets_[part_id + 1] - offsets_[part_id];
      int64_t size_in_bytes =
          offsets_in_bytes_[part_id + 1] - offsets_in_bytes_[part_id];
      if (cache_part_id_ == INVALID_PART_ID) {
        // 分配缓存空间
        if (posix_memalign(&cache_ptr_, 4096, max_size_in_bytes_)) {
          SPDLOG_ERROR("Unable to allocate memory: {} bytes\nError: {}",
                       max_size_in_bytes_,
                       errno);
          throw std::runtime_error("Unable to allocate memory");
        }
      }
      if (prefetch_) {
        if (cache_part_id_ == INVALID_PART_ID) {
          // 唤醒预取线程
          reader_->start_read();
        }
        reader_->move_to_cache(cache_ptr_, size_in_bytes, part_id);
      } else {
        if (pread(fd_,
                  (void *)cache_ptr_,
                  size_in_bytes,
                  offsets_in_bytes_[part_id]) == -1) {
          SPDLOG_ERROR("Unable to read part_id={}\nError: {}", part_id, errno);
          throw std::runtime_error("Unable to read part_id=" + part_id);
        }
      }
      // 构造cache tensor
      cache_ = torch::from_blob(cache_ptr_, {num, dim_}, dtype_);
      cache_part_id_ = part_id;
    }
  }

  /*从磁盘和缓存中读取对应id的特征数据*/
  torch::Tensor gather(const torch::Tensor &ids) {
    if (cache_part_id_ != INVALID_PART_ID) {
      return gather_cache_ssd_with_fd(fd_,
                                      ids,
                                      dim_,
                                      cache_,
                                      offsets_[cache_part_id_],
                                      offsets_[cache_part_id_ + 1]);
    } else {
      return gather_ssd_with_fd(fd_, ids, dim_);
    }
  }

  /**
   * 获取磁盘中的所有数据
   */
  torch::Tensor gather_all() {
    return gather_range_with_fd(fd_, 0, num_, dim_);
  }
  /*将数据写入到磁盘中*/
  void write_data(const torch::Tensor &nodes, const torch::Tensor &data) {
    writer_->async_write(data, nodes);
  }
  /*同步：等待所有写入操作完成*/
  void flush() { writer_->flush(); }

 private:
  std::string file_path_;
  int fd_;
  std::vector<int64_t> offsets_;
  std::vector<int64_t> offsets_in_bytes_;
  int64_t max_size_in_bytes_;
  // `(num_, dim_)` tensor
  int64_t num_;
  int64_t dim_;
  // 是否预取特征到缓存中
  bool prefetch_;
  torch::Dtype dtype_;
  int64_t dtype_size_;
  torch::Tensor cache_;
  void *cache_ptr_;
  // 当前缓存的分区ID
  int64_t cache_part_id_;
  std::unique_ptr<ReadNextPartition> reader_;
  std::unique_ptr<WritePartition> writer_;
};
