#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "helper_cuda.h"

template <typename T>
struct CudaDeleter {
  void operator()(T* ptr) const {
    if (ptr) {
      cudaFree(ptr);
    }
  }
};

template <typename T>
class CudaView {
 private:
  T* ptr_;

 public:
  CudaView() : ptr_(nullptr) {}
  explicit CudaView(T* ptr) : ptr_(ptr) {}

  auto get() const -> T* { return ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }

  void memset(int value = 0, size_t count = 1) {
    if (ptr_) {
      checkCudaErrors(cudaMemset(ptr_, value, count * sizeof(T)));
    }
  }
};

template <typename T>
class CudaUniquePtr {
 private:
  std::unique_ptr<T, CudaDeleter<T>> ptr_;

 public:
  CudaUniquePtr() : ptr_(nullptr) {}

  static auto make(size_t count) -> CudaUniquePtr<T> {
    T* raw_ptr = nullptr;
    checkCudaErrors(cudaMalloc(&raw_ptr, count * sizeof(T)));
    return CudaUniquePtr<T>(raw_ptr);
  }

  // Release ownership and return the pointer
  auto release() -> T* { return ptr_.release(); }

  void reset(T* ptr = nullptr) { ptr_.reset(ptr); }

  explicit operator bool() const { return ptr_ != nullptr; }

  void memset(int value = 0, size_t count = 1) {
    if (ptr_) {
      checkCudaErrors(cudaMemset(ptr_.get(), value, count * sizeof(T)));
    }
  }

  // Remove get() to prevent accidental pointer sharing
  CudaUniquePtr(const CudaUniquePtr&) = delete;
  auto operator=(const CudaUniquePtr&) -> CudaUniquePtr& = delete;
  CudaUniquePtr(CudaUniquePtr&&) = default;
  auto operator=(CudaUniquePtr&&) -> CudaUniquePtr& = default;

  // Add method to create a view
  auto view() const -> CudaView<T> { return CudaView<T>(ptr_.get()); }

 private:
  explicit CudaUniquePtr(T* ptr) : ptr_(ptr) {}
};