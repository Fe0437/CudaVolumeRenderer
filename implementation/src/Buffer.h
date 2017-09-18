#pragma once
#include <cassert>
#include <cstddef>
#include <string_view>
#include <typeinfo>

struct Buffer {
  void* data = nullptr;
  size_t size_bytes = 0;

#ifdef _DEBUG
  std::string_view type_name;

  Buffer() {}
  Buffer(void* ptr, size_t bytes, std::string_view debug_type_name)
      : data(ptr), size_bytes(bytes), type_name(debug_type_name) {}

  template <typename T>
  T* as() const {
    assert(data && "Null buffer access!");
    assert(type_name == typeid(T).name() && "Buffer type mismatch!");
    return static_cast<T*>(data);
  }

  Buffer operator+(size_t offset_bytes) const {
    assert(offset_bytes <= size_bytes && "Buffer offset out of bounds!");
    return Buffer(static_cast<char*>(data) + offset_bytes,
                  size_bytes - offset_bytes, type_name);
  }
#else
  Buffer(void* ptr, size_t bytes) : data(ptr), size_bytes(bytes) {}

  template <typename T>
  auto as() const -> T* {
    return static_cast<T*>(data);
  }

  auto operator+(size_t offset_bytes) const -> Buffer {
    return {static_cast<char*>(data) + offset_bytes, size_bytes - offset_bytes};
  }
#endif
};

// Helper function to create buffers
template <typename T>
auto make_buffer(T* data, size_t count) -> Buffer {
#ifdef _DEBUG
  return Buffer(data, count * sizeof(T), typeid(T).name());
#else
  return Buffer(data, count * sizeof(T));
#endif
}

struct Buffer2D {
  void* data = nullptr;
  size_t width = 0;        // Width in pixels
  size_t width_bytes = 0;  // Width in bytes
  size_t height = 0;
  size_t pitch_bytes = 0;  // Row stride in bytes

#ifdef _DEBUG
  std::string_view type_name;

  Buffer2D() {}
  Buffer2D(void* ptr, size_t w, size_t width_b, size_t h, size_t pitch_b,
           std::string_view debug_type_name)
      : data(ptr),
        width(w),
        width_bytes(width_b),
        height(h),
        pitch_bytes(pitch_b),
        type_name(debug_type_name) {}

  template <typename T>
  T* as() const {
    assert(data && "Null buffer access!");
    assert(type_name == typeid(T).name() && "Buffer type mismatch!");
    return static_cast<T*>(data);
  }

  Buffer2D operator+(size_t offset_bytes) const {
    return Buffer2D(static_cast<unsigned char*>(data) + offset_bytes, width,
                    width_bytes, height, pitch_bytes, type_name);
  }

  // Get a 1D buffer for a specific row
  Buffer row(size_t row_index) const {
    assert(row_index < height && "Row index out of bounds!");
    return Buffer(static_cast<unsigned char*>(data) + (row_index * pitch_bytes),
                  width_bytes, type_name);
  }
#else
  Buffer2D(void* ptr, size_t w, size_t width_b, size_t h, size_t pitch_b)
      : data(ptr),
        width(w),
        width_bytes(width_b),
        height(h),
        pitch_bytes(pitch_b) {}

  template <typename T>
  auto as() const -> T* {
    return static_cast<T*>(data);
  }

  auto operator+(size_t offset_bytes) const -> Buffer2D {
    return {static_cast<unsigned char*>(data) + offset_bytes, width,
            width_bytes, height, pitch_bytes};
  }

  [[nodiscard]] auto row(size_t row_index) const -> Buffer {
    return {static_cast<unsigned char*>(data) + (row_index * pitch_bytes),
            width_bytes};
  }
#endif
};

// Helper function to create 2D buffers
template <typename T>
auto make_buffer2D(T* data, size_t width, size_t height, size_t pitch = 0)
    -> Buffer2D {
  size_t width_bytes = width * sizeof(T);
  size_t pitch_bytes = pitch ? pitch : width_bytes;

#ifdef _DEBUG
  return Buffer2D(data, width, width_bytes, height, pitch_bytes,
                  typeid(T).name());
#else
  return Buffer2D(data, width, width_bytes, height, pitch_bytes);
#endif
}