#ifndef RENDERERFACTORY_H_
#define RENDERERFACTORY_H_

#include "AbstractRenderer.h"
#include "Config.h"
#include "CudaVolPath.h"
#include "ImageBufferTransfer.h"
#include "InteractiveRenderer.h"
#include "RenderKernelLauncher.h"

class RendererFactory {
 public:
  static std::unique_ptr<AbstractRenderer> createRenderer(
      const Config& config) {
    switch (config.algorithm) {
      case Config::Algorithm::kCudaVolPath:
        return getCudaVolPath(
            config, std::make_unique<HostImageBufferTansferDelegate>());
      default:
        throw std::runtime_error("Unknown algorithm type in createRenderer");
    }
  }

  static std::unique_ptr<GLViewController> createInteractiveRenderer(
      const Config& config) {
    auto buffer_processor = createBufferProcessorDelegate(config);
    auto controller = createInputController(config);

    auto view_controller = std::make_unique<GLViewController>(
        config.resolution.x, config.resolution.y);
    view_controller->setBufferProcessorDelegate(std::move(buffer_processor));
    view_controller->setInputController(std::move(controller));
    return view_controller;
  }

 private:
  static std::unique_ptr<AbstractProgressiveRenderer> getCudaVolPath(
      const Config& config,
      std::unique_ptr<Buffer2DTransferDelegate<UtilityFunctors::Scale>>&&
          output_delegate) {
    std::unique_ptr<AbstractProgressiveRenderer> renderer{};

    if (config.cuda_config.unified_memory) {
      switch (config.kernel) {
        case Config::Kernel::kNaivesk:
          renderer = std::make_unique<CudaVolPath<
              NaiveVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        case Config::Kernel::kNaivemk:
          renderer = std::make_unique<CudaVolPath<
              NaiveVolPTmk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        case Config::Kernel::kRegenerationsk:
          renderer = std::make_unique<CudaVolPath<RegenerationVolPTsk<
              SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        case Config::Kernel::kStreamingmk:
          renderer = std::make_unique<CudaVolPath<StreamingVolPTmk<
              SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        case Config::Kernel::kStreamingsk:
          renderer = std::make_unique<CudaVolPath<StreamingVolPTsk<
              SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        case Config::Kernel::kSortingsk:
          renderer = std::make_unique<CudaVolPath<
              SortingVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        default:
          throw std::runtime_error("Unknown kernel type for unified memory");
      }
    } else {
      switch (config.kernel) {
        case Config::Kernel::kNaivesk:
          renderer = std::make_unique<CudaVolPath<
              NaiveVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        case Config::Kernel::kNaivemk:
          renderer = std::make_unique<CudaVolPath<
              NaiveVolPTmk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        case Config::Kernel::kRegenerationsk:
          renderer = std::make_unique<CudaVolPath<
              RegenerationVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        case Config::Kernel::kStreamingmk:
          renderer = std::make_unique<CudaVolPath<
              StreamingVolPTmk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        case Config::Kernel::kStreamingsk:
          renderer = std::make_unique<CudaVolPath<
              StreamingVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        case Config::Kernel::kSortingsk:
          renderer = std::make_unique<CudaVolPath<
              SortingVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>>(
              config, std::move(output_delegate));
          break;
        default:
          throw std::runtime_error("Unknown kernel type for device memory");
      }
    }
    return renderer;
  }

  static std::unique_ptr<BufferProcessorDelegate> createBufferProcessorDelegate(
      const Config& config) {
    switch (config.algorithm) {
      case Config::Algorithm::kCudaVolPath: {
        std::unique_ptr<Buffer2DTransferDelegate<UtilityFunctors::Scale>>
            transfer_delegate;

        if (config.tiling_config.n_tiles.x == 1 &&
            config.tiling_config.n_tiles.y == 1) {
          transfer_delegate =
              std::make_unique<DeviceImageBufferTansferDelegate>();
        } else {
          transfer_delegate =
              std::make_unique<DeviceTiledImageBufferTansferDelegate>(
                  config.resolution.x * config.resolution.y * sizeof(float4));
        }

        auto renderer = getCudaVolPath(config, std::move(transfer_delegate));
        return std::make_unique<CudaInteractiveRenderer>(std::move(renderer));
      }
      default:
        throw std::runtime_error(
            "Unknown algorithm type in createBufferProcessorDelegate");
    }
  }

  static std::unique_ptr<InputController> createInputController(
      const Config& config) {
    switch (config.algorithm) {
      case Config::Algorithm::kCudaVolPath:
        return std::make_unique<CameraController>(config.scene.getCamera());
      default:
        throw std::runtime_error(
            "Unknown algorithm type in createInputController");
    }
  }
};

#endif
