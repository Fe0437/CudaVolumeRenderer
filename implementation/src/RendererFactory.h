#ifndef RENDERERFACTORY_H_
#define RENDERERFACTORY_H_

#include "InteractiveRenderer.h"
#include "Config.h"
#include "AbstractRenderer.h"
#include "CudaVolPath.h"
#include "ImageBufferTransfer.h"
#include "RenderKernelLauncher.h"

class RendererFactory
{

public:
	
	static AbstractRenderer* createRenderer(const Config& config) {
		switch (config.algorithm) {
			case Config::Algorithm::kCudaVolPath:
					return getCudaVolPath(config, new HostImageBufferTansferDelegate());
					break;
		}
	}

	static GLViewController* createInteractiveRenderer(const Config& config) {

		auto buffer_processor = createBufferProcessorDelegate(config);
		auto controller = createInputController(config);

		GLViewController* view_controller = new GLViewController(config.resolution.x, config.resolution.y);
		view_controller->setBufferProcessorDelegate(buffer_processor);
		view_controller->setInputController(controller);
		return view_controller;
	}

private:

	static AbstractProgressiveRenderer* getCudaVolPath(const Config& config, Buffer2DTransferDelegate<UtilityFunctors::Scale>* output_delegate) {
		AbstractProgressiveRenderer* renderer;

		if (config.cuda_config.unified_memory) {
			switch (config.kernel) {
			case Config::Kernel::kNaivesk:
				renderer = new CudaVolPath<NaiveVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>(config, output_delegate);
				break;
			case Config::Kernel::kNaivemk:
				renderer = new CudaVolPath<NaiveVolPTmk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>(config, output_delegate);
				break;
			case Config::Kernel::kRegenerationsk:
				renderer = new CudaVolPath<RegenerationVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>(config, output_delegate);
				break;
			case Config::Kernel::kStreamingmk:
				renderer = new CudaVolPath<StreamingVolPTmk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>(config, output_delegate);
				break;
			case Config::Kernel::kStreamingsk:
				renderer = new CudaVolPath<StreamingVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>(config, output_delegate);
				break;
			case Config::Kernel::kSortingsk:
				renderer = new CudaVolPath<SortingVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>(config, output_delegate);
				break;
			}
		}else {
			switch (config.kernel) {
			case Config::Kernel::kNaivesk:
				renderer = new CudaVolPath<NaiveVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>(config, output_delegate);
				break;
			case Config::Kernel::kNaivemk:
				renderer = new CudaVolPath<NaiveVolPTmk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>(config, output_delegate);
				break;
			case Config::Kernel::kRegenerationsk:
				renderer = new CudaVolPath<RegenerationVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>(config, output_delegate);
				break;
			case Config::Kernel::kStreamingmk:
				renderer = new CudaVolPath<StreamingVolPTmk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>(config, output_delegate);
				break;
			case Config::Kernel::kStreamingsk:
				renderer = new CudaVolPath<StreamingVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>(config, output_delegate);
				break;
			case Config::Kernel::kSortingsk:
				renderer = new CudaVolPath<SortingVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>(config, output_delegate);
				break;
			}
		}
		return renderer;
	}

	static BufferProcessorDelegate* createBufferProcessorDelegate(const Config& config) {
		switch (config.algorithm) {
			case Config::Algorithm::kCudaVolPath:
				Buffer2DTransferDelegate<UtilityFunctors::Scale>* transfer_delegate;

				if(config.tiling_config.n_tiles.x==1 && config.tiling_config.n_tiles.y == 1){
					transfer_delegate = new DeviceImageBufferTansferDelegate();
				}
				else {
					transfer_delegate = new DeviceTiledImageBufferTansferDelegate(config.resolution.x* config.resolution.y * sizeof(float4));
				}

				auto renderer = getCudaVolPath(config, transfer_delegate);

				return new CudaInteractiveRenderer(renderer);
				break;
		}
	}

	static InputController* createInputController(const Config& config) {
		switch (config.algorithm) {
		case Config::Algorithm::kCudaVolPath:
				return new CameraController(config.scene.getCamera());
			break;
		}
	}
};

#endif

