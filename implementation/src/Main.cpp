/*
 ============================================================================
 Name        : CudaVolumeRenderer.cpp
 Author      : Federico Forti
 Version     :
 Copyright   :
 Description :
 ============================================================================
 */

// 1. CUDA headers first
#include <cuda_runtime.h>

// 2. Third-party libraries
// GLEW must be included before any OpenGL headers
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// 3. System/Standard headers
#include <stdio.h>
#include <time.h>

#include <fstream>
#include <iostream>

// 4. Project headers
#include "Config.h"
#include "ConfigParser.h"
#include "Defines.h"
#include "ImageBufferTransfer.h"
#include "RendererFactory.h"

auto initConfig(Config &config, int argc, char **argv) -> bool {
  ConfigParser parser;
  if (!parser.parseCommandline(argc, argv)) {
    return false;
  }
  config = parser.createConfig();
  return true;
}

#ifdef RAYS_STATISTICS
int n_rays_traced_statistic = 0;
#endif

void runTest(Config config) {
  float mean_time = 0;
  int mean_rays_traced_statistic = 0;
  std::vector<float> times;
  int trials = config.test_trials;
  clock_t startT;
  clock_t endT;

  for (int i = 0; i < trials; i++) {
    printf(
        "---------------------------------------------------------------trial "
        ": %d \n",
        i);
    // necessary for memory leak checking
    cudaDeviceReset();
#ifdef RAYS_STATISTICS
    n_rays_traced_statistic = 0;
#endif
    startT = clock();
    Image image(config.resolution.x, config.resolution.y);
    auto renderer = RendererFactory::createRenderer(config);
    endT = clock();
    LOG_DEBUG("initialization time : %.2f sec \n",
              float(endT - startT) / CLOCKS_PER_SEC)

    Buffer2D outBuffer = make_buffer2D<float4>(
        image.pixels, image.getResolution().x, image.getResolution().y);
    startT = clock();
    renderer->render(outBuffer);
    endT = clock();
    printf("rendering time      : %.2f sec \n",
           float(endT - startT) / CLOCKS_PER_SEC);

    if (i > 0) {
      // discard the first iteration
      times.push_back(float(endT - startT) / CLOCKS_PER_SEC);
      mean_time += times.back();

#ifdef RAYS_STATISTICS
      mean_rays_traced_statistic += n_rays_traced_statistic;
#endif
    }

#ifdef RAYS_STATISTICS
    printf("total traced rays %d \n", n_rays_traced_statistic);
#endif

    startT = clock();
    image.saveHDR(config.output_name);
    endT = clock();
    LOG_DEBUG("saving time         : %.2f sec \n",
              float(endT - startT) / CLOCKS_PER_SEC)
  }

  if (trials > 1) {
    mean_time /= times.size();
    mean_rays_traced_statistic =
        (float)mean_rays_traced_statistic / times.size();

    float time_variance = 0;
    for (auto t : times) {
      time_variance += (t - mean_time) * (t - mean_time);
    }
    time_variance /= times.size();

    printf("execution mean time of %.2f sec on %zu iterations and std %.5f \n",
           mean_time, times.size(), sqrtf(time_variance));
    printf("paths per sec %lf \n",
           (double)config.resolution.x * config.resolution.y *
               config.path_tracing_config.iterations / (double)mean_time);
#ifdef RAYS_STATISTICS
    printf("milions of rays per sec %lf \n",
           (double)mean_rays_traced_statistic / ((double)mean_time * 1e6));
#endif
  }
}

void runInteractive(Config config) {
  auto view_controller = std::unique_ptr<GLViewController>(
      RendererFactory::createInteractiveRenderer(config));
  view_controller->init();
  view_controller->mainLoop();
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
auto main(int argc, char **argv) -> int {
  Config config;
  if (initConfig(config, argc, argv)) {
    if (config.interactive) {
      runInteractive(config);
    } else {
      runTest(config);
    }
  }

  std::cout << "Press Enter to exit..." << std::endl;
  std::cin.get();
  return 0;
}
