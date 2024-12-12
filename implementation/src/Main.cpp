/*
 ============================================================================
 Name        : CudaVolumeRenderer.cpp
 Author      : Federico Forti
 Version     :
 Copyright   : 
 Description :
 ============================================================================
 */

#include <GL/glew.h>
#include "Defines.h"

#include <fstream>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include "Config.h"
#include "RendererFactory.h"

bool initConfig(Config& config, int argc, char **argv) {

	ConfigParser parser;
	if(!parser.parseCommandline(argc, argv)) return false;
	config = parser.createConfig();
}


#ifdef RAYS_STATISTICS
int n_rays_traced_statistic = 0;
#endif

void runTest(Config config) {

	float mean_time = 0;
	int mean_rays_traced_statistic = 0;
	std::vector<float> times;
	int trials = config.test_trials;
	clock_t startT, endT;

	for (int i = 0; i < trials; i++) {
		printf("---------------------------------------------------------------trial : %d \n", i);
		//necessary for memory leak checking
		cudaDeviceReset();
#ifdef RAYS_STATISTICS
		n_rays_traced_statistic = 0;
#endif
		startT = clock();
		Image image(config.resolution.x, config.resolution.y);
		auto renderer = RendererFactory::createRenderer(config);
		endT = clock();
		LOG_DEBUG("initialization time : %.2f sec \n", float(endT - startT) / CLOCKS_PER_SEC)

	    startT = clock();
		renderer->render(image.pixels);
		endT = clock();
		printf("rendering time      : %.2f sec \n", float(endT - startT) / CLOCKS_PER_SEC);

		if (i>0){
			//discard the first iteration
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
		LOG_DEBUG("saving time         : %.2f sec \n", float(endT - startT) / CLOCKS_PER_SEC)

		delete renderer;
	}

	if (trials > 1){

		mean_time /= times.size();
		mean_rays_traced_statistic = (float)mean_rays_traced_statistic / times.size();

		float time_variance = 0;
		for (auto t : times) {
			time_variance += (t - mean_time)*(t - mean_time);
		}
		time_variance /= times.size();

		printf("execution mean time of %.2f sec on %d iterations and std %.5f \n", mean_time, times.size(), sqrtf(time_variance));
		printf("paths per sec %lf \n", (double)config.resolution.x * config.resolution.y * config.path_tracing_config.iterations / (double)mean_time);
#ifdef RAYS_STATISTICS
		printf("milions of rays per sec %lf \n", (double)mean_rays_traced_statistic / ((double)mean_time*1e6));
#endif
	}
}

void runInteractive(Config config) {
	auto view_controller = std::unique_ptr<GLViewController>(RendererFactory::createInteractiveRenderer(config));
	view_controller->init();
	view_controller->mainLoop();
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
	Config config;
	if (initConfig(config, argc, argv)){

		if (config.interactive) {
			runInteractive(config);
		}else{
			runTest(config);
		}
	}
	
	//getchar();
	return 0;
}

