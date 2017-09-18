/*
 * Config.h
 *
 *  Created on: 21/ago/2017
 *      Author: macbook
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <glm/glm.hpp>

#include <string>
using namespace std;

#include "Utilities.h"
#include "RenderCamera.h"
#include "Scene.h"
/*
if (vm.count("help")) {
	cout << desc << "\n";
	return 0;
}
	cout << "Compression level was set to "
		 << vm["compression"].as<double>() << ".\n";
} else {
	cout << "Compression level was not set.\n";
}

*/

struct Config{

	//CUDA
	int devId;
	dim3 block_size_default;
	Scene scene;

	/**INTERACTIVE**/
	bool interactive;
	glm::ivec2 resolution;

	//*RENDER VARIABLES*/
	uint samples;
	float sample_per_pass;
	uint passes;
	uint max_ray_bounces;

	Config(const Scene& _scene):
			scene(_scene),
			resolution(400,400),
			samples(50),
			sample_per_pass(1),
			max_ray_bounces(80),
			devId(0),
			interactive(false)
			{
				passes = int(ceil(samples/sample_per_pass));
				block_size_default = dim3(32,32);
			}

	glm::ivec2 getResolution() const{
		return resolution;
	}

	/*setup and return the camera*/
	RenderCamera getCamera() const{
		RenderCamera camera;
		camera.setFovFromY(0.5);
		camera.calculateInvViewMatrixFromBasis();
		return camera;
	}

};

class ConfigParser{

	po::options_description option_desc;
	po::variables_map variables_map;

public:
	ConfigParser(){
		option_desc.add_options()
			("help", "produce help message")
			("compression", po::value<double>(), "set compression level")
		;
	}

    void parseCommandline(int ac, char **av){
    	printf("command line parsing \n");
    	po::variables_map vm;
    	po::store(po::parse_command_line(ac, av, option_desc), vm);
    	po::notify(vm);
	}

    Config createConfig(){

    	printf("XML scene builder building ");
    	XMLSceneBuilder sc_builder("whatever");

    	printf("XML scene builder built ");
    	SceneAssembler assembler;
    	assembler.setBuilder(&sc_builder);
    	printf("config creating");
    	Config config(assembler.getScene());
    	printf("config created");
    	return config;
    }

    bool contain(string str){
    	return (bool) variables_map.count(str.c_str());
    }

};


#endif /* CONFIG_H_ */
