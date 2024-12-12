/*
 * Config.h
 *
 *  Created on: 21/ago/2017
 *      Author: macbook
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include <argparse/argparse.hpp>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include "Scene.h"
#include "RawSceneBuilder.h"
#include "MhaSceneBuilder.h"
#include "XmlSceneBuilder.h"
#include "Image.h"

/*
how to use

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

struct CudaConfig {
	//CUDA
	int devId = -1;
	cudaDeviceProp device_properties;
	bool unified_memory = false;
	dim3 block_size = dim3(32);
	dim3 grid_size = dim3(1);
	size_t dynamic_shared_memory = 0;
};

struct PathTracingConfig {
	
	// PATH TRACING
	float       max_time;
	unsigned int iterations;
	unsigned int max_path_length;

	PathTracingConfig(
		unsigned int _iterations = 10
		) :
		iterations(_iterations),
		max_path_length(1000), 
		max_time(-1)
	{
	}
};


struct TilingConfig {

	// PATH TRACING
	uint2 n_tiles;
	uint2 tile_dim;
	uint2  resolution;

	TilingConfig(
		uint2  _resolution = make_uint2(400, 400),
		uint2 n_tiles = make_uint2(1,1)
	) :
		resolution(_resolution),
		n_tiles(n_tiles)
	{
		tile_dim.x = (int)ceil(resolution.x / n_tiles.x);
		tile_dim.y = (int)ceil(resolution.y / n_tiles.y);
	}
};

class Config {

public:

	enum Algorithm
	{
		kCudaVolPath,
		kAlgorithmUnknown,
	};

	enum Kernel
	{
		kNaivesk,
		kNaivemk,
		kRegenerationsk,
		kStreamingmk,
		kStreamingsk,
		kSortingsk,
		kKernelUnknown
	};

	Kernel kernel = kRegenerationsk;
	Algorithm algorithm = kCudaVolPath;

	int test_trials = 1;
	uint2  resolution{ 400,400 };
	std::string output_name = "test";
	bool interactive = false;
	Scene scene;

	PathTracingConfig path_tracing_config;
	TilingConfig tiling_config;
	CudaConfig cuda_config;
	
	Config() {}

	Config(
		const Scene& _scene,
		Algorithm _algorithm = Algorithm::kCudaVolPath
	) :
		scene(_scene),
		algorithm(_algorithm)
	{
		auto res = scene.getCamera()->getResolution();
		resolution = { (uint)res.x, (uint)res.y };
	}

	Config& cudaConfig(CudaConfig _cuda_config) {
		cuda_config = _cuda_config;

		auto albedo = scene.getMedium().albedo_volume;
		size_t albedo_texture_bytes = albedo.getBytes();
		float max_memory_threshold = 0.8;

		if (cuda_config.device_properties.totalGlobalMem * max_memory_threshold < albedo_texture_bytes) { //not enough global memory
			cuda_config.unified_memory = true;
			COUT_DEBUG("not enough device global memory -> using unified memory")
		}

		return *this;
	}

	Config& kernelConfig(Kernel _kernel) {
		kernel = _kernel;
		return *this;
	}

	Config& algorithmConfig(Algorithm _alg) {
		algorithm = _alg;
		return *this;
	}

	Config& pathTracingConfig(PathTracingConfig _path_tracing_config) {
		path_tracing_config = _path_tracing_config;
		return *this;
	}

	Config& tilingConfig(TilingConfig _tiling_config) {
		tiling_config = _tiling_config;
		assert(tiling_config.resolution.x == resolution.x);
		assert(tiling_config.resolution.y == resolution.y);
		return *this;
	}

	static const std::vector<std::string> getAlgorithmNamesOrderedVector() {
		return
		{
			"cudaVolPath",
		};
	}

	static const std::string getAlgorithmAcronym(Algorithm alg) {
		auto algs = getAlgorithmNamesOrderedVector();
		BOOST_ASSERT_MSG(algs.size() == Algorithm::kAlgorithmUnknown, "check the ConfigParser class algorithms, kUnknown must be at the end");
		if (alg > algs.size())
			LOG_CONFIG(" algorithm not present in the list of algorithms");
		if (alg < 0) {
			LOG_CONFIG(" error value < 0");
		}
		return algs[alg];
	}

	static Algorithm getAlgorithm(std::string alg)
	{
		for (int i = 0; i<Algorithm::kAlgorithmUnknown; i++)
			if (alg == getAlgorithmAcronym(Algorithm(i)))
				return Algorithm(i);

		return kAlgorithmUnknown;
	}


	static const std::vector<std::string> getKernelNamesOrderedVector() {
		return
		{
			"naiveSK",
			"naiveMK",
			"regenerationSK",
			"streamingMK",
			"streamingSK",
			"sortingSK"
		};
	}

	static const std::string getKernelAcronym(Kernel ker) {
		auto kernels = getKernelNamesOrderedVector();
		BOOST_ASSERT_MSG(kernels.size() == Kernel::kKernelUnknown, "check the ConfigParser class Kernel, kUnknown must be at the end");
		if (ker > kernels.size())
			LOG_CONFIG(" kernel not present in the list of kernels");
		if (ker < 0) {
			LOG_CONFIG(" error value < 0");
		}
		return kernels[ker];
	}

	static Kernel getKernel(std::string ker)
	{
		for (int i = 0; i<Kernel::kKernelUnknown; i++)
			if (ker == getKernelAcronym(Kernel(i)))
				return Kernel(i);

		return kKernelUnknown;
	}

	friend std::ostream & operator<<(std::ostream & str, Config const & c) {
		str <<
			"algorithm_" << c.getAlgorithmAcronym(c.algorithm) <<
			"_kernel_" << c.getKernelAcronym(c.kernel) <<
			"_iter_" << c.path_tracing_config.iterations;
		return str;
	}

	std::string toString() const {
		std::ostringstream sstr;
		sstr << *this;
		return sstr.str();
	}
};


class ConfigParser{

	const std::string print_prefix = "CONFIG: ";
	argparse::ArgumentParser program("CudaVolumeRenderer");

public:

	ConfigParser(){

		positional_options_.add("scene-file", -1);

		generic_options_.add_options()
			("help,h", "produce help message")
			("scene-file,s", po::value<std::string>(), "scene file to parse")
			("scene-type", po::value<std::string>()->default_value("Raw"), "scene type to parse, possible values: \n\t MitsubaXml \n\t VtkMha \n\t Raw")
			("interactive", po::value<bool>()->default_value(true), "run the algorithm with the interactive view")
			("trials", po::value<unsigned int>()->default_value(1), "number of times to run the algorithm to get a more accurate running time estimation")
			("algorithm,a", po::value<std::string>()->default_value("cudaVolPath"), "algorithm to use, possible values \n\t cudaVolPath")
		 	("kernel,k", po::value<std::string>()->default_value("regenerationSK"), "kernel to use, possible values: \n\t sortingSK \n\t streamingMK \n\t streamingSK \n\t regenerationSK \n\t naiveMK \n\t naiveSK")
			("number-of-tiles", po::value<std::vector<unsigned int>>()->multitoken()->default_value({ 1,1 },"1,1"), " number of tiles for rendering, the first value specify the number of tiles along the width of the image and the second along the height")
			("use-unified-memory", po::value<bool>()->default_value(false), "if it is true the unified memory is used for global storage instead of the device memory")
		;

		scene_options_override_.add_options()
			("iterations,i", po::value<unsigned int>()->default_value(20), "number of iterations")
			("output,o", po::value<std::string>(), "output name of the rendered image")
			("resolution,r", po::value<std::vector<unsigned int>>()->multitoken()->default_value({ 400,400 }, "400,400"), "resolution of the rendered image")
			;
	}

    bool parseCommandline(int argc, char **argv) {
        program.add_argument("scene-file")
            .help("scene file to parse");

        program.add_argument("--scene-type")
            .default_value(std::string("Raw"))
            .help("scene type to parse, possible values: MitsubaXml, VtkMha, Raw");

        program.add_argument("--interactive")
            .default_value(true)
            .implicit_value(true)
            .help("run the algorithm with the interactive view");

        program.add_argument("--trials")
            .default_value(1)
            .help("number of times to run the algorithm to get a more accurate running time estimation");

        program.add_argument("--algorithm")
            .default_value(std::string("cudaVolPath"))
            .help("algorithm to use, possible values: cudaVolPath");

        program.add_argument("--kernel")
            .default_value(std::string("regenerationSK"))
            .help("kernel to use, possible values: sortingSK, streamingMK, streamingSK, regenerationSK, naiveMK, naiveSK");

        program.add_argument("--number-of-tiles")
            .default_value(std::vector<unsigned int>{1, 1})
            .nargs(2)
            .help("number of tiles for rendering");

        program.add_argument("--use-unified-memory")
            .default_value(false)
            .implicit_value(true)
            .help("use unified memory for global storage instead of device memory");

        program.add_argument("--iterations")
            .default_value(20)
            .help("number of iterations");

        program.add_argument("--output")
            .default_value(std::string("test"))
            .help("output name of the rendered image");

        program.add_argument("--resolution")
            .default_value(std::vector<unsigned int>{400, 400})
            .nargs(2)
            .help("resolution of the rendered image");

        try {
            program.parse_args(argc, argv);
        } catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::cerr << program;
            return false;
        }

        // Store parsed arguments in member variables
        scene_file_ = program.get<std::string>("scene-file");
        scene_type_ = program.get<std::string>("--scene-type");
        interactive_ = program.get<bool>("--interactive");
        trials_ = program.get<int>("--trials");
        algorithm_ = program.get<std::string>("--algorithm");
        kernel_ = program.get<std::string>("--kernel");
        number_of_tiles_ = program.get<std::vector<unsigned int>>("--number-of-tiles");
        use_unified_memory_ = program.get<bool>("--use-unified-memory");
        iterations_ = program.get<int>("--iterations");
        output_ = program.get<std::string>("--output");
        resolution_ = program.get<std::vector<unsigned int>>("--resolution");

        return true;
    }

	Config createConfig(){

		std::string scene_path = "";
		if (variables_map_.count("scene-file")) {
			scene_path = variables_map_["scene-file"].as<std::string>();
		} else {
			throw(" Error : no scene file provided");
		}

		SceneBuilder* sc_builder;
		std::string scene_type = variables_map_["scene-type"].as<std::string>();

		if (scene_type == "MitsubaXml") {
			sc_builder = new XmlSceneBuilder(scene_path);
		}
#ifdef MHA_SUPPORT
		else if(scene_type == "VtkMha"){
			sc_builder = new MhaSceneBuilder(scene_path);
		}
#endif
		else if (scene_type == "Raw") {
			sc_builder = new RawSceneBuilder(scene_path);
		}
		else {
			throw(" Error : scene type not correct");
		}

		SceneAssembler assembler;
		assembler.setBuilder(sc_builder);
		Config config(assembler.getScene());

		auto alg = variables_map_["algorithm"].as<std::string>();
		std::cout << print_prefix << "algorithm set to "
			<< alg << ".\n";
		config.algorithmConfig(config.getAlgorithm(alg));

		auto ker = variables_map_["kernel"].as<std::string>();
		std::cout << print_prefix << "kernel set to "
			<< ker << ".\n";
		config.kernelConfig(config.getKernel(ker));
		
		config.path_tracing_config.iterations = variables_map_["iterations"].as<unsigned int>();
		std::cout << print_prefix << "iterations set to "
			<< config.path_tracing_config.iterations << ".\n";
		
		config.interactive = variables_map_["interactive"].as<bool>();
		config.test_trials = variables_map_["trials"].as<unsigned int>();
		
		auto n_of_tiles = variables_map_["number-of-tiles"].as<std::vector<unsigned int>>();
		if(n_of_tiles.size() == 1) n_of_tiles.push_back(n_of_tiles[0]) ;

		if (variables_map_.count("resolution")) {

			auto resolution = variables_map_["resolution"].as<std::vector<unsigned int>>();
			if (resolution.size() == 1) resolution.push_back(resolution[0]);

			config.resolution = { resolution[0],resolution[1] };
			config.tiling_config = TilingConfig(config.resolution, { n_of_tiles[0],n_of_tiles[1] });
			auto camera = config.scene.getCamera();
			camera->setResolution(resolution[0], resolution[1]);
		
		}
		else {
			config.tiling_config = TilingConfig(config.resolution, { n_of_tiles[0],n_of_tiles[1] });
		}

		CudaConfig cuda_config;
		cuda_config.unified_memory = variables_map_["use-unified-memory"].as<bool>();
		cuda_config.device_properties = findCudaDevice(cuda_config.devId);
		config.cudaConfig(cuda_config);

		if (variables_map_.count("output")) {
			config.output_name = variables_map_["output"].as<std::string>();
		}
		else {
			config.output_name = config.toString();
		}

		delete sc_builder;
    	return config;
    }

    bool contain(std::string str){
    	return (bool) variables_map_.count(str.c_str());
    }

};


#endif /* CONFIG_H_ */
