#include "ConfigParser.h"

#include <iostream>
#include <algorithm>  // for std::transform

#include "RawSceneBuilder.h"
#include "VDBSceneBuilder.h"
#include "XmlSceneBuilder.h"

auto ConfigParser::parseCommandline(int argc, char** argv) -> bool {
  namespace po = boost::program_options;

  po::options_description generic("Generic:");
  generic.add_options()("help,h", "produce help message")(
      "scene-file,s", po::value<std::string>(), "scene file to parse")(
      "scene-type", po::value<std::string>()->default_value("Auto"),
      "scene type to parse, possible values: \n\t Auto \n\t MitsubaXml \n\t Vdb \n\t "
      "Raw")("interactive", po::value<bool>()->default_value(true),
             "run the algorithm with the interactive view")(
      "trials", po::value<unsigned int>()->default_value(1),
      "number of times to run the algorithm")(
      "algorithm,a", po::value<std::string>()->default_value("cudaVolPath"),
      "algorithm to use")(
      "kernel,k", po::value<std::string>()->default_value("regenerationSK"),
      "kernel to use")(
      "number-of-tiles",
      po::value<std::vector<unsigned int>>()->multitoken()->default_value(
          std::vector<unsigned int>{1, 1}, "1 1"),
      "number of tiles")("use-unified-memory",
                         po::value<bool>()->default_value(false),
                         "use unified memory");

  po::options_description scene_override("Scene configuration override:");
  scene_override.add_options()("iterations,i",
                               po::value<unsigned int>()->default_value(20),
                               "number of iterations")(
      "output,o", po::value<std::string>(), "output name")(
      "resolution,r",
      po::value<std::vector<unsigned int>>()->multitoken()->default_value(
          std::vector<unsigned int>{1024, 1024}, "1024 1024"),
      "resolution");

  po::options_description all_options;
  all_options.add(generic).add(scene_override);

  po::positional_options_description pos;
  pos.add("scene-file", -1);

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(all_options)
                  .positional(pos)
                  .run(),
              variables_map_);
    po::notify(variables_map_);

    if (variables_map_.count("help")) {
      std::cout << all_options << "\n";
      return false;
    }
  } catch (std::exception& e) {
    std::cerr << print_prefix << "Error: " << e.what() << "\n";
    return false;
  }

  return true;
}

auto ConfigParser::createConfig() -> Config {
  // Create scene
  if (!variables_map_.count("scene-file")) {
    throw std::runtime_error("Error: no scene file provided");
  }

  std::string scene_path = variables_map_["scene-file"].as<std::string>();
  std::string scene_type = variables_map_["scene-type"].as<std::string>();

  // Add auto-detection of scene type based on file extension
  if (scene_type == "Auto") {
    size_t dot_pos = scene_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
      std::string extension = scene_path.substr(dot_pos + 1);
      // Convert extension to lowercase for case-insensitive comparison
      std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
      
      if (extension == "xml") {
        scene_type = "MitsubaXml";
      } else if (extension == "vdb") {
        scene_type = "Vdb";
      } else {
        scene_type = "Raw";
      }
    } else {
      scene_type = "Raw"; // Default to Raw if no extension found
    }
    std::cout << print_prefix << "Auto-detected scene type: " << scene_type << "\n";
  }

  std::unique_ptr<SceneBuilder> sc_builder;
  if (scene_type == "MitsubaXml") {
    sc_builder = std::make_unique<XmlSceneBuilder>(scene_path);
  } else if (scene_type == "Vdb") {
    sc_builder = std::make_unique<VDBSceneBuilder>(scene_path);
  } else if (scene_type == "Raw") {
    sc_builder = std::make_unique<RawSceneBuilder>(scene_path);
  } else {
    throw std::runtime_error("Error: scene type not correct");
  }

  SceneAssembler assembler{};
  assembler.setBuilder(std::move(sc_builder));
  Config config{assembler.getScene()};

  auto alg = variables_map_["algorithm"].as<std::string>();
  std::cout << print_prefix << "algorithm set to " << alg << ".\n";
  config.algorithmConfig(Config::getAlgorithm(alg));

  auto ker = variables_map_["kernel"].as<std::string>();
  std::cout << print_prefix << "kernel set to " << ker << ".\n";
  config.kernelConfig(Config::getKernel(ker));

  config.path_tracing_config.iterations =
      variables_map_["iterations"].as<unsigned int>();
  std::cout << print_prefix << "iterations set to "
            << config.path_tracing_config.iterations << ".\n";

  config.interactive = variables_map_["interactive"].as<bool>();
  config.test_trials = variables_map_["trials"].as<unsigned int>();

  auto n_of_tiles =
      variables_map_["number-of-tiles"].as<std::vector<unsigned int>>();
  if (n_of_tiles.size() == 1) {
    n_of_tiles.push_back(n_of_tiles[0]);
  }

  if (variables_map_.count("resolution")) {
    auto resolution =
        variables_map_["resolution"].as<std::vector<unsigned int>>();
    if (resolution.size() == 1) {
      resolution.push_back(resolution[0]);
    }

    config.resolution = {resolution[0], resolution[1]};
    config.tiling_config =
        TilingConfig(config.resolution, {n_of_tiles[0], n_of_tiles[1]});
    auto camera = config.scene.getCamera();
    camera->setResolution(resolution[0], resolution[1]);
  } else {
    config.tiling_config =
        TilingConfig(config.resolution, {n_of_tiles[0], n_of_tiles[1]});
  }

  CudaConfig cuda_config;
  cuda_config.unified_memory = variables_map_["use-unified-memory"].as<bool>();
  cuda_config.device_properties = findCudaDevice(cuda_config.devId);
  config.cudaConfig(cuda_config);

  if (variables_map_.count("output")) {
    config.output_name = variables_map_["output"].as<std::string>();
  } else {
    config.output_name = config.toString();
  }

  return config;
}