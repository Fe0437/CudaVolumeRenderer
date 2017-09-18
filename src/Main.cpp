/*
 ============================================================================
 Name        : CudaVolumeRenderer.cpp
 Author      : Federico Forti
 Version     :
 Copyright   : 
 Description :
 ============================================================================
 */

#ifdef WIN32
#include <windows.h>
#endif

//#include <GL/glew.h>
//#include <GLFW/glfw3.h>
//#include <glm/glm.hpp>
/*
#include "Defines.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>

#include <fstream>
#include <iostream>
using namespace std;

#include "helper_cuda_extension.h"
#include "Config.h"
#include "RenderCamera.h"
#include "CurandStatesFactory.cuh"

#include "CudaVolPath.cuh"
#include "UtilMathStructs.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

/*
void errorCallback(int error, const char* description) {
    fprintf(stderr, "%s\n", description);
}

void initGL(){

    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    config.window = glfwCreateWindow(config.width, config.height, "CUDA Path Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    //glfwSetKeyCallback(window, keyCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);
}

void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        runCuda();

        string title = "CUDA Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}
*/


/*Config initConfig(int argc, char **argv) {

	ConfigParser parser;
	parser.parseCommandline(argc, argv);
	printf("command line parsed \n");
	Config config = parser.createConfig();
	printf("command line parsed \n");

	if(config.interactive){
		//INIT GL
	}else{
		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		//
		findCudaDevice(config.devId);
	}

    return config;
}
*/



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
	/*DEVICE_RESET

	Config config = initConfig(argc, argv);

	glm::ivec2 vec = glm::ivec2(1,1);
	printf("config done \n");
	CudaVolPath volpath(config, vec);

	printf("rendering \n");
	volpath.render();
	printf("rendered \n");

	volpath.saveImage();

	cudaDeviceSynchronize();
	DEVICE_RESET

/*
	// Sets up framebuffer
	Framebuffer fbuffer;
	config.mFramebuffer = &fbuffer;

    //start logs

    char *ref_file = NULL;
	if (checkCmdLineFlag(argc, (const char **)argv, "file"))
	{
		getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
		fpsLimit = frameCheckNumber;
	}

	if (ref_file)
	{
		chooseCudaDevice(argc, (const char **)argv, false);
	}
	else
	{
		// First initialize OpenGL context, so we can properly set the GL for CUDA.
		// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
		initGL(&argc, argv);

		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		chooseCudaDevice(argc, (const char **)argv, true);
	}

	parseArguments(argc, argv);

    sdkCreateTimer(&timer);

    //printf("Press ']' and '[' to change brightness\n");

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
    initRandom(gridSize, blockSize, width, height, sample_per_pass);

    if (ref_file)
    {
        runSingleTest(ref_file, argv[0]);
    }
    else
    {
        // This is the normal rendering path for VolumeRender
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutReshapeFunc(reshape);
        glutIdleFunc(idle);

        initPixelBuffer();

#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        glutMainLoop();
    }
*/
	return 0;
}

