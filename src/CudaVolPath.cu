/*
 * CudaVolPath.cpp
 *
 *  Created on: 30/ago/2017
 *      Author: Federico
 */

#include "CudaVolPath.cuh"
#include "Debug.h"


CudaVolPath::CudaVolPath(const Config& config, const glm::ivec2& n_tiles):config_(config){
	n_tiles_ = n_tiles;
	rendering_resolution_ = config_.getResolution();
	//int ray_bounces_per_batch = (int)ceil(config_.max_ray_bounces/(float)batch_bounces_);
	tile_dim_.x = (int)ceil(rendering_resolution_.x/n_tiles_.x);
	tile_dim_.y  = (int)ceil(rendering_resolution_.y/n_tiles_.y);
	n_active_pixels_ = tile_dim_.x * tile_dim_.y * config.sample_per_pass;

	initCamera();
	initRandom();
	initTextures();
}

CudaVolPath::~CudaVolPath() {
	releaseDeviceMemory();
}

void CudaVolPath::initCamera(){
	//camera_ = config_.getCamera();
	//setup the view matrix
	//glm::mat4 inv_view_mat = camera_.getInvViewMatrix();
	//float* invViewMatrix = glm::value_ptr(inv_view_mat);

	float invViewMatrix[12];
	float modelView[16] =
	    {
	        1.0f, 0.0f, 0.0f, 0.0f,
	        0.0f, 1.0f, 0.0f, 0.0f,
	        0.0f, 0.0f, 1.0f, 0.0f,
	        0.0f, 0.0f, -10005.0f, 1.0f
	    };

	invViewMatrix[0] = modelView[0];
	invViewMatrix[1] = modelView[4];
	invViewMatrix[2] = modelView[8];
	invViewMatrix[3] = modelView[12];
	invViewMatrix[4] = modelView[1];
	invViewMatrix[5] = modelView[5];
	invViewMatrix[6] = modelView[9];
	invViewMatrix[7] = modelView[13];
	invViewMatrix[8] = modelView[2];
	invViewMatrix[9] = modelView[6];
	invViewMatrix[10] = modelView[10];
	invViewMatrix[11] = modelView[14];


	// call CUDA kernel, writing results to PBO
	copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);
}

void CudaVolPath::initRandom(){

	dim3 block_size = config_.block_size_default;

	dim3 grid_size = dim3(iDivUp(rendering_resolution_.x, block_size.x), iDivUp(rendering_resolution_.y, block_size.y));

	CurandStatesFactory rand = CurandStatesFactory(grid_size, block_size);
	d_states_ = rand.createCurandStatesOnDevice(rendering_resolution_.x, rendering_resolution_.y);
}

// http://docs.thrust.googlecode.com/hg/group__counting.html
// http://docs.thrust.googlecode.com/hg/group__stream__compaction.html
struct isNegative
{
	__host__ __device__
	bool operator()(const int & x)
	{
		return x < 0;
	}
};


void CudaVolPath::initTextures(){
	Volume<ALBEDO_T> albedo_volume = config_.scene.getAlbedoVolume();
	Volume<FLOAT> density_volume = config_.scene.getDensityVolume();

	albedo_tex_ = createTextureWithVolume<ALBEDO_T>(d_albedo_array_, albedo_volume.getVolumeData(), albedo_volume.getCudaExtent());
	density_tex_ = createTextureWithVolume<FLOAT>(d_density_array_,  density_volume.getVolumeData(), density_volume.getCudaExtent());
}

template < class VolumeType>
cudaTextureObject_t CudaVolPath::createTextureWithVolume(cudaArray* d_array, VolumeType *h_volume, cudaExtent volumeSize)
{
	cudaTextureObject_t texture=0;
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    cudaMalloc3DArray(&d_array, &channelDesc, volumeSize);
    CHECK_CUDA_ERROR("allocated 3DArray");

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_array;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    CHECK_CUDA_ERROR("cudaMemcpy3D 3DArray");

    cudaResourceDesc    texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array  = d_array;
	cudaTextureDesc     texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&texture, &texRes, &texDescr, NULL);

    CHECK_CUDA_ERROR("created 3D texture object");
    return texture;

}


void CudaVolPath::initRendering(){

	int tot_pixels = rendering_resolution_.x*rendering_resolution_.y;
	checkCudaErrors(cudaMalloc((void **)&d_output_, tot_pixels * sizeof(float4)));
	checkCudaErrors(cudaMemset(d_output_, 0, tot_pixels * sizeof(float4)));
	checkCudaErrors(cudaMalloc(&d_active_pixels_, sizeof(int) * n_active_pixels_));
	checkCudaErrors(cudaMalloc(&d_rays_, sizeof(Ray) * n_active_pixels_));
	CHECK_CUDA_ERROR("init rendering");

}

void CudaVolPath::renderingLoop(){

	glm::ivec2 resolution = config_.getResolution();
	uint n_processed_pixels;
	uint tracing_grid_size;

	dim3 block_size = config_.block_size_default;
	uint tracing_block_size = block_size.x*block_size.y;
	dim3 grid_size = dim3(
						(tile_dim_.x + block_size.x - 1) / block_size.x ,
						(tile_dim_.y + block_size.y - 1) / block_size.y
						);

	float2 tile_start;
	tile_start.x=0;
	tile_start.y=0;

	glm::vec3 bmin = config_.scene.getBoxMin();
	float3 box_min = make_float3(bmin.x, bmin.y, bmin.z);

	glm::vec3 bmax = config_.scene.getBoxMax();
    float3 box_max = make_float3(bmax.x, bmax.y, bmax.z);

    float2 tile_dim = make_float2(tile_dim_.x, tile_dim_.y);
    float2 f2resolution = make_float2(rendering_resolution_.x, rendering_resolution_.y);

	for(; tile_start.x<rendering_resolution_.x; tile_start.x+=tile_dim_.x){
			for(; tile_start.y<rendering_resolution_.y; tile_start.y+=tile_dim_.y){

				for(int i=0; i<config_.passes; i++){
					n_processed_pixels = n_active_pixels_;
					tracing_grid_size = (n_active_pixels_ + tracing_block_size - 1) / tracing_block_size ;


					d_initRays<<<grid_size, block_size>>>(
							d_rays_,
							d_active_pixels_,
							d_output_,
							tile_start,
							tile_dim,
							f2resolution,
							box_min,
							box_max,
							d_states_,
							config_.sample_per_pass,
							config_.samples
							);
				    CHECK_CUDA_ERROR(" init rays");


					for(int j=0; j<10; j++){

						d_traceRay<<<tracing_grid_size, tracing_block_size>>>
								(
										d_rays_,
										d_active_pixels_,
										n_processed_pixels,
										d_output_,
										box_min,
										box_max,
										1,
										100,
										d_states_,
										10,
										config_.samples,
										albedo_tex_,
										density_tex_
								);

					    CHECK_CUDA_ERROR("traceRay");

						//thrust::device_ptr<int> device_pointer(d_active_pixels_);
						//thrust::device_ptr<int> end = thrust::remove_if(device_pointer, device_pointer + n_processed_pixels, isNegative());
						//n_processed_pixels = end.get() - d_active_pixels_;
						//tracing_grid_size = (n_processed_pixels + tracing_block_size - 1) / tracing_block_size;
					}

					//printf("pass :  %d \n", i);
					//checkCudaErrors(cudaDeviceSynchronize());
				}
				printf("tile :  %d ,  %d  \n", (int)(tile_start.x/tile_dim_.x), (int)(tile_start.y/tile_dim_.y));
			}
		}
}

void CudaVolPath::saveImage() {
	Image image(rendering_resolution_.x, rendering_resolution_.y);
    //float4 *f4_output = (float4 *)malloc( rendering_resolution_.x*rendering_resolution_.y*sizeof(float4));
	checkCudaErrors(cudaMemcpy(image.pixels, d_output_, rendering_resolution_.x*rendering_resolution_.y*sizeof(float4), cudaMemcpyDeviceToHost));
	image.saveHDR("test");
}

void CudaVolPath::releaseDeviceMemory() {
	checkCudaErrors(cudaFree(d_output_));
	checkCudaErrors(cudaFreeArray(d_albedo_array_));
	checkCudaErrors(cudaFreeArray(d_density_array_));
	checkCudaErrors(cudaFree(d_states_));
	checkCudaErrors(cudaFree(d_active_pixels_));
	checkCudaErrors(cudaFree(d_rays_));
}

void CudaVolPath::render() {
	initRendering();
	renderingLoop();
}
