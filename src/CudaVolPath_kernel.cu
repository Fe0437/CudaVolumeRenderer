#include "CudaVolPath.cuh"
#include "Intersect.h"

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

void CudaVolPath::copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix) {
	cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix);
}

//generates local orthonormal basis around _dir
__device__ void generateLocalBasis(const float3 &_dir, float3 &_vec1, float3 &_vec2)
{
	float invNorm1 = 1.0f / sqrtf(_dir.x * _dir.x + _dir.z * _dir.z);
	_vec1 = make_float3(_dir.z * invNorm1, 0.0f, -_dir.x * invNorm1);
	_vec2 = cross(_dir, _vec1);
}

__device__ inline float3 sphericalDirection(float _sinTheta, float _cosTheta, float _phi,
	const float3 &_x, const float3 &_y, const float3 &_z)
{
	return _sinTheta * cosf(_phi) * _x +
		_sinTheta * sinf(_phi) * _y +
		_cosTheta * _z;
}

__device__ inline float PhaseHG(float _cosTheta, float _g)
{
	return INV_FOURPI * (1.0f - _g * _g) / powf(1.0f + _g * _g - 2.0f * _g * _cosTheta, 1.5f);
}

__device__ inline float PhaseHG(const float3 &_vecIn, float3 &_vecOut, float _g)
{
	float cosTheta = dot(_vecIn, _vecOut);
	return PhaseHG(cosTheta, _g);
}

__device__ inline float PdfHG(float _cosTheta, float _g)
{
	return PhaseHG(_cosTheta, _g);
}

__device__ inline float PdfHG(const float3 &_vecIn, float3 &_vecOut, float _g)
{
	return PhaseHG(_vecIn, _vecOut, _g);
}

__device__ float3 ImportanceSampleHG(const float3 &_v, float _g, float e1, float e2)
{
	float cosTheta;
	if (fabsf(_g) > 0.001f)
	{
		float sqrTerm = (1.0f - _g * _g) / (1.0f - _g + 2.0f * _g * e1);
		cosTheta = (1.0f + _g * _g - sqrTerm * sqrTerm) / (2.0f * fabsf(_g));
	}
	else
	{
		cosTheta = 1.0f - 2.0f * e1;
	}

	float sinTheta = sqrtf(max(0.0f, 1.0f - cosTheta * cosTheta));
	float phi = TWOPI * e2;

	float3 v1, v2;
	generateLocalBasis(_v, v1, v2);

	return sphericalDirection(sinTheta, cosTheta, phi, v1, v2, _v);
}


//------------------------------------------------------distance sampling--------------------------------------------------------------------

//analytical
__device__ inline float sampleDistance(float e, float sigma) {
	return -logf(1.0f - e) / sigma;
}

__device__ inline float woodcockStep(float _dMaxInv, float _xi)
{
	return -logf(max(_xi, 0.00001f)) * _dMaxInv;
}

__device__ float woodcockTracking(float3 _origin,
	float3 _dir,
	float _t1,
	float _maxSigmaT,
	cudaTextureObject_t _texDensityVolume,
	float3 _volumeExtent,
	curandStateXORWOW_t* state)
{
	float3 invVolumeExtent = 1.0f / (_volumeExtent);
	float invMaxSigmaT = 1.0f / _maxSigmaT;
	float eventDensity = 0.0f;
	float t = 0.0;
	float3 posUV;

	do
	{
		t += woodcockStep(invMaxSigmaT, curand_uniform(state));

		posUV = (_origin + (t * _dir))  * invVolumeExtent;
		eventDensity = tex3D<float>(_texDensityVolume, posUV.x + 0.5f, posUV.y + 0.5f, posUV.z + 0.5f);

	} while (t <= _t1 && eventDensity * invMaxSigmaT < curand_uniform(state));

	return t;
}

// transform vector by matrix (no translation)
__host__ __device__
float3 mul(const float3x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__host__ __device__
float4 mul(const float3x4 &M, const float4 &v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

inline __device__ void atomicAdd(float4* a, float4 b)
{
	atomicAdd(&a->x, b.x);
	atomicAdd(&a->y, b.y);
	atomicAdd(&a->z, b.z);
}


__global__ void d_initRays(
	Ray* d_rays,
	int* d_active_pixels,
	float4* d_output,
	float2 tile_start,
	float2 tile_dim,
	float2 resolution,
	float3 box_min,
	float3 box_max,
	curandStateXORWOW_t *d_states,
	int sample_per_pass,
	int samples
)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= tile_dim.x) || (y >= tile_dim.y)) return;

	int start_ray_id = (x + tile_dim.x * y)*sample_per_pass;
	int im_x = x + tile_start.x;
	int im_y = y + tile_start.y;
	int img_id = im_x + (resolution.x * im_y);
	int ray_id;

	curandStateXORWOW_t state;

	for (int i = 0; i<sample_per_pass; i++) {

		ray_id = start_ray_id + i;
		state = d_states[ray_id];

		float fx = im_x + curand_uniform(&state);
		float fy = im_y + curand_uniform(&state);

		float u = (fx / (float)resolution.x)*2.0f - 1.0f;
		float v = (fy / (float)resolution.y)*2.0f - 1.0f;

		float fovX = 0.006;
		float fovY = (resolution.y / resolution.x) *fovX;
		u = u * tanf(fovX * PI / 360.f);
		v = v * tanf(fovY * PI / 360.f);

		// calculate eye ray in world space
		Ray eyeRay;
		eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		eyeRay.d = normalize(make_float3(u, v, 1.0f));
		eyeRay.d = mul(c_invViewMatrix, eyeRay.d);
		eyeRay.throughput = make_float4(1.0f);

		//printf("eyeray created \n");
		/*
		* orthographic camera (not working for mitsbua)
		eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(u, v, 0.0f, 1.0f)));
		eyeRay.d = normalize(make_float3(0, 0, -1.0f));
		eyeRay.d = mul(c_invViewMatrix, eyeRay.d);
		*/

		float tnear, tfar;
		//printf("box_min %f %f %f \n",  box_min.x, box_min.y, box_min.z);
		//printf("box_max %f %f %f \n",  box_max.x, box_max.y, box_max.z);
		int hit = intersectBox(eyeRay, box_min, box_max, &tnear, &tfar);
		//printf("hit intersected %d , box_min %f %f %f , box_max %f %f %f \n", hit, box_min.x, box_min.y, box_min.z,  box_max.x, box_max.y, box_max.z);
		if (!hit) {
			d_active_pixels[ray_id] = -1;
			//printf("pixel taken \n");
			//d_output[img_id]+= make_float4(1,1,1,1) * 1/samples;
			atomicAdd(&d_output[img_id], make_float4(1, 1, 1, 1) * 1.0 / samples);
			//printf("add done \n");
			continue;
		}

		if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane
		float eps = 0.000001f;
		eyeRay.o = eyeRay.o + eyeRay.d*(tnear + eps);
		//printf("img_id %d \n",  img_id);
		d_active_pixels[ray_id] = img_id;
		d_rays[ray_id] = eyeRay;
		d_states[ray_id] = state;
	}
}



__global__ void
d_traceRay(
	Ray* d_rays,
	int* d_activePixels,
	int n_active_pixels,
	float4 *d_output,
	float3 boxMin,
	float3 boxMax,
	float brightness,
	float max_density,
	curandStateXORWOW_t *d_states,
	int steps_per_call,
	int samples,
	cudaTextureObject_t albedo_tex,
	cudaTextureObject_t density_tex
)
{
	uint ray_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (ray_id >= n_active_pixels) return;

	int img_id = d_activePixels[ray_id];
	if (img_id == -1) return;

	curandStateXORWOW_t state = d_states[ray_id];
	Ray ray = d_rays[ray_id];
	float3 range = boxMax - boxMin;

	// find intersection with box
	float tnear, tfar;
	float eps = 0.000001f;
	float min_throughput = eps;
	float s = 0;

	for (int i = 0; i<steps_per_call; i++) {

		float e0 = curand_uniform(&state), e1 = curand_uniform(&state), e2 = curand_uniform(&state);

		int hit = intersectBox(ray, boxMin, boxMax, &tnear, &tfar);
		//printf("hit intersected %d , box_min %f %f %f , box_max %f %f %f \n", hit, boxMin.x, boxMin.y, boxMin.z,  boxMax.x, boxMax.y, boxMax.z);

		float t;
		if (tnear>eps) t = tnear;
		else t = tfar;

		/*s = woodcockTracking(
		ray.o,
		ray.d,
		t,
		max_density,
		density_tex,
		range,
		&state
		);*/

		s = sampleDistance(curand_uniform(&state), 100);

		if (!hit || (tfar<-eps && tnear<-eps) || (tnear>eps && tfar>eps) || (s>t)) {
			d_output[img_id] += ray.throughput * 1.0 / samples;
			//atomicAdd(&d_output[img_id], ray.throughput * 1.0/samples);
			d_activePixels[ray_id] = -1;
			break;
		}

		ray.o = ray.o + ray.d * s;

		float3 coord = ray.o / range;
		float4 albedo = tex3D<float4>(albedo_tex, coord.x + 0.5, coord.y + 0.5, coord.z + 0.5);
		//printf("albedo  %f %f %f \n", albedo.x, albedo.y, albedo.z, albedo.w );
		ray.throughput = ray.throughput * albedo;

		/*
		*------survived rays----
		*
		float pSurvive = fmin(1.f, fmax(ray.throughput));
		float roulette = curand_uniform(&state);

		if(roulette > pSurvive){
		d_activePixels[ray_id] = -1;
		break;
		}

		ray.throughput = ray.throughput * 1.0/pSurvive;
		*/

		ray.d = ImportanceSampleHG(ray.d, 0.0, e1, e2);
	}


	d_rays[ray_id] = ray;
	d_states[ray_id] = state;

}

