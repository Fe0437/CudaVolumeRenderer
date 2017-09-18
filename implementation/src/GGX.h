#ifndef GGX_H_
#define GGX_H_

#include "Defines.h"
#include "helper_cuda.h"
#include "helper_math.h"

/*
BSDF UTILS
*/

/// Full Fresnel equations
__host__ __device__ inline float fresnelDielectric(float eta, float ndotwi,
                                                   float* p_ndotwt) {
  if (eta == 1) {
    *p_ndotwt = -ndotwi;
    return 0.0;
  }

  float scale = (ndotwi > 0) ? 1 / eta : eta;
  float sin_sqr = (1 - (ndotwi * ndotwi));
  float ndotwt_sqr = 1 - (sin_sqr * scale * scale);

  if (ndotwt_sqr <= 0.0F) {
    *p_ndotwt = 0.0F;
    return 1.0F;
  }

  float abs_ndotwi = fabsf(ndotwi);
  float abs_ndotwt = sqrtf(ndotwt_sqr);

  float Rs = (abs_ndotwi - eta * abs_ndotwt) / (abs_ndotwi + eta * abs_ndotwt);
  float Rp = (eta * abs_ndotwi - abs_ndotwt) / (eta * abs_ndotwi + abs_ndotwt);

  *p_ndotwt = (ndotwi > 0) ? -abs_ndotwt : abs_ndotwt;

  return 0.5F * (Rs * Rs + Rp * Rp);
}

__host__ __device__ inline void reflect(float ndotwi, float3 wi, float3 wh,
                                        float3* wo) {
  *wo = 2.F * (ndotwi)*wh - wi;
}

__host__ __device__ inline void refract(float eta, float ndotwi, float ndotwt,
                                        float3 wi, float3 n, float3* wo) {
  if (ndotwt < 0) eta = 1 / eta;

  *wo = n * (ndotwi * eta + ndotwt) - wi * eta;
}

/// Full Fresnel equations
__host__ __device__ inline float fresnelDielectric(float etai, float etat,
                                                   float ndotwi, float ndotwt) {
  // Parallel and perpendicular polarization
  float rparl =
      ((etat * ndotwi) - (etai * ndotwt)) / ((etat * ndotwi) + (etai * ndotwt));
  float rperp =
      ((etai * ndotwi) - (etat * ndotwt)) / ((etai * ndotwi) + (etat * ndotwt));
  return (rparl * rparl + rperp * rperp) * 0.5F;
}

/*
GGX
*/

// Distribution function
__host__ __device__ inline float GGX_D(float roughness, float3 m) {
  float ndotm = fabs(m.z);
  float ndotm2 = ndotm * ndotm;
  float sinmn = sqrtf(1.F - clamp(ndotm * ndotm, 0.F, 1.F));
  float tanmn = ndotm > EPSILON ? sinmn / ndotm : 0.F;
  float a2 = roughness * roughness;
  float denom =
      (PI * ndotm2 * ndotm2 * (a2 + tanmn * tanmn) * (a2 + tanmn * tanmn));
  return denom > DENOM_EPS ? (a2 / denom) : 1.F;
}

/**
 * \brief Visible normal sampling code for the alpha=1 case
 *
 * Source: supplemental material of "Importance Sampling
 * Microfacet-Based BSDFs using the Distribution of Visible Normals"
 */
__host__ __device__ inline float2 sampleVisible11(float thetaI, float2 sample) {
  const float SQRT_PI_INV = 1 / sqrtf(PI);
  float2 slope;

  /* Special case (normal incidence) */
  float phi = 2 * PI * sample.y;
  if (thetaI < 1e-4F) {
    float sinPhi;
    float cosPhi;
    float r = sqrtf(fmaxf(0.0F, sample.x / (1 - sample.x)));
    sinPhi = sin(phi);
    cosPhi = cos(phi);
    return make_float2(r * cosPhi, r * sinPhi);
  }

  /* Precomputations */
  float tanThetaI = tan(thetaI);
  float a = 1 / tanThetaI;
  a = 1.0F + (1.0F / (a * a));
  float G1 = 2.0F / (1.0F + sqrtf(a));

  /* Simulate X component */
  float A = (2.0F * sample.x / G1) - 1.0F;
  if (abs(A) == 1) {
    A -= copysignf((float)1.0, A) * EPSILON;
  }

  float tmp = 1.0F / (A * A - 1.0F);
  float B = tanThetaI;
  float D = sqrtf(fmaxf(0.0F, (B * B * tmp * tmp) - ((A * A - B * B) * tmp)));
  float slope_x_1 = (B * tmp) - D;
  float slope_x_2 = (B * tmp) + D;
  slope.x = (A < 0.0F || slope_x_2 > 1.0F / tanThetaI) ? slope_x_1 : slope_x_2;

  /* Simulate Y component */
  float S = NAN;
  if (sample.y > 0.5F) {
    S = 1.0F;
    sample.y = 2.0F * (sample.y - 0.5F);
  } else {
    S = -1.0F;
    sample.y = 2.0F * (0.5F - sample.y);
  }

  /* Improved fit */
  float z =
      (sample.y * (sample.y * (sample.y * (-(float)0.365728915865723) +
                               (float)0.790235037209296) -
                   (float)0.424965825137544) +
       (float)0.000152998850436920) /
      (sample.y * (sample.y * (sample.y * (sample.y * (float)0.169507819808272 -
                                           (float)0.397203533833404) -
                               (float)0.232500544458471) +
                   (float)1) -
       (float)0.539825872510702);

  slope.y = S * z * sqrt(1.0F + (slope.x * slope.x));

  return slope;
}

__host__ __device__ inline void mitsuba_GGX_sampleVNDF(float3 _wi, float2 alpha,
                                                       float2 sample,
                                                       float3* wo) {
  /* Step 1: stretch wi */
  float3 wi = normalize(make_float3(alpha.x * _wi.x, alpha.y * _wi.y, _wi.z));

  /* Get polar coordinates */
  float theta = 0;
  float phi = 0;
  if (wi.z < (float)0.999999) {
    theta = acos(wi.z);
    phi = atan2(wi.y, wi.x);
  }
  float sinPhi;
  float cosPhi;
  sinPhi = sin(phi);
  cosPhi = cos(phi);

  /* Step 2: simulate P22_{wi}(slope.x, slope.y, 1, 1) */
  float2 slope = sampleVisible11(theta, sample);

  /* Step 3: rotate */
  slope = make_float2((cosPhi * slope.x) - (sinPhi * slope.y),
                      (sinPhi * slope.x) + (cosPhi * slope.y));

  /* Step 4: unstretch */
  slope.x *= alpha.x;
  slope.y *= alpha.y;

  /* Step 5: compute normal */
  float normalization =
      1.F / sqrtf((slope.x * slope.x) + (slope.y * slope.y) + 1.0F);

  *wo = make_float3(-slope.x * normalization, -slope.y * normalization,
                    normalization);
}

/*
code from https://hal.archives-ouvertes.fr/hal-01509746/document
*/
__host__ __device__ inline void GGX_sampleVNDF(float3 wi, float2 alpha,
                                               float2 sample, float3* wo) {
  float3 alpha_extended = make_float3(alpha, 1);
  float3 streched_wi = normalize(wi * alpha_extended);

  // generate orthonormal basis
  float3 T1 = (wi.z < 0.9999) ? normalize(cross(wi, make_float3(0, 0, 1)))
                              : make_float3(1, 0, 0);
  float3 T2 = cross(T1, wi);

  // sample point with polar coordinates (r, phi)
  float a = 1.0 / (1.0 + wi.z);
  float r = sqrtf(sample.x);
  float phi =
      (sample.y < a) ? sample.y / a * PI : PI + (sample.y - a) / (1.0 - a) * PI;
  float P1 = r * cosf(phi);
  float P2 = r * sin(phi) * ((sample.y < a) ? 1.0 : wi.z);

  // compute normal
  float3 N = (P1 * T1) + (P2 * T2) +
             (sqrtf(max(0.0, 1.0 - (P1 * P1) - (P2 * P2))) * streched_wi);

  *wo = normalize(make_float3(alpha.x * N.x, alpha.y * N.y, max(0.0, N.z)));
}

// from mitsuba code
/// Compute the effective roughness projected on direction \c v
__host__ __device__ inline float projectRoughness(const float3& v,
                                                  const float2& alpha) {
  float invSinTheta2 = 1 / (1.0f - v.z * v.z);

  if (alpha.x == alpha.y || invSinTheta2 <= 0) {
    return alpha.x;
  }

  float cosPhi2 = v.x * v.x * invSinTheta2;
  float sinPhi2 = v.y * v.y * invSinTheta2;

  return sqrtf((cosPhi2 * alpha.x * alpha.x) + (sinPhi2 * alpha.y * alpha.y));
}

__host__ __device__ inline float GGX_G1(const float2& alpha, const float3& v,
                                        const float3& m) {
  /* Ensure consistent orientation (can't see the back
  of the microfacet from the front and vice versa) */
  if (dot(v, m) * v.z <= 0) {
    return 0.0F;
  }

  // tantheta
  float temp = 1 - (v.z * v.z);
  if (temp <= 0.0F) {
    return 0.0F;
  }

  float tan = sqrtf(temp) / v.z;

  /* Perpendicular incidence -- no shadowing/masking */
  tan = abs(tan);
  if (tan == 0.0F) {
    return 1.0F;
  }

  float proj_alpha = projectRoughness(v, alpha);
  float root = proj_alpha * tan;

  // hypot2
  float hypot2_root = sqrtf(1.0F + (root * root));
  return 2.0F / (1.0F + hypot2_root);
}

// Shadowing function also depends on microfacet distribution
__host__ __device__ inline float GGX_G(float2 roughness, float3 wi, float3 wo,
                                       float3 wh) {
  return GGX_G1(roughness, wi, wh) * GGX_G1(roughness, wo, wh);
}

#include "Rng.h"

__host__ __device__ inline bool GGX_sample(float2 roughness,
                                           // intIor / extIor
                                           float eta,
                                           // Incoming direction
                                           float3 wi,
                                           // Sample
                                           Rng* rng,
                                           // Outgoing  direction
                                           float3* wo,
                                           // PDF at wo
                                           float* weight) {
  float ndotwi = wi.z;
  if (ndotwi == 0.F) {
    *weight = 0;
    return false;
  }

  *weight = 1.0F;

  volatile float sign = wi.z / abs(wi.z);

  float3 wh;

#ifdef MITSUBA_COMPARABLE
  mitsuba_GGX_sampleVNDF(sign * wi, roughness, rng->getFloat2(), &wh);
#else
  GGX_sampleVNDF(sign * wi, roughness, rng->getFloat2(), &wh);
#endif

  float whdotwt = NAN;
  float whdotwi = dot(wh, wi);
  volatile float F = fresnelDielectric(eta, whdotwi, &whdotwt);

  if (rng->getFloat() <= F) {
    // REFLECTION
    reflect(whdotwi, wi, wh, wo);

    /* check */
    if (wi.z * wo->z <= 0) {
      *weight = 0.0F;
      return false;
    }
  } else {  // REFRACTION

    /* check */
    if (whdotwt == 0.0F) {
      *weight = 0.0F;
      return false;
    }

    refract(eta, whdotwi, whdotwt, wi, wh, wo);

    /* check */
    if (wi.z * wo->z >= 0) {
      *weight = 0.0F;
      return false;
    }
  }

  *weight *= GGX_G1(roughness, *wo, wh);
  return true;
}

#endif  // GGX_H_
