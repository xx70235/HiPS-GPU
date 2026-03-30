/**
 * INDEX阶段GPU加速 - CUDA Kernel实现
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define TWOPI 6.283185307179586476925286766559
#define HALFPI 1.5707963267948966192313216916398
#define TWOTHIRD 0.66666666666666666666666666666667

__constant__ int d_x2pix[128];
__constant__ int d_y2pix[128];

extern "C" void initMortonTables() {
    int h_x2pix[128], h_y2pix[128];
    
    for (int i = 0; i < 128; i++) {
        int x = i, y = i;
        int xi = 0, yi = 0;
        
        for (int k = 0; k < 7; k++) {
            xi |= ((x & 1) << (2 * k));
            yi |= ((y & 1) << (2 * k + 1));
            x >>= 1;
            y >>= 1;
        }
        h_x2pix[i] = xi;
        h_y2pix[i] = yi;
    }
    
    cudaMemcpyToSymbol(d_x2pix, h_x2pix, sizeof(h_x2pix));
    cudaMemcpyToSymbol(d_y2pix, h_y2pix, sizeof(h_y2pix));
}

__device__ long xy2pix_device(int x, int y) {
    return d_x2pix[x & 127] | d_y2pix[y & 127] |
           (d_x2pix[(x >> 7) & 127] | d_y2pix[(y >> 7) & 127]) << 14 |
           (long)(d_x2pix[(x >> 14) & 127] | d_y2pix[(y >> 14) & 127]) << 28;
}

__device__ long ang2pix_nest_device(int nside, double theta, double phi) {
    double z = cos(theta);
    double za = fabs(z);
    
    double tt = fmod(phi, TWOPI);
    if (tt < 0) tt += TWOPI;
    tt = tt / HALFPI;
    
    long face_num, ix, iy;
    
    if (za <= TWOTHIRD) {
        double temp1 = nside * (0.5 + tt - z * 0.75);
        double temp2 = nside * (0.5 + tt + z * 0.75);
        
        long jp = (long)temp1;
        long jm = (long)temp2;
        
        long ifp = jp / nside;
        long ifm = jm / nside;
        
        if (ifp == ifm) {
            face_num = (ifp == 4) ? 4 : ifp + 4;
        } else if (ifp < ifm) {
            face_num = ifp + 4;
        } else {
            face_num = ifm + 8;
        }
        
        ix = jm & (nside - 1);
        iy = nside - (jp & (nside - 1)) - 1;
    } else {
        long ntt = (long)tt;
        if (ntt >= 4) ntt = 3;
        double tp = tt - ntt;
        
        double tmp;
        if (za < 0.99) {
            tmp = nside * sqrt(3.0 * (1.0 - za));
        } else {
            double sa = sqrt((1.0 - za) * 2.0);
            tmp = nside * sqrt(3.0) * sa / sqrt(2.0);
        }
        
        long jp = (long)(tp * tmp);
        long jm = (long)((1.0 - tp) * tmp);
        
        if (jp >= nside) jp = nside - 1;
        if (jm >= nside) jm = nside - 1;
        
        if (z >= 0) {
            face_num = ntt;
            ix = nside - jm - 1;
            iy = nside - jp - 1;
        } else {
            face_num = ntt + 8;
            ix = jp;
            iy = jm;
        }
    }
    
    long npface = (long)nside * nside;
    return face_num * npface + xy2pix_device((int)ix, (int)iy);
}

__device__ void pixelToCelestial_device(
    double px, double py,
    double crval1, double crval2,
    double crpix1, double crpix2,
    double cd1_1, double cd1_2,
    double cd2_1, double cd2_2,
    double* ra, double* dec
) {
    double dx = px - crpix1;
    double dy = py - crpix2;
    
    double xi = cd1_1 * dx + cd1_2 * dy;
    double eta = cd2_1 * dx + cd2_2 * dy;
    
    double xi_rad = xi * M_PI / 180.0;
    double eta_rad = eta * M_PI / 180.0;
    
    double crval1_rad = crval1 * M_PI / 180.0;
    double crval2_rad = crval2 * M_PI / 180.0;
    
    double cos_dec0 = cos(crval2_rad);
    double sin_dec0 = sin(crval2_rad);
    
    double rho = sqrt(xi_rad * xi_rad + eta_rad * eta_rad);
    double c = atan(rho);
    
    double sin_c = sin(c);
    double cos_c = cos(c);
    
    double dec_rad, ra_rad;
    
    if (rho < 1e-10) {
        dec_rad = crval2_rad;
        ra_rad = crval1_rad;
    } else {
        dec_rad = asin(cos_c * sin_dec0 + eta_rad * sin_c * cos_dec0 / rho);
        ra_rad = crval1_rad + atan2(xi_rad * sin_c, 
                                     rho * cos_dec0 * cos_c - eta_rad * sin_dec0 * sin_c);
    }
    
    *ra = ra_rad * 180.0 / M_PI;
    *dec = dec_rad * 180.0 / M_PI;
    
    while (*ra < 0.0) *ra += 360.0;
    while (*ra >= 360.0) *ra -= 360.0;
}

__global__ void batchPixelToHpxMultiOrderKernel(
    const double* __restrict__ crval1,
    const double* __restrict__ crval2,
    const double* __restrict__ crpix1,
    const double* __restrict__ crpix2,
    const double* __restrict__ cd1_1,
    const double* __restrict__ cd1_2,
    const double* __restrict__ cd2_1,
    const double* __restrict__ cd2_2,
    const int* __restrict__ sampleX,
    const int* __restrict__ sampleY,
    const int* __restrict__ imageIndices,
    int numSamples,
    int orderMax,
    long* __restrict__ npixResults
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSamples) return;
    
    int imgIdx = imageIndices[idx];
    
    double cv1 = crval1[imgIdx];
    double cv2 = crval2[imgIdx];
    double cp1 = crpix1[imgIdx];
    double cp2 = crpix2[imgIdx];
    double c11 = cd1_1[imgIdx];
    double c12 = cd1_2[imgIdx];
    double c21 = cd2_1[imgIdx];
    double c22 = cd2_2[imgIdx];
    
    double px = (double)sampleX[idx];
    double py = (double)sampleY[idx];
    
    double ra, dec;
    pixelToCelestial_device(px, py, cv1, cv2, cp1, cp2, c11, c12, c21, c22, &ra, &dec);
    
    double ra_rad = ra * M_PI / 180.0;
    double dec_rad = dec * M_PI / 180.0;
    double theta = HALFPI - dec_rad;
    double phi = ra_rad;
    
    while (phi < 0.0) phi += TWOPI;
    while (phi >= TWOPI) phi -= TWOPI;
    if (theta < 0.0) theta = 0.0;
    if (theta > M_PI) theta = M_PI;
    
    for (int order = 0; order <= orderMax; order++) {
        int nside = 1 << order;
        long npix = ang2pix_nest_device(nside, theta, phi);
        npixResults[idx * (orderMax + 1) + order] = npix;
    }
}

extern "C" void launchBatchPixelToHpxMultiOrderKernel(
    const double* d_crval1, const double* d_crval2,
    const double* d_crpix1, const double* d_crpix2,
    const double* d_cd1_1, const double* d_cd1_2,
    const double* d_cd2_1, const double* d_cd2_2,
    const int* d_sampleX, const int* d_sampleY,
    const int* d_imageIndices,
    int numSamples,
    int orderMax,
    long* d_npixResults,
    cudaStream_t stream
) {
    int blockSize = 256;
    int numBlocks = (numSamples + blockSize - 1) / blockSize;
    
    batchPixelToHpxMultiOrderKernel<<<numBlocks, blockSize, 0, stream>>>(
        d_crval1, d_crval2, d_crpix1, d_crpix2,
        d_cd1_1, d_cd1_2, d_cd2_1, d_cd2_2,
        d_sampleX, d_sampleY, d_imageIndices,
        numSamples, orderMax, d_npixResults
    );
}
