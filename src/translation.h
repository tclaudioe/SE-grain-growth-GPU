#ifndef TRANSLATION_H
#define TRANSLATION_H

#include "geometry.h"
#include "utils.h"

/**
 * Translation Module.
 * Manages memory between host and device.
 */

/**
 * Adjust the pointers of all the data structure. If the data is passed from host to device,
 * then the pointes in device structs are pointing to host and must be corrected.
 * If the data is passed from device to host, pointers are pointing to device and must be corrected.
 *
 * @param dev_vrt_start   Pointer to vertices array in device
 * @param vrt_start       Pointer to vertices array in host
 * @param n_vertices      Number of vertices
 * @param dev_bnd_start   Pointer to boundaries array in device
 * @param bnd_start       Pointer to boundaries array in host
 * @param n_boundaries    Number of boundaries
 * @param dev_grain_start Pointer to grains array in device
 * @param grain_start     Pointer to grains array in host
 * @param n_grains        Number of grains
 * @param rollback        If true, the data is passed from device to host. Othewise the data is passed
 *                        from host to device
 */
__global__ void adjust_pointers(vertex *dev_vrt_start, vertex *vrt_start, int n_vertices,
                              boundary *dev_bnd_start, boundary *bnd_start, int n_boundaries,
                              grain *dev_grain_start, grain *grain_start, int n_grains, bool rollback) {
    int stid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid;
    /* Translate vertices pointers */
    tid = stid;
    while(tid < n_vertices) {
        if(rollback) {
            for(int i = 0; i < 3; i++) {
                // Boundary array
                int bpos = dev_vrt_start[tid].boundaries[i] - dev_bnd_start;
                dev_vrt_start[tid].boundaries[i] = &bnd_start[bpos];
                // Grain array
                int gpos = dev_vrt_start[tid].grains[i] - dev_grain_start;
                dev_vrt_start[tid].grains[i] = &grain_start[gpos];
            }
            int voted_pos = dev_vrt_start[tid].voted - dev_bnd_start;
            dev_vrt_start[tid].voted = &bnd_start[voted_pos];
            //int grn_pos = dev_vrt_start[tid].grn - dev_grain_start;
            //dev_vrt_start[tid].grn = &grain_start[grn_pos];
        } else {
            for(int i = 0; i < 3; i++) {
                // Boundary array
                int bpos = dev_vrt_start[tid].boundaries[i] - bnd_start;
                dev_vrt_start[tid].boundaries[i] = &dev_bnd_start[bpos];
                // Grain array
                int gpos = dev_vrt_start[tid].grains[i] - grain_start;
                dev_vrt_start[tid].grains[i] = &dev_grain_start[gpos];
            }
            int voted_pos = dev_vrt_start[tid].voted - bnd_start;
            dev_vrt_start[tid].voted = &dev_bnd_start[voted_pos];
        }
        tid += gridDim.x * blockDim.x;
    }
    /* Translate boundaries pointers */
    tid = stid;
    while(tid < n_boundaries) {
        if(rollback) {
            // Initial vertex
            int vrt_pos_ini = dev_bnd_start[tid].ini - dev_vrt_start;
            dev_bnd_start[tid].ini = &vrt_start[vrt_pos_ini];
            // End vertex
            int vrt_pos_end = dev_bnd_start[tid].end - dev_vrt_start;
            dev_bnd_start[tid].end = &vrt_start[vrt_pos_end];
        } else {
            // Initial vertex
            int vrt_pos_ini = dev_bnd_start[tid].ini - vrt_start;
            dev_bnd_start[tid].ini = &dev_vrt_start[vrt_pos_ini];
            // End vertex
            int vrt_pos_end = dev_bnd_start[tid].end - vrt_start;
            dev_bnd_start[tid].end = &dev_vrt_start[vrt_pos_end];
        }
        tid += gridDim.x * blockDim.x;
    }
    /* Translate grains pointers */
    tid = stid;
    while(tid < n_grains) {
        if(rollback) {
            // Vertices list
            for(int i = 0; i < dev_grain_start[tid].vlen; i++) {
                int vrt_pos = dev_grain_start[tid].vertices[i] - dev_vrt_start;
                dev_grain_start[tid].vertices[i] = &vrt_start[vrt_pos];
            }
        } else {
            // Vertices list
            for(int i = 0; i < dev_grain_start[tid].vlen; i++) {
                int vrt_pos = dev_grain_start[tid].vertices[i] - vrt_start;
                dev_grain_start[tid].vertices[i] = &dev_vrt_start[vrt_pos];
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Allocate device memory and check errors.
 *
 * @param dev_vertices   Vertices device array
 * @param n_vertices     Number of vertices
 * @param dev_boundaries Boundaries device array
 * @param n_boundaries   Number of boundaries
 * @param dev_grains     Grains device array
 * @param n_grains       Number of grains
 */
inline void allocate_device_memory(vertex *&dev_vertices, int n_vertices,
                                   boundary *&dev_boundaries, int n_boundaries,
                                   grain *&dev_grains, int n_grains) {
    HERR(cudaMalloc(&dev_vertices, sizeof(vertex) * n_vertices));
    CNULL(dev_vertices);
    HERR(cudaMalloc(&dev_boundaries, sizeof(boundary) * n_boundaries));
    CNULL(dev_boundaries);
    HERR(cudaMalloc(&dev_grains, sizeof(grain) * n_grains));
    CNULL(dev_grains);
}

/**
 * Translates the memory of vertices, boundaries and grains to device
 *
 * @param vertices       Host vertices device array
 * @param n_vertices     Number of vertices
 * @param boundaries     Host boundaries device array
 * @param n_boundaries   Number of boundaries
 * @param grains         Host grains device array
 * @param n_grains       Number of grains
 * @param dev_vertices   Vertices device array
 * @param dev_boundaries Boundaries device array
 * @param dev_grains     Grains device array
 * @param n_blocks       Number of CUDA blocks
 * @param n_threads      Number of CUDA threads
 */
inline void host_to_device(vertex *vertices, int n_vertices,
                           boundary *boundaries, int n_boundaries,
                           grain *grains, int n_grains, vertex* &dev_vertices,
                           boundary* &dev_boundaries, grain* &dev_grains,
                           int n_blocks, int n_threads) {
    // Allocate memory
    allocate_device_memory(dev_vertices, n_vertices, dev_boundaries, n_boundaries, dev_grains, n_grains);
    // Copy memory
    HERR(cudaMemcpy(dev_vertices, vertices, sizeof(vertex) * n_vertices, cudaMemcpyHostToDevice));
    HERR(cudaMemcpy(dev_boundaries, boundaries, sizeof(boundary) * n_boundaries, cudaMemcpyHostToDevice));
    HERR(cudaMemcpy(dev_grains, grains, sizeof(grain) * n_grains, cudaMemcpyHostToDevice));
    // Adjust pointers
    adjust_pointers<<<n_blocks, n_threads>>>(dev_vertices, vertices, n_vertices,
                                        dev_boundaries, boundaries, n_boundaries,
                                        dev_grains, grains, n_grains, false);
}

/**
 * Free device memory of the given arrays
 * @param dev_vertices   Vertices device array
 * @param dev_boundaries Boundaries device array
 * @param dev_grains     Grains device array
 */
inline void free_device_memory(vertex *dev_vertices, boundary *dev_boundaries, grain *dev_grains) {
    HERR(cudaFree(dev_vertices));
    HERR(cudaFree(dev_boundaries));
    HERR(cudaFree(dev_grains));
}

/**
 * Translates memory from device to host and repairs pointers.
 *
 * @param vertices       Host vertices device array
 * @param n_vertices     Number of vertices
 * @param boundaries     Host boundaries device array
 * @param n_boundaries   Number of boundaries
 * @param grains         Host grains device array
 * @param n_grains       Number of grains
 * @param dev_vertices   Vertices device array
 * @param dev_boundaries Boundaries device array
 * @param dev_grains     Grains device array
 * @param n_blocks       Number of CUDA blocks
 * @param n_threads      Number of CUDA threads
 */
inline void device_to_host(vertex *vertices, int n_vertices,
                           boundary *boundaries, int n_boundaries,
                           grain *grains, int n_grains, vertex* dev_vertices,
                           boundary* dev_boundaries, grain* dev_grains,
                           int n_blocks, int n_threads) {
    // Adjust pointers back to host
    adjust_pointers<<<n_blocks, n_threads>>>(dev_vertices, vertices, n_vertices,
                                        dev_boundaries, boundaries, n_boundaries,
                                        dev_grains, grains, n_grains, true);
    // Copy back data
    HERR(cudaMemcpy(vertices, dev_vertices, sizeof(vertex) * n_vertices, cudaMemcpyDeviceToHost));
    HERR(cudaMemcpy(boundaries, dev_boundaries, sizeof(boundary) * n_boundaries, cudaMemcpyDeviceToHost));
    HERR(cudaMemcpy(grains, dev_grains, sizeof(grain) * n_grains, cudaMemcpyDeviceToHost));
    // Free memory
    free_device_memory(dev_vertices, dev_boundaries, dev_grains);
}

#endif // TRANSLATION_H