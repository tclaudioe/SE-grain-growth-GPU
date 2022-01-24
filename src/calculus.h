#ifndef CALCULUS_H
#define CALCULUS_H

#include "geometry.h"
#include "utils.h"

const double grad_eps = 1e-8;


/**
 * Compute the energy with a perturbed component of a vertex.
 * Only four of n_vertices are computed again.
 *
 * @param  comp           Indicates the perturbated component (x-component or y-component)
 * @param  vrt            Pointer to vertex struct
 * @param  DOMAIN_BOUND   Numerical domain bound
 * @return                Perturbed energy difference
 */
__device__ inline double energy_diff(const int comp, vertex *vrt, const double DOMAIN_BOUND) {
    // Get vertex position and perturb it
    vector2 new_pos = vrt->pos;
    if(comp == XCOMP)
        new_pos.x += grad_eps;
    else
        new_pos.y += grad_eps;
    new_pos = vector2_adjust(new_pos, DOMAIN_BOUND);

    // Compute perturbed arclengths
    double arclengths[3] = {0,0,0};
    for(int j = 0; j < 3; j++) {
        // Get the positions of the neighbor vertex
        if(vrt->boundaries[j]->ini == vrt) {
            arclengths[j] = compute_arclength(new_pos, vrt->boundaries[j]->end->pos, DOMAIN_BOUND);
        } else {
            arclengths[j] = compute_arclength(new_pos, vrt->boundaries[j]->ini->pos, DOMAIN_BOUND);
        }
    }

    // Compute perturbed areas
    double areas[3] = {0,0,0};
    for(int j = 0; j < 3; j++) {
        double grn_X[MAX_VRT_PER_GRN], grn_Y[MAX_VRT_PER_GRN];
        for(int k = 0; k < vrt->grains[j]->vlen; k++) {
            grn_X[k] = vrt->grains[j]->vertices[k]->pos.x;
            grn_Y[k] = vrt->grains[j]->vertices[k]->pos.y;
            // Replace the position of i-th vertex with perturbed position
            if(vrt->grains[j]->vertices[k] == vrt) {
                grn_X[k] = new_pos.x;
                grn_Y[k] = new_pos.y;
            }
        }
        // Save area of modified grain
        areas[j] = compute_area(grn_X, grn_Y, vrt->grains[j]->vlen, DOMAIN_BOUND);
    }

    // These are the corrected energy term difference
    double bnd_term = 0.0, grn_term = 0.0;
    for(int j = 0; j < 3; j++) {
        bnd_term += (vrt->boundaries[j]->energy * (arclengths[j] - vrt->boundaries[j]->arclength));
        grn_term += (vrt->grains[j]->SE * (abs(areas[j]) - abs(vrt->grains[j]->area)));
    }
    // Store each velocity term separated and already divided by grad_eps
    if(comp == XCOMP) {
        vrt->vel_bnd_x = -bnd_term / grad_eps;
        vrt->vel_grn_x = -grn_term / grad_eps;
    } else {
        vrt->vel_bnd_y = -bnd_term / grad_eps;
        vrt->vel_grn_y = -grn_term / grad_eps;
    }


    return bnd_term + grn_term;
}

/**
 * Compute all the arclengths of enabled boundaries
 * and stores them in the respective boundary structure.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param DOMAIN_BOUND   Numerical domain bound
 */
__global__ void compute_boundary_arclengths(boundary* dev_boundaries, int n_boundaries, const double DOMAIN_BOUND) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    boundary *bnd;
    while(tid < n_boundaries) {
        bnd = &dev_boundaries[tid];
        if(bnd->enabled) {
            bnd->arclength = boundary_compute_arclength(bnd, DOMAIN_BOUND);
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Compute all the energies of enabled boundaries
 * and stores them in the respective boundary structure.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param eps            Epsilon for GB energy function
 * @param energy_scaling Scale energy by some factor
 */
__global__ void compute_boundary_energies(boundary* dev_boundaries, int n_boundaries, double eps, double energy_scaling) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    boundary *bnd;
    while(tid < n_boundaries) {
        bnd = &dev_boundaries[tid];
        if(bnd->enabled) {
            bnd->energy = boundary_compute_energy(bnd, eps) * energy_scaling;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Compute all the energies of enabled vertices
 * and stores them in the respective vertex structure.
 *
 * @param dev_vertices Pointer to vertex device array
 * @param n_vertices   Number of vertices
 */
__global__ void compute_vertex_energies(vertex* dev_vertices, int n_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    vertex *vrt;
    while(tid < n_vertices) {
        vrt = &dev_vertices[tid];
        if(vrt->enabled) {
            vrt->energy = vertex_compute_energy(vrt);
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * For each perturbation of the components of each vertex, compute the total energy
 * of the system and stores it in a device array.
 *
 * @param dev_vertices   Pointer to vertex device array
 * @param n_vertices     Number of vertices
 * @param DOMAIN_BOUND   Numerical domain bound
 */
__global__ void compute_vertex_velocities(vertex* dev_vertices, int n_vertices, const double DOMAIN_BOUND) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    vertex *vrt;
    while(tid < n_vertices) {
        vrt = &dev_vertices[tid];
        if(vrt->enabled) {
            vrt->vel.x = -energy_diff(XCOMP, vrt, DOMAIN_BOUND)/grad_eps;
            vrt->vel.y = -energy_diff(YCOMP, vrt, DOMAIN_BOUND)/grad_eps;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Compute dihedral angles at vertices.
 *
 * @param dev_vertices   Pointer to vertex device array
 * @param n_vertices     Number of vertices
 * @param DOMAIN_BOUND   Numerical domain bound
 */
__global__ void compute_dihedral_angles(vertex *dev_vertices, int n_vertices, const double DOMAIN_BOUND) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    vertex *vrt;
    while(tid < n_vertices) {
        vrt = &dev_vertices[tid];
        if(vrt->enabled) {
            vector2 T[3];
            for(int i = 0; i < 3; i++) {
                // Compute unit tangent vector per boundary
                boundary *bnd = vrt->boundaries[i];
                T[i] = vector2_delta_to(bnd->end->pos, bnd->ini->pos, DOMAIN_BOUND);
                T[i] = vector2_unitary(T[i]);
                if(vrt == bnd->end) {
                    T[i].x *= -1;
                    T[i].y *= -1;
                }
            }
            double rule, angle;
            //double sum = 0;
            int j, k;
            for(int i = 0; i < 3; i++) {
                j = (i+1) % 3;
                k = (i+2) % 3;
                if(abs(T[i].x) == 0 && abs(T[i].y) == 0) {
                    // Compute the dihedral angle between the remaining vectors
                    vrt->angles[j] = acos(vector2_dot(T[j], T[k]));
                    vrt->angles[i] = 0.5*(2*M_PI - vrt->angles[j]);
                    vrt->angles[k] = 0.5*(2*M_PI - vrt->angles[j]);
                    break;
                } else {
                    rule = T[i].x * T[j].y - T[i].y * T[j].x;
                    angle = acos(vector2_dot(T[i], T[j]));
                    if(rule < 0) {
                        angle = 2*M_PI - angle;
                    }
                    vrt->angles[i] = angle;
                }
            }
            // final check for debug
            /*sum = vrt->angles[0] + vrt->angles[1] + vrt->angles[2];
            if(sum > 2*M_PI+1e-4 ||sum < 2*M_PI-1e-4) {
                printf("[%.16f, %.16f, %.16f]\n[(%.16f, %.16f),(%.16f, %.16f),(%.16f, %.16f)]\nWarning %.16f vs %.16f\n",
                    vrt->angles[0], vrt->angles[1], vrt->angles[2],
                    T[0].x, T[0].y, T[1].x, T[1].y, T[2].x, T[2].y,
                    sum, 2*M_PI);
            }*/
        }
        tid += gridDim.x * blockDim.x;
    }
}


/**
 * Compute all the areas of enabled grains
 * and stores them in the respective grain structure.
 *
 * @param dev_grains Pointer to grain device array
 * @param n_grains   Number of grains
 * @param DOMAIN_BOUND   Numerical domain bound
 *
 */
__global__ void compute_grain_areas(grain *dev_grains, int n_grains, const double DOMAIN_BOUND) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    grain *grn;
    while(tid < n_grains) {
        grn = &dev_grains[tid];
        if(grn->enabled) {
            grn->area = grain_compute_area(grn, DOMAIN_BOUND);
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Compute dA/dt of enabled grains and stores them in the respective grain structure.
 *
 * @param dev_grains Pointer to grain device array
 * @param n_grains   Number of grains
 * @param DOMAIN_BOUND   Numerical domain bound
 */
__global__ void compute_grain_dAdts(grain *dev_grains, int n_grains, const double DOMAIN_BOUND) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    grain *grn;
    while(tid < n_grains) {
        grn = &dev_grains[tid];
        if(grn->enabled) {
            grn->dAdt = grain_compute_dAdt(grn, DOMAIN_BOUND);
        }
        tid += gridDim.x * blockDim.x;
    }
}


/**
 * Kernel which computes the number of current grains alive in the system
 * per execution block.
 *
 * @param dev_grains Pointer to grains device array
 * @param n_grains   Number of grains
 * @param dev_buffer Pointer to buffer of partial sums
 */
__global__ void grains_per_block(grain* dev_grains, int n_grains, int *dev_buffer) {
    //extern __shared__ int tmp_ngrains[];
    __shared__ int tmp_ngrains[N_TRDS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    int tmp = 0;
    while(tid < n_grains) {
        // Check if the vertex is enabled before add
        if(dev_grains[tid].enabled) {
            tmp++;
        }
        tid += gridDim.x * blockDim.x;
    }
    tmp_ngrains[cacheIndex] = tmp;
    __syncthreads();
    int i = blockDim.x/2;
    while(i != 0) {
        if (cacheIndex < i)
            tmp_ngrains[cacheIndex] += tmp_ngrains[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    if(cacheIndex == 0)
        dev_buffer[blockIdx.x] = tmp_ngrains[0];
}

/**
 * Compute the current number of grains enabled in system.
 *
 * @param  dev_grains Pointer to device grains array
 * @param  n_grains   Number of grains
 * @param  n_blocks   Number of CUDA blocks
 * @param  n_threads  Number of CUDA thrads
 * @return            Current number of grains
 */
inline int current_n_grains(grain* dev_grains, const int n_grains,
                                const int n_blocks, const int n_threads,
                                int *buffer, int *dev_buffer)
{
    /* Buffers for counting grains */
    int loc_n_grains = 0;
    grains_per_block<<<n_blocks, n_threads>>>(dev_grains, n_grains, dev_buffer);
    CERR();
    HERR(cudaMemcpy(buffer, dev_buffer, sizeof(int) * n_blocks, cudaMemcpyDeviceToHost));
    CERR();
    for(int i = 0; i < n_blocks; i++) {
        loc_n_grains += buffer[i];
    }
    return loc_n_grains;
}

/**
 * Partial sum of nucleated area per block.
 *
 * @param dev_grains Pointer to device grains array
 * @param n_grains   Number of grains
 * @param dev_buffer Pointer to buffer of partial sums
 */
__global__ void nucl_area_per_block(grain* dev_grains, int n_grains, double *dev_buffer) {
    //extern __shared__ double tmp_nucl_area[];
    __shared__ double tmp_nucl_area[N_TRDS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    double tmp_area = 0.0;
    while(tid < n_grains) {
        // Check if the vertex is enabled before add
        if(dev_grains[tid].enabled) {
            if(dev_grains[tid].type == NUCLEATED) {
                tmp_area += dev_grains[tid].area;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
    tmp_nucl_area[cacheIndex] = tmp_area;
    __syncthreads();
    int i = blockDim.x/2;
    while(i != 0) {
        if (cacheIndex < i)
            tmp_nucl_area[cacheIndex] += tmp_nucl_area[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    if(cacheIndex == 0)
        dev_buffer[blockIdx.x] = tmp_nucl_area[0];
}


/**
 * Compute percentage of total area nucleated
 *
 * @param  dev_grains   Pointer to device grains array
 * @param  n_grains     Number of grains
 * @param  n_blocks     Number of CUDA blocks
 * @param  n_threads    Number of CUDA threads
 * @param  buffer       Pointer to host partial results
 * @param  dev_buffer   Pointer to device partial results
 * @param  DOMAIN_BOUND Numerical domain bound
 * @return              Percentage of total area nucleated
 */
inline double nucleated_fraction_area(grain* dev_grains, const int n_grains,
                                const int n_blocks, const int n_threads,
                                double *buffer, double *dev_buffer, const double DOMAIN_BOUND)
{
    /*  Buffers for counting nucleated grains */
    double nucl_area = 0.0;
    nucl_area_per_block<<<n_blocks, n_threads>>>(dev_grains, n_grains, dev_buffer);
    CERR();
    HERR(cudaMemcpy(buffer, dev_buffer, sizeof(double) * n_blocks, cudaMemcpyDeviceToHost));
    CERR();
    for(int i = 0; i < n_blocks; i++) {
        nucl_area += buffer[i];
    }
    //free(nucl_buffer);
    return 100*nucl_area / (DOMAIN_BOUND * DOMAIN_BOUND);
}



/**
 * Update vertices positions inside periodic domain.
 * It is assumed that gradE * dt is small enough.
 *
 * @param dev_vertices Pointer to vertex device array
 * @param n_vertices   Number of vertices
 * @param dt           Time-step parameter
 * @param DOMAIN_BOUND Numerical domain bound
 */
__global__ void update_vertices_positions(vertex* dev_vertices, int n_vertices, const double dt,
    const double DOMAIN_BOUND) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    vertex *vrt;
    while(tid < n_vertices) {
        vrt = &dev_vertices[tid];
        if(vrt->enabled) {
            vrt->pos.x += vrt->vel.x * dt;
            vrt->pos.y += vrt->vel.y * dt;
            vrt->pos = vector2_adjust(vrt->pos, DOMAIN_BOUND);
        }
        tid += gridDim.x * blockDim.x;
    }
}

#endif // CALCULUS_H