#ifndef TOPOLOGICAL_H
#define TOPOLOGICAL_H

#include "geometry.h"
#include "calculus.h"

/**
 * Topological Module.
 * Manages grain removal and flipping procedure.
 * Also contains kernels for counting events.
 */

typedef enum {FLIP, REMOVAL} event_t;

/**
 * Count two three sided grains configurations per boundary per block

 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param dev_buffer     Device buffer to count results per block
 */
__global__ void two_three_sided_grains_per_block(boundary* dev_boundaries, int n_boundaries, int *dev_buffer) {
    __shared__ int tmp_data[N_TRDS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    int tmp = 0;
    while(tid < n_boundaries) {
        // Check if the vertex is enabled before add
        boundary *bnd = &dev_boundaries[tid];
        grain* gic[2];
        if(bnd->enabled) {
            grains_intersect(bnd->ini, bnd->end, gic);
            if(gic[0]->vlen == 3 && gic[1]->vlen == 3) {
                tmp++;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
    tmp_data[cacheIndex] = tmp;
    __syncthreads();
    int i = blockDim.x/2;
    while(i != 0) {
        if (cacheIndex < i)
            tmp_data[cacheIndex] += tmp_data[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    if(cacheIndex == 0)
        dev_buffer[blockIdx.x] = tmp_data[0];
}

/**
 * Count configurations of two three sided grains
 *
 * @param  dev_boundaries Pointer to boundary device array
 * @param  n_boundaries   Number of boundaries
 * @param  dev_buffer     Device buffer to count results per block
 * @param  n_blocks       Number of blocks
 * @param  n_threads      Number of threads
 * @return                Number of configurations of two neighbor grains with three sides
 */
inline int two_three_sided_grains(boundary *dev_boundaries, int n_boundaries,
                                    const int n_blocks, const int n_threads,
                                    int *buffer, int *dev_buffer) {
    int result = 0;
    two_three_sided_grains_per_block<<<n_blocks, n_threads>>>(dev_boundaries, n_boundaries, dev_buffer);
    CERR();
    HERR(cudaMemcpy(buffer, dev_buffer, sizeof(int) * n_blocks, cudaMemcpyDeviceToHost));
    for(int i = 0; i < n_blocks; i++) {
        result += buffer[i];
    }
    return result;
}

/**
 * Check conflictive boundaries. Checks if neighbor boundaries will flip,
 * labeling as conflictive the one with lowest t_ext or lower pointer if
 * both t_ext are equal
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 */
__global__ void check_conflictive_boundaries(boundary* dev_boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_boundaries) {
        boundary *bnd = &dev_boundaries[tid];
        if(bnd->enabled && !bnd->candidate) {
            //boundary b1, b2, b3, b4;
            boundary *bc_ini = NULL, *bc_end = NULL;
            for(int i = 0; i < 3; i++) {
                if(bnd->ini->boundaries[i] != bnd && bnd->ini->boundaries[i]->candidate) {
                    bc_ini = bnd->ini->boundaries[i];
                    break;
                }
            }
            for(int i = 0; i < 3; i++) {
                if(bnd->end->boundaries[i] != bnd && bnd->end->boundaries[i]->candidate) {
                    bc_end = bnd->end->boundaries[i];
                    break;
                }
            }
            if(bc_ini != NULL && bc_end != NULL) {
                if(bc_ini->t_ext == bc_end->t_ext) {
                    if(bc_ini->id < bc_end->id) {
                        bnd->near_conflictive_bnd = bc_ini->id;
                    } else {
                        bnd->near_conflictive_bnd = bc_end->id;
                    }
                } else {
                    if(bc_ini->t_ext < bc_end->t_ext) {
                        bnd->near_conflictive_bnd = bc_ini->id;
                    } else {
                        bnd->near_conflictive_bnd = bc_end->id;
                    }
                }
                printf("In %s: (%d) Boundary %d marked as conflictive\n", __func__, bnd->id, bnd->near_conflictive_bnd);
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Unlabel the conflictive boundaries in order to avoid flipping them
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 */
__global__ void unlabel_conflictive_boundaries(boundary* dev_boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_boundaries) {
        boundary *bnd = &dev_boundaries[tid];
        if(bnd->enabled && bnd->candidate) {
            for(int i = 0; i < 3; i++) {
                if(bnd != bnd->ini->boundaries[i] && bnd->ini->boundaries[i]->near_conflictive_bnd != -1)
                {
                    if(bnd->ini->boundaries[i]->near_conflictive_bnd != bnd->id) {
                        bnd->candidate = false;
                        break;
                    }
                }
            }
            for(int i = 0; i < 3; i++) {
                if(bnd != bnd->end->boundaries[i] && bnd->end->boundaries[i]->near_conflictive_bnd != -1)
                {
                    if(bnd->end->boundaries[i]->near_conflictive_bnd != bnd->id) {
                        bnd->candidate = false;
                        break;
                    }
                }
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Apply a flipping to a boundary.
 *
 *                                     C   G2  F
 *                                      \     /
 *      C    G2    F                     \   /
 *       \        /                        B
 *   G3   A------B   G1    =flip=>   G3    |   G1
 *       /        \                        A
 *      D    G4    E                     /   \
 *                                      /     \
 *                                     D   G4  E
 *
 * @param bnd          Pointer to boundary struct
 * @param eps          Epsilon for grain boundary energy function
 * @param DOMAIN_BOUND Numerical domain bound
 */
__device__ void flip(boundary *bnd, double eps, const double DOMAIN_BOUND) {
    vertex *A = bnd->ini;
    vertex *B = bnd->end;

    // Get grains in common and not in common
    grain *gic[2], *gnic[2];
    grains_intersect(A, B, gic);
    grains_symdiff(A, B, gnic);

    // Get the boundary A-C
    boundary *AC, *BE;
    int i = 0, j = 0;
    for(i = 0; i < 3; i++) {
        if(A->boundaries[i] == bnd) {
            i = (i + 1) % 3;
            AC = A->boundaries[i];
            break;
        }
    }
    // Get the boundary B-E
    for(j = 0; j < 3; j++) {
        if(B->boundaries[j] == bnd) {
            j = (j + 1) % 3;
            BE = B->boundaries[j];
            break;
        }
    }

    // Swap the boundary pointers on front ends
    A->boundaries[i] = BE;
    B->boundaries[j] = AC;
    // Connect the swapping boundary to the other ends.
    if(AC->ini == A) {
        AC->ini = B;
    } else {
        AC->end = B;
    }
    if(BE->ini == B) {
        BE->ini = A;
    } else {
        BE->end = A;
    }

    // Remove A from grain G2
    int g1 = 0, g2 = 0, g3 = 0, g4 = 0;
    for(g2 = 0; g2 < 2; g2++) {
        if(grain_contains_consecutive_vertices(gic[g2], A, B) == 1) {

            A->mod[0].fix = REMOVE;
            A->mod[0].vrt_id = A->id;
            A->mod[0].grn_id = gic[g2]->id;
            gic[g2]->fix = true;
            break;
        }
    }

    // Remove B from grain G4
    for(g4 = 0; g4 < 2; g4++) {
        if(grain_contains_consecutive_vertices(gic[g4], B, A) == 1) {

            B->mod[0].fix = REMOVE;
            B->mod[0].vrt_id = B->id;
            B->mod[0].grn_id = gic[g4]->id;
            gic[g4]->fix = true;
            break;
        }
    }

    // Add B between A and C in G3
    for(g3 = 0; g3 < 2; g3++) {
        int A_pos = grain_contains_vertex(gnic[g3], A);
        if(A_pos >= 0) {

            A->mod[1].fix = ADD;
            A->mod[1].vrt_id = B->id;
            A->mod[1].vrt_prev_id = A->id;
            A->mod[1].grn_id = gnic[g3]->id;
            gnic[g3]->fix = true;
            break;
        }
    }

    // Add A between B and E in G1
    for(g1 = 0; g1 < 2; g1++) {
        int B_pos = grain_contains_vertex(gnic[g1], B);
        if(B_pos >= 0) {

            B->mod[1].fix = ADD;
            B->mod[1].vrt_id = A->id;
            B->mod[1].vrt_prev_id = B->id;
            B->mod[1].grn_id = gnic[g1]->id;
            gnic[g1]->fix = true;
            break;
        }
    }
    // Fix grain list for A and B
    for(int k = 0; k < 3; k++) {
        if(A->grains[k] == gic[g2]) {
            A->grains[k] = gnic[g1];
        }
        if(B->grains[k] == gic[g4]) {
            B->grains[k] = gnic[g3];
        }
    }
    // Adjust vertices positions
    vector2 center = vector2_mean(A->pos, B->pos, DOMAIN_BOUND);
    A->pos = center;
    B->pos = center;
    // Make order of boundaries counterclock-wise
    vertex_invert_boundary_order(A);
    vertex_invert_boundary_order(B);
    // Compute energy
    bnd->energy = boundary_compute_energy(bnd, eps);
    // Set velocity to zero
    vector2 nullv; nullv.x=0; nullv.y=0;
    A->vel = nullv;
    B->vel = nullv;
}

/**
 * Compute extinction time "t_ext" of a boundary.
 * As this is the first step of the flipping procedure,
 * All the boundaries are marked as non candidate.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param dt             Time-step
 * @param DOMAIN_BOUND   Numerical domain bound
 */
__global__ void compute_t_ext(boundary* dev_boundaries, int n_boundaries, double dt, const double DOMAIN_BOUND) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    boundary *bnd;
    double norm, xpos[2], ypos[2];
    while(tid < n_boundaries) {
        bnd = &dev_boundaries[tid];
        if(bnd->enabled) {
            bnd->candidate = false;
            // Get vertices positions
            vertex *ini = bnd->ini;
            vertex *end = bnd->end;
            xpos[0] = ini->pos.x; xpos[1] = end->pos.x;
            ypos[0] = ini->pos.y; ypos[1] = end->pos.y;
            adjust_origin_for_points(xpos, ypos, 2, DOMAIN_BOUND);
            vector2 diff, T, P;
            diff.x = xpos[1] - xpos[0];
            diff.y = ypos[1] - ypos[0];
            norm = vector2_mag(diff);
            T = vector2_unitary(diff);
            P.x = end->vel.x - ini->vel.x;
            P.y = end->vel.y - ini->vel.y;
            // Store extinction time in boundary
            double aux = vector2_dot(T, P);
            if(aux == 0) {
                bnd->t_ext = 100;
            } else {
                bnd->t_ext = -norm / aux;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Label boundaries with extinction time between 0 and dt.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param dt             Time-step
 */
__global__ void label_boundaries(boundary* dev_boundaries, int n_boundaries, double dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    boundary *bnd;
    while(tid < n_boundaries) {
        bnd = &dev_boundaries[tid];
        bnd->candidate = boundary_is_candidate(bnd, dt);
        // Here reset the conflictive boundary flag
        bnd->near_conflictive_bnd  = -1;
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Vertices vote for one of the there possible boundaries which
 * has the minimum extinction time.
 *
 * @param dev_vertices Pointer to vertex device array
 * @param n_vertices   Number of vertices
 */
__global__ void vertices_vote(vertex* dev_vertices, int n_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_vertices) {
        vertex *vrt = &dev_vertices[tid];
        if(vrt->enabled) {
            double min_t_ext = 99999999;
            int argmin = 0;
            for(int i = 0; i < 3; i++) {
                boundary *bnd = vrt->boundaries[i];
                if(bnd->t_ext <= min_t_ext && bnd->candidate) {
                    min_t_ext = bnd->t_ext;
                    argmin = i;
                }
            }
            vrt->voted = vrt->boundaries[argmin];
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Reset the boundary voted by a vertex.
 *
 * @param dev_vertices Pointer to vertex device array
 * @param n_vertices   Number of vertices
 */
__global__ void reset_vertices_votes(vertex* dev_vertices, int n_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_vertices) {
        vertex *vrt = &dev_vertices[tid];
        if(vrt->enabled) {
            vrt->voted = NULL;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Each boundary counts its votes. The range of values is between 0 and 2.
 * 2 votes implies that the boundary has to flip this timestep.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 */
__global__ void count_boundaries_votes(boundary *dev_boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_boundaries) {
        boundary *bnd = &dev_boundaries[tid];
        if(bnd->enabled && bnd->candidate) {
            if(bnd->ini->voted == bnd)
                bnd->n_votes++;
            if(bnd->end->voted == bnd)
                bnd->n_votes++;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Reset the number of votes for each boundary to zero.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 */
__global__ void reset_boundaries_votes(boundary *dev_boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_boundaries) {
        boundary *bnd = &dev_boundaries[tid];
        if(bnd->enabled) { bnd->n_votes = 0; }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Unlabel all boundaries near a boundary which the smaller extinction time.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 */
__global__ void unlabel(boundary *dev_boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_boundaries) {
        boundary *bnd = &dev_boundaries[tid];
        if(bnd->enabled && bnd->candidate && bnd->n_votes == 2) {
            for(int i = 0; i < 3; i++) {
                if(bnd != bnd->ini->boundaries[i]) {
                    bnd->ini->boundaries[i]->candidate = false;
                }
                if(bnd != bnd->end->boundaries[i]) {
                    bnd->end->boundaries[i]->candidate = false;
                }
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Write the candidate state of each boundary to a buffer.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param dev_buffer     Buffer where last candidate state are stored.
 */
__global__ void write_candidate_results(boundary *dev_boundaries, int n_boundaries, bool *candidate_buffer) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_boundaries) {
        boundary *bnd = &dev_boundaries[tid];
        if(bnd->enabled) {
            candidate_buffer[tid] = bnd->candidate;
        } else {
            candidate_buffer[tid] = false;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * [flips_per_block description]
 * @param  dev_boundaries   Pointer to boundary device array
 * @param  n_boundaries     Number of boundaries
 * @param  candidate_buffer Buffer where last candidate state are stored
 * @param  dev_buffer       Device buffer of candidate state for each boundary
 */
__global__ void flips_per_block(boundary *dev_boundaries, int n_boundaries,
                                bool *candidate_buffer, bool *dev_buffer) {
    //extern __shared__ bool tmp_compare[];
    __shared__ bool tmp_compare[N_TRDS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    bool tmp = true;
    while(tid < n_boundaries) {
        boundary *bnd = &dev_boundaries[tid];
        if(bnd->enabled) {
            tmp *= (bnd->candidate == candidate_buffer[tid]);
        }
        tid += gridDim.x * blockDim.x;
    }
    tmp_compare[cacheIndex] = tmp;
    __syncthreads();
    int i = blockDim.x/2;
    while(i != 0) {
        if (cacheIndex < i)
            tmp_compare[cacheIndex] += tmp_compare[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    if(cacheIndex == 0)
        dev_buffer[blockIdx.x] = tmp_compare[0];
}

/**
 * Compare the current candidate state of each valid boundary with
 * the candidate state stored in candidate_buffer.
 *
 * @param  dev_boundaries   Pointer to boundary device array
 * @param  n_boundaries     Number of boundaries
 * @param  buffer           Host buffer of candidate state for each boundary
 * @param  dev_buffer       Device buffer of candidate state for each boundary
 * @param  candidate_buffer Buffer where last candidate state are stored
 * @param  n_blocks         Number of CUDA blocks
 * @param  n_threads        Number of CUDA threads
 * @return                  True if the configuration is equal, False otherwise
 */
inline bool compare_candidates(boundary *dev_boundaries, int n_boundaries,
                            bool *buffer, bool *dev_buffer, bool *candidate_buffer,
                            int n_blocks, int n_threads) {
    flips_per_block<<<n_blocks, n_threads>>>(dev_boundaries, n_boundaries, candidate_buffer, dev_buffer);
    CERR();
    HERR(cudaMemcpy(buffer, dev_buffer, sizeof(bool) * n_blocks, cudaMemcpyDeviceToHost));
    bool tmp = true;
    for(int i = 0; i < n_blocks; i++) {
        tmp *= buffer[i];
    }
    return tmp;
}

/**
 * Boundaries must be unlabeled iteratively until no more unlabel can be performed.
 *
 * @param dev_vertices   Pointer to vertex device array
 * @param n_vertices     Number of vertices
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param n_blocks       Number of CUDA blocks
 * @param n_threads      Number of CUDA threads
 */
inline void polling_system(vertex *dev_vertices, int n_vertices,
                                   boundary *dev_boundaries, int n_boundaries,
                                   bool *buffer, bool *dev_buffer, bool *candidate_buffer,
                                   int n_blocks, int n_threads) {
    // Convergence implies two configurations are the same over time.
    bool convergence = false;
    while(!convergence) {
        // Reset the boundary total number of votes
        reset_boundaries_votes<<<n_blocks, n_threads>>>(dev_boundaries, n_boundaries);
        CERR();
        // Reset the vertices votes
        reset_vertices_votes<<<n_blocks, n_threads>>>(dev_vertices, n_vertices);
        CERR();
        // Make the vertices vote over the remaining candidate boundaries
        vertices_vote<<<n_blocks, n_threads>>>(dev_vertices, n_vertices);
        CERR();
        // Count each candidate boundary votes
        count_boundaries_votes<<<n_blocks, n_threads>>>(dev_boundaries, n_boundaries);
        CERR();
        // Unlabel boundaries from new valid flips
        unlabel<<<n_blocks, n_threads>>>(dev_boundaries, n_boundaries);
        CERR();
        // Check if the current candidates are the same as before, stored in candidate_buffer
        convergence = compare_candidates(dev_boundaries, n_boundaries, buffer,
                                         dev_buffer, candidate_buffer, n_blocks, n_threads);
        // write the results to buffer
        write_candidate_results<<<n_blocks, n_threads>>>(dev_boundaries, n_boundaries, candidate_buffer);
        CERR();
    }
}

/**
 * Unlabel boundaries after grain removal. During grain removal new 3-sided grains
 * can be created. Related boundaries still can flip and produce 2-sided grains.
 * We can unmark the boundary and detect it at next time-step.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 */
__global__ void unlabel_boundaries_st2(boundary* dev_boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    boundary *bnd;
    while(tid < n_boundaries) {
        bnd = &dev_boundaries[tid];
        if(bnd->enabled && bnd->candidate) {
            grain *gic[2];
            grains_intersect(bnd->ini, bnd->end, gic);
            if(gic[0]->vlen == 3 || gic[1]->vlen == 3) {
                bnd->candidate = false;
                printf("In %s: 3 sided grain about to flip before flipping (%d and %d)\n", gic[0]->id, gic[1]->id);
            }
            if(gic[0]->vlen == 3 && gic[1]->vlen == 3) {
                printf("In %s: Error: 3 Sided grains will flip  (%d and %d)\n", gic[0]->id, gic[1]->id);
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Clear the candidate state of all grains to false.
 *
 * @param dev_grains Pointer to grain device array
 * @param n_grains   Number of grains
 */
__global__ void clear_grains_state(grain *dev_grains, int n_grains) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_grains) {
        grain *grn = &dev_grains[tid];
        if(grn->enabled) {
            grn->candidate = false;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Label three sided grains which have a boundary candidate to flip.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param dev_grains     Pointer to grain device array
 * @param n_grains       Number of grains
 * @param dt             Delta t
 */
__global__ void label_3sided_grains(boundary *dev_boundaries, int n_boundaries,
                                  grain *dev_grains, int n_grains, double dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_grains) {
        grain *grn = &dev_grains[tid];
        if(grn->enabled && grn->vlen == 3) {
            int ans = 0;
            for(int i = 0; i < 3; i++) {
                vertex *vrt = grn->vertices[i];
                // Check the boundaries in current grain
                for(int k = 0; k < 3; k++) {
                    boundary *bnd = vrt->boundaries[k];
                    int ini_pos = grain_contains_vertex(grn, bnd->ini);
                    int end_pos = grain_contains_vertex(grn, bnd->end);
                    // If the boundary belongs to current grain and
                    // the boundary is labeled for flipping and also is enabled
                    if(ini_pos >= 0 && end_pos >= 0 && bnd->candidate && bnd->enabled) {
                        grn->candidate = true;
                        break;
                    }
                    // Case when a grain will flip all its boundaries but
                    // they were unlabeled.
                    if(ini_pos >= 0 && end_pos >= 0 &&
                       boundary_is_candidate(bnd, dt) && bnd->enabled) {
                        ans++;
                    }
                }
            }
            ans /= 2;
            if(ans == 3)
                grn->candidate = true;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Remove grain leaving a single vertex.
 *
 *        D
 *        |
 *        |                           D
 *        A                           |
 *  g2   / \   g1                g2   |  g1
 *      / g \       =remove=>         A
 *     B-----C                      /   \
 *    /       \                    / g3  \
 *   /   g3    \                  E       F
 *  E           F
 *
 * @param grn          Pointer to grain struct
 * @param DOMAIN_BOUND Numerical domain bound
 */
__device__ inline void remove_grain(grain *grn, const double DOMAIN_BOUND) {
    // Recover individual vertices of grain
    // The survivor grain is A
    vertex *A, *B, *C;
    A = grn->vertices[0];
    B = grn->vertices[1];
    C = grn->vertices[2];
    // Reposition A to the incenter
    double xpos[3] = {A->pos.x, B->pos.x, C->pos.x};
    double ypos[3] = {A->pos.y, B->pos.y, C->pos.y};
    double nAB = compute_arclength(A->pos, B->pos, DOMAIN_BOUND);
    double nBC = compute_arclength(B->pos, C->pos, DOMAIN_BOUND);
    double nCA = compute_arclength(C->pos, A->pos, DOMAIN_BOUND);
    double denom = nAB + nBC + nCA;
    adjust_origin_for_points(xpos, ypos, 3, DOMAIN_BOUND);
    A->pos.x += (nBC*xpos[0] + nCA*xpos[1] + nAB*xpos[2]) / denom;
    A->pos.y += (nBC*ypos[0] + nCA*ypos[1] + nAB*ypos[2]) / denom;
    A->pos = vector2_adjust(A->pos, DOMAIN_BOUND);
    // Recover the grains
    grain *gic_BC[2], *gic_AB[2], *gic_AC[2], *G1, *G2, *G3;
    grains_intersect(B, C, gic_BC);
    if(gic_BC[0] == grn) {
        G3 = gic_BC[1];
    } else {
        G3 = gic_BC[0];
    }
    grains_intersect(A, B, gic_AB);
    if(gic_AB[0] == grn) {
        G2 = gic_AB[1];
    } else {
        G2 = gic_AB[0];
    }
    grains_intersect(A, C, gic_AC);
    if(gic_AC[0] == grn) {
        G1 = gic_AC[1];
    } else {
        G1 = gic_AC[0];
    }
    // Set velocity to zero
    A->vel.x = 0.0;
    A->vel.y = 0.0;
    // Find boundary B-C and disable it
    for(int i = 0; i < 3; i++) {
        if((C->boundaries[i]->ini == B && C->boundaries[i]->end == C) ||
           (C->boundaries[i]->end == B && C->boundaries[i]->ini == C)) {
            C->boundaries[i]->enabled = false;
            break;
        }
    }
    // Find boundary B-E
    boundary *BE, *CF;
    for(int i = 0; i < 3; i++) {
        if(!(B->boundaries[i]->ini == A || B->boundaries[i]->end == A) &&
           !(B->boundaries[i]->ini == C || B->boundaries[i]->end == C)) {
            BE = B->boundaries[i];
            break;
        }
    }
    // Find boundary C-F
    for(int i = 0; i < 3; i++) {
        if(!(C->boundaries[i]->ini == A || C->boundaries[i]->end == A) &&
           !(C->boundaries[i]->ini == B || C->boundaries[i]->end == B)) {
            CF = C->boundaries[i];
            break;
        }
    }
    // Change B-E to A-E
    if(BE->ini == B) {
        BE->ini = A;
    } else {
        BE->end = A;
    }
    // CHange C-F to A-F
    if(CF->ini == C) {
        CF->ini = A;
    } else {
        CF->end = A;
    }

    // Disable old boundaries of A and rebuild list
    for(int i = 0; i < 3; i++) {
        if(A->boundaries[i]->ini == B || A->boundaries[i]->end == B) {
            A->boundaries[i]->enabled = false;
            A->boundaries[i] = BE;
            break;
        }
    }
    for(int i = 0; i < 3; i++) {
        if(A->boundaries[i]->ini == C || A->boundaries[i]->end == C) {
            A->boundaries[i]->enabled = false;
            A->boundaries[i] = CF;
            break;
        }
    }
    // Disable vertices b and c
    B->enabled = false;
    C->enabled = false;
    grn->enabled = false;

    // The modifications are stored in vertex A
    // Delete B and C from grains
    int B_pos = grain_contains_vertex(G2, B);
    B->mod[0].grn_id = G2->id;
    B->mod[0].fix = REMOVE;
    B->mod[0].vrt_id = B->id;
    G2->fix = true;

    int C_pos = grain_contains_vertex(G1, C);
    C->mod[0].grn_id = G1->id;
    C->mod[0].fix = REMOVE;
    C->mod[0].vrt_id = C->id;
    G1->fix = true;

    B_pos = grain_contains_vertex(G3, B);
    B->mod[1].grn_id = G3->id;
    B->mod[1].fix = REMOVE;
    B->mod[1].vrt_id = B->id;
    G3->fix = true;

    C_pos = grain_contains_vertex(G3, C);
    C->mod[1].fix = REPLACE;
    C->mod[1].vrt_id = A->id;
    C->mod[1].vrt_prev_id = C->id;
    C->mod[1].grn_id = G3->id;

    G1->fix = true;
    G2->fix = true;
    G3->fix = true;

    // Fix the list of grains for A
    for(int i = 0; i < 3; i++) {
        if(A->grains[i] == grn) {
            A->grains[i] = G3;
        }
    }
}

/**
 * Remove configuration of two grains of three sides.
 * This configuration is removed only when the boundary A-B
 * collapses. All the data is read from this boundary.
 *
 *            A
 *          / | \
 *         /  |  \
 *   F----D   |   C----E   =remove=>   F---------E
 *         \  |  /
 *          \ | /
 *            B
 *
 * @param bnd  Pointer to boundary collapsing in configuration
 * @param gnic Grains to be fixed after removal
 */
__device__ inline void remove_two_grains(boundary *bnd, grain **gnic) {
    vertex *A, *B, *C, *D, *F;
    boundary *AC, *BD, *CE, *DF;
    A = bnd->ini;
    B = bnd->end;
    int i, j;
    // Find boundaries AC and BD
    for(i = 0; i < 3; i++) {
        if(A->boundaries[i] == bnd) {
            i = (i + 1) % 3;
            AC = A->boundaries[i];
            break;
        }
    }
    for(j = 0; j < 3; j++) {
        if(B->boundaries[j] == bnd) {
            j = (j + 1) % 3;
            BD = B->boundaries[j];
            break;
        }
    }
    // Find C and D
    if(AC->ini == A) {
        C = AC->end;
    } else {
        C = AC->ini;
    }
    if(BD->ini == B) {
        D = BD->end;
    } else {
        D = BD->ini;
    }
    // Find boundary CE and DF
    int k, l;
    for(k = 0; k < 3; k++) {
        if(C->boundaries[k] == AC) {
            k = (k + 2) % 3;
            CE = C->boundaries[k];
            break;
        }
    }
    for(l = 0; l < 3; l++) {
        if(D->boundaries[l] == BD) {
            l = (l + 2) % 3;
            DF = D->boundaries[l];
            break;
        }
    }
    if(DF->ini == D) {
        F = DF->end;
    } else {
        F = DF->ini;
    }
    // Change CE to FE by changing the vertex it is pointing
    if(CE->ini == C) {
        CE->ini = F;
    } else {
        CE->end = F;
    }
    // F now has to take the boundary CE in its list
    for(int p = 0; p < 3; p++) {
        if(F->boundaries[p] == DF) {
            F->boundaries[p] = CE;
            break;
        }
    }

    grain *G3, *G4;
    if(grain_contains_vertex(gnic[0], A) >= 0) {
        G3 = gnic[0];
        G4 = gnic[1];
    } else {
        G3 = gnic[1];
        G4 = gnic[0];
    }

    // Order grains to be fixed
    A->mod[0].fix = REMOVE;
    A->mod[0].grn_id = G3->id;
    A->mod[0].vrt_id = A->id;
    G3->fix = true;

    A->mod[1].fix = REMOVE;
    A->mod[1].grn_id = G3->id;
    A->mod[1].vrt_id = C->id;

    A->mod[2].fix = REMOVE;
    A->mod[2].grn_id = G3->id;
    A->mod[2].vrt_id = D->id;

    B->mod[0].fix = REMOVE;
    B->mod[0].grn_id = G4->id;
    B->mod[0].vrt_id = B->id;
    G4->fix = true;

    B->mod[1].fix = REMOVE;
    B->mod[1].grn_id = G4->id;
    B->mod[1].vrt_id = C->id;

    B->mod[2].fix = REMOVE;
    B->mod[2].grn_id = G4->id;
    B->mod[2].vrt_id = D->id;

    // Disable boundaries related to both grains
    for(int i = 0; i < 3; i++) {
        A->boundaries[i]->enabled = false;
        B->boundaries[i]->enabled = false;
    }
    DF->enabled = false;
    bnd->enabled = false;
    // Disable vertices
    A->enabled = false;
    D->enabled = false;
    C->enabled = false;
    B->enabled = false;
}

/**
 * Remove the configuration of two grains of 3 sides
 * by retrieving information from the shared boundary.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param  debug_lvl     Level of debug
 */
__global__ void  remove_two_grains_configurations(boundary* dev_boundaries, int n_boundaries, int debug_lvl) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    boundary *bnd;
    while(tid < n_boundaries) {
        bnd = &dev_boundaries[tid];
        if(bnd->enabled){// && bnd->candidate) {
            grain *gic[2], *gnic[2];
            grains_intersect(bnd->ini, bnd->end, gic);
            grains_symdiff(bnd->ini, bnd->end, gnic);
            if(gic[0]->vlen == 3 && gic[1]->vlen == 3) {
                remove_two_grains(bnd, gnic);
                gic[0]->enabled = false;
                gic[1]->enabled = false;
                if(debug_lvl >= 1) {
                    printf("[W] Grains %d and %d removed\n", gic[0]->id, gic[1]->id);
                }
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Remove all grains that do not introduce inconsistencies.
 *
 * @param dev_grains   Pointer to grain device array
 * @param n_grains     Number of grains
 * @param DOMAIN_BOUND Numerical domain bound
 */
__global__ void remove_grains(grain* dev_grains, int n_grains, const double DOMAIN_BOUND) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    grain *grn;
    while(tid < n_grains) {
        grn = &dev_grains[tid];
        if(grn->enabled && grn->candidate) {
            remove_grain(grn, DOMAIN_BOUND);
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Apply flippings procedure. Also compute boundary energy.
 * This procedure does not update grain lists. This must be done after
 * every flip has been performed.
 *
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param eps            Epsilon of grain boundary energy function
 * @param DOMAIN_BOUND   Numerical domain bound
 */
__global__ void apply_flippings(boundary *dev_boundaries, int n_boundaries, double eps, const double DOMAIN_BOUND) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    boundary *bnd;
    while(tid < n_boundaries) {
        bnd = &dev_boundaries[tid];
        if(bnd->enabled && bnd->candidate) {
            flip(bnd, eps, DOMAIN_BOUND);
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Perform ADD operations over grains.
 *
 * @param dev_grains Pointer to grain device array
 * @param n_grains   Number of grains
 * @param debug_lvl  Level of debug
 */
__global__ void perform_ADD_ops(grain* dev_grains, int n_grains, vertex* dev_vertices, int debug_lvl) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_grains) {
        grain *grn = &dev_grains[tid];
        if(grn->enabled && grn->fix) {
            for(int i = 0; i < grn->vlen; i++) {
                vertex *vrt = grn->vertices[i];
                for(int j = 0; j < 4; j++) {
                    if(vrt->mod[j].fix == ADD && vrt->mod[j].grn_id == grn->id) {
                        //printf("Fix %d %d\n",vrt->mod[j].grn_id,grn->id);
                        if(debug_lvl >= 1) {
                            printf("In %s (%d) ADD %d to grain %d\n", __func__, vrt->id,
                                dev_vertices[vrt->mod[j].vrt_id].id,
                                dev_grains[vrt->mod[j].grn_id].id);
                        }
                        grain_add_vertex(grn, &dev_vertices[vrt->mod[j].vrt_id], &dev_vertices[vrt->mod[j].vrt_prev_id]);
                    }
                    /*
                    if(vrt->mod[j].fix != NONE && vrt->mod[j].grn == grn) {
                        printf("(%d) fix is %d over grain %d\n",vrt->id, vrt->mod[j].fix, grn->id);
                        printf("(%d) grn addr is %d with id %d\n",vrt->id, vrt->mod[j].grn, vrt->mod[j].grn->id);
                        //printf("(%d) My friend, grn is null :(\n", vrt->id);
                    } //else {

                        //printf("(%d) fix of %d\n", vrt->id, vrt->mod[j].grn->id);
                    //}
                    if(vrt->mod[j].fix == ADD && vrt->mod[j].grn == grn) {
                        if(debug_lvl >= 1) {
                            printf("(%d) ADD %d to grain %d\n", vrt->id, vrt->mod[j].vrt->id, vrt->mod[j].grn->id);
                        }
                        grain_add_vertex(grn, vrt->mod[j].vrt, vrt->mod[j].vrt_prev);
                    }*/
                }
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Perform REMOVE operations over grains.
 *
 * @param dev_grains Pointer to grain device array
 * @param n_grains   Number of grains
 * @param debug_lvl  Level of debug
 */
__global__ void perform_REMOVE_ops(grain* dev_grains, int n_grains, vertex* dev_vertices, int debug_lvl) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_grains) {
        grain *grn = &dev_grains[tid];
        if(grn->enabled && grn->fix) {
            mod_t buff[MAX_VRT_PER_GRN * 4];
            for(int i = 0; i < MAX_VRT_PER_GRN * 4; i++) { buff[i].fix = NONE; }
            int vlen = grn->vlen;
            for(int i = 0; i < vlen; i++) {
                vertex *vrt = grn->vertices[i];
                for(int j = 0; j < 4; j++) {
                    buff[i*4 + j].fix = vrt->mod[j].fix;
                    buff[i*4 + j].grn_id = vrt->mod[j].grn_id;
                    buff[i*4 + j].vrt_id = vrt->mod[j].vrt_id;
                    buff[i*4 + j].vrt_prev_id = vrt->mod[j].vrt_prev_id;
                }
            }
            for(int i = 0; i < MAX_VRT_PER_GRN * 4; i++) {
                if(buff[i].fix == REMOVE && buff[i].grn_id == grn->id) {
                    if(debug_lvl >= 1) {
                        printf("In %s REMOVE %d from grain %d\n",__func__,
                         dev_vertices[buff[i].vrt_id].id, dev_grains[buff[i].grn_id].id);
                    }
                    grain_remove_vertex(grn, &dev_vertices[buff[i].vrt_id]);
                }
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Peform REPLACE operations over grains.
 *
 * @param dev_grains Pointer to grain device array
 * @param n_grains   Number of grains
 * @param debug_lvl  Level of debug
 */
__global__ void perform_REPLACE_ops(grain* dev_grains, int n_grains, vertex* dev_vertices, int debug_lvl) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_grains) {
        grain *grn = &dev_grains[tid];
        if(grn->enabled && grn->fix) {
            for(int i = 0; i < grn->vlen; i++) {
                vertex *vrt = grn->vertices[i];
                for(int j = 0; j < 4; j++) {
                    if(vrt->mod[j].fix == REPLACE && vrt->mod[j].grn_id == grn->id) {
                        if(debug_lvl >= 1) {
                            printf("In %s (%d) REPLACE %d to %d in grain %d\n", __func__, vrt->id,
                                dev_vertices[vrt->mod[j].vrt_prev_id].id,
                                dev_vertices[vrt->mod[j].vrt_id].id, dev_grains[vrt->mod[j].grn_id].id);
                        }
                        grain_replace_vertex(grn, &dev_vertices[vrt->mod[j].vrt_prev_id], &dev_vertices[vrt->mod[j].vrt_id]);
                    }
                }
            }

        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Clear the fix state of grains and reset all related commands
 * to be performed to NONE.
 *
 * @param dev_grains Pointer to grain device array
 * @param n_grains   Number of grains
 */
__global__ void clear_grains_fix_state(grain* dev_grains, int n_grains) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_grains) {
        grain *grn = &dev_grains[tid];
        if(grn->enabled) {
            grn->fix = false;
            for(int i = 0; i < grn->vlen; i++) {
                vertex *vrt = grn->vertices[i];
                reset_mod_list(vrt);
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Debug function which prints all the stored changes to be performed over grains.
 *
 * @param dev_grains   Pointer to grain device array
 * @param n_grains     Number of grains
 * @param dev_vertices Pointer to vertex device array
 */
__global__ void check_movements(grain* dev_grains, int n_grains, vertex* dev_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < n_grains) {
        grain *grn = &dev_grains[tid];
        if(grn->enabled && grn->fix) {
            for(int i = 0; i < grn->vlen; i++) {
                vertex *vrt = grn->vertices[i];
                for(int j = 0; j < 4; j++) {
                    if(vrt->mod[j].fix == ADD) {
                        printf("In %s: (%d) ADD %d to grain %d\n", __func__, vrt->id,
                            dev_vertices[vrt->mod[j].vrt_id].id, dev_grains[vrt->mod[j].grn_id].id);
                    }
                    if(vrt->mod[j].fix == REPLACE) {
                        printf("In %s: (%d) REPLACE %d to %d in grain %d\n", __func__, vrt->id,
                            dev_vertices[vrt->mod[j].vrt_prev_id].id, dev_vertices[vrt->mod[j].vrt_id].id,
                            dev_grains[vrt->mod[j].grn_id].id);
                    }
                    if(vrt->mod[j].fix == REMOVE) {
                        printf("In %s: (%d) REMOVE %d from grain %d\n", __func__, vrt->id,
                            dev_vertices[vrt->mod[j].vrt_id].id, dev_grains[vrt->mod[j].grn_id].id);
                    }
                }
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Fix the grain lists using the commands given by each vertex.
 *
 * @param dev_grains Pointer to grain device array
 * @param n_grains   Number of grains
 * @param n_blocks   Number of CUDA blocks
 * @param n_threads  Number of CUDA threads
 * @param debug_lvl  Level of debug
 */
inline void fix_grain_lists(grain* dev_grains, int n_grains, vertex* dev_vertices,
                           int n_blocks, int n_threads, int debug_lvl) {
    // Check all the things to be done
    if(debug_lvl >= 1) {
        check_movements<<<n_blocks, n_threads>>>(dev_grains, n_grains, dev_vertices);
        CERR();
    }
    perform_ADD_ops<<<1,1>>>(dev_grains, n_grains, dev_vertices, debug_lvl);
    CERR();
    cudaDeviceSynchronize();
    perform_REMOVE_ops<<<1,1>>>(dev_grains, n_grains, dev_vertices, debug_lvl);
    CERR();
    cudaDeviceSynchronize();
    perform_REPLACE_ops<<<1,1>>>(dev_grains, n_grains, dev_vertices, debug_lvl);
    CERR();
    cudaDeviceSynchronize();
    clear_grains_fix_state<<<n_blocks, n_threads>>>(dev_grains, n_grains);
    CERR();
}

#endif // TOPOLOGICAL_H