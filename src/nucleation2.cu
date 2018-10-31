#ifndef NUCLEATION_CU
#define NUCLEATION_CU

#include <curand_kernel.h>

/**
 * Init random number generators
 *
 * @param seed
 * @param state
 */
__global__ void rand_setup(curandState *state)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(32, tid, 0, &state[tid]);
}


/**
 * Get ids of vertices, boundaries and grain that are free to use
 * for nucleation process.
 *
 * @param dev_grains     Pointer to grain device array
 * @param n_grains       Number of grains
 * @param dev_boundaries Pointer to boundary device array
 * @param n_boundaries   Number of boundaries
 * @param dev_vertices   Pointer to vertices array
 * @param n_vertices     Number of vertices
 * @param dev_vrt_ids    Pointer to device array where to store the free vertex id
 * @param dev_bnd_ids    Pointer to device array where to store the free boundaries ids
 * @param dev_grn_id     Pointer to device array where to store the free grains ids
 */
__global__ void get_nucleation_ids(const grain* dev_grains, const int n_grains,
                                   const boundary* dev_boundaries, const int n_boundaries,
                                   const vertex* dev_vertices, const int n_vertices,
                                   int* dev_vrt_ids, int* dev_bnd_ids, int* dev_grn_id) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0) {
        // Set all ids to invalid values prior to the search
        dev_vrt_ids[0] = -1; dev_vrt_ids[1] = -1;
        dev_bnd_ids[0] = -1; dev_bnd_ids[1] = -1; dev_bnd_ids[2] = -1;
        dev_grn_id[0] = -1;
        // Search for the ids
        int i;
        for(i = 0; i < n_grains; i++) {
            if(!dev_grains[i].enabled) {
                dev_grn_id[0] = i;
                break;
            }
        }
        int found_bnd = 0;
        for(i = 0; i < n_boundaries; i++) {
            if(!dev_boundaries[i].enabled) {
                dev_bnd_ids[found_bnd] = i;
                found_bnd++;
                if(found_bnd == 3) {
                    break;
                }
            }
        }
        int found_vrt = 0;
        for(i = 0; i < n_vertices; i++) {
            if(!dev_vertices[i].enabled) {
                dev_vrt_ids[found_vrt] = i;
                found_vrt++;
                if(found_vrt == 2) {
                    break;
                }
            }
        }
    }
}


/**
 * Given a vertex, computes the position of a grain created
 * with a certain given radius. The new vertices A, B and C
 * lies on the former boundaries AD, AE and AF respectively.
 * Also pointers to g1, g2 and g3 are returned.
 * The delta energy of introducing the new configuration is also returned
 *
 *
 *                                      D
 *                                      |
 *        D                             |
 *        |                            [A]
 *    g2  |  g1                   [g2] / \  [g1]
 *        A       == Build =>         /[g]\
 *      /   \                       [B]---[C]
 *     / g3  \                      /       \
 *    E       F                    /  [g3]   \
 *                               [E]         [F]
 *
 *
 * @param  A              Vertex site for nucleation
 * @param  energy_eps     Epsilon for grain boundary energy function
 * @param  energy_scaling Scaling for grain boundary energy
 * @param  Apos           New position of vertex A
 * @param  Bpos           New position of vertex B
 * @param  Cpos           New position of vertex C
 * @param  D              Pointer to vertex D
 * @param  E              Pointer to vertex E
 * @param  F              Pointer to vertex F
 * @param  AD             Pointer to boundary A-D
 * @param  AE             Pointer to boundary A-E
 * @param  AF             Pointer to boundary A-F
 * @param  g1             Neighbor grain g1 of A
 * @param  g2             Neighbor grain g2 of A
 * @param  g3             Neighbor grain g3 of A
 * @param  DeltaE         Computed Delta E
 * @param  DOMAIN_BOUND   Numerical domain bound
 * @param  do_scale       If true, scale the size of the nucleating grain,
 *                        useful to compute Delta^2 E
 * @return
 */
__device__ int generate_grain_for_nucleation(vertex *A, double energy_eps, double energy_scaling,
    vector2 &Apos, vector2 &Bpos, vector2 &Cpos, vertex *&D, vertex *&E, vertex *&F,
    boundary *&AD, boundary *&AE, boundary *&AF, grain *&g1, grain *&g2, grain *&g3,
    double &DeltaE, const double DOMAIN_BOUND, bool do_scale)
{
    double E0 = 0.0, E1 = 0.0;
    double LAB, LBC, LCA, A1, A2, A3;
    AD = A->boundaries[0];
    AE = A->boundaries[1];
    AF = A->boundaries[2];
    // Find vertices D, E and F
    D = A == AD->ini ? AD->end : AD->ini;
    E = A == AE->ini ? AE->end : AE->ini;
    F = A == AF->ini ? AF->end : AF->ini;
    // Find g1, g2 and g3
    grain *grainsAD[2], *grainsAE[2], *grainsAF[2];
    grains_intersect(A, D, grainsAD);
    grains_intersect(A, E, grainsAE);
    grains_intersect(A, F, grainsAF);
    // Find g1 and g2
    if(grainsAD[0] == grainsAE[0] || grainsAD[0] == grainsAE[1]) {
        g2 = grainsAD[0];
        g1 = grainsAD[1];
    } else if(grainsAD[1] == grainsAE[0] || grainsAD[1] == grainsAE[1]) {
        g2 = grainsAD[1];
        g1 = grainsAD[0];
    } else {
        printf("In %s: Grains g1, g2 were not found for vertex %d.\n[%d %d], [%d %d], [%d %d]\n\n",
         __func__, A->id, grainsAD[0]->id, grainsAD[1]->id, grainsAE[0]->id, grainsAE[1]->id,
         grainsAF[0]->id, grainsAF[1]->id);
        return -1;
    }
    // Find g3
    if(grainsAE[0] == g2) {
        g3 = grainsAE[1];
    } else if(grainsAE[1] == g2) {
        g3 = grainsAE[0];
    } else {
        printf("In %s: Grain g3 was not found for vertex %d.\n", __func__, A->id);
        print_vertex(A);
        return -1;
    }
    // Build position of new vertices A, B and C
    vector2 tmp = vector2_delta_to(D->pos, A->pos, DOMAIN_BOUND);
    double w = A->nucl_r / vector2_mag(tmp);
    if(do_scale) { w *= 0.99; }
    Apos.x = w * tmp.x + A->pos.x;
    Apos.y = w * tmp.y + A->pos.y;
    Apos = vector2_adjust(Apos, DOMAIN_BOUND);

    tmp = vector2_delta_to(E->pos, A->pos, DOMAIN_BOUND);
    w = A->nucl_r / vector2_mag(tmp);
    if(do_scale) { w *= 0.99; }
    Bpos.x = w * tmp.x + A->pos.x;
    Bpos.y = w * tmp.y + A->pos.y;
    Bpos = vector2_adjust(Bpos, DOMAIN_BOUND);

    tmp = vector2_delta_to(F->pos, A->pos, DOMAIN_BOUND);
    w = A->nucl_r / vector2_mag(tmp);
    if(do_scale) { w *= 0.99; }
    Cpos.x = w * tmp.x + A->pos.x;
    Cpos.y = w * tmp.y + A->pos.y;
    Cpos = vector2_adjust(Cpos, DOMAIN_BOUND);

    // Build arclengths of nucleated grain
    LAB = compute_arclength(Apos, Bpos, DOMAIN_BOUND);
    LBC = compute_arclength(Bpos, Cpos, DOMAIN_BOUND);
    LCA = compute_arclength(Cpos, Apos, DOMAIN_BOUND);
    // Build areas of nucleated grain
    double xx[3], yy[3];
    xx[0] = A->pos.x; xx[1] = Cpos.x; xx[2] = Apos.x;
    yy[0] = A->pos.y; yy[1] = Cpos.y; yy[2] = Apos.y;
    A1 = compute_area(xx, yy, 3, DOMAIN_BOUND);
    xx[0] = A->pos.x; xx[1] = Apos.x; xx[2] = Bpos.x;
    yy[0] = A->pos.y; yy[1] = Apos.y; yy[2] = Bpos.y;
    A2 = compute_area(xx, yy, 3, DOMAIN_BOUND);
    xx[0] = A->pos.x; xx[1] = Bpos.x; xx[2] = Cpos.x;
    yy[0] = A->pos.y; yy[1] = Bpos.y; yy[2] = Cpos.y;
    A3 = compute_area(xx, yy, 3, DOMAIN_BOUND);

    // Compute E0
    for(int i = 0; i < 3; i++) {
        if(do_scale) {
            E0 += 0.99 * A->nucl_r * A->boundaries[i]->energy;
        } else {
            E0 += A->nucl_r * A->boundaries[i]->energy;
        }
    }
    E0 += A1*g1->SE + A2*g2->SE + A3*g3->SE;
    E1 = boundary_energy_func(A->nucl_ori - g1->orientation, energy_eps) * energy_scaling * LCA;
    E1 += boundary_energy_func(A->nucl_ori - g2->orientation, energy_eps) * energy_scaling * LAB;
    E1 += boundary_energy_func(A->nucl_ori - g3->orientation, energy_eps) * energy_scaling * LBC;

    // Compute Delta Energy
    DeltaE = E1-E0;
    return 0;
}


/**
 * Compute the nucleation factor for each vertex.
 * This factor is useful to decide if a vertex is suitable for a nucleation site.
 *
 *
 *                                      D
 *                                      |
 *        D                             |
 *        |                            [A]
 *    g2  |  g1                   [g2] / \  [g1]
 *        A       == Build =>         /[g]\
 *      /   \                       [B]---[C]
 *     / g3  \                      /       \
 *    E       F                    /  [g3]   \
 *                               [E]         [F]
 *
 *
 * @param dev_vertices   Pointer to vertex device array
 * @param n_vertices     Number of vertices
 * @param energy_eps     Epsilon for grain boundary energy function
 * @param energy_scaling Scaling for grain boundary energy
 * @param DOMAIN_BOUND   Numerical domain bound
 * @param state          Random number generator state
 */
__global__ void compute_nucleation_factor(vertex* dev_vertices, const int n_vertices,
    const double energy_eps, const double energy_scaling, const double DOMAIN_BOUND, curandState *state)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[tid];

    vertex *vrt;
    while(tid < n_vertices) {
        vrt = &dev_vertices[tid];
        if(vrt->enabled) {
            vrt->DeltaE = 0;
            vrt->nucleation_factor = 0;
            vertex *D, *E, *F;
            boundary *AD, *AE, *AF;
            grain *g1, *g2, *g3;
            vector2 Apos, Bpos, Cpos;
            double DeltaE, DeltaE_;
            // Find min arclength around vertex
            double ALs[3], min_AL;
            for(int i = 0; i < 3; i++) {
                ALs[i] = vrt->boundaries[i]->arclength;
            }
            min_AL = min(ALs, 3);
            vrt->nucl_r = 0.8 * min_AL * curand_uniform_double(&localState) + 0.1;
            vrt->nucl_ori = 2 * M_PI * curand_uniform_double(&localState);
            // Generate nucleating grain configuration and obtain the Delta Energy
            // This call stores the used values of orientations and r
            generate_grain_for_nucleation(vrt, energy_eps, energy_scaling,
                                            Apos, Bpos, Cpos, D, E, F, AD, AE, AF, g1, g2, g3,
                                            DeltaE, DOMAIN_BOUND, false);
            vrt->DeltaE = DeltaE;
            // Perturb the nucleation scaling and obtain a perturbed Delta Energy
            generate_grain_for_nucleation(vrt, energy_eps, energy_scaling,
                                            Apos, Bpos, Cpos, D, E, F, AD, AE, AF, g1, g2, g3,
                                            DeltaE_, DOMAIN_BOUND, true);
            // Estimate Delta^2 Energy from this difference
            // We want to know how the sign behaves
            vrt->nucleation_factor = DeltaE - DeltaE_;
        }
        tid += gridDim.x * blockDim.x;
    }
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    state[tid] = localState;
}

/**
 * Choose randomly a vertex for nucleation per block
 *
 * @param dev_vertices   Pointer to vertex device array
 * @param n_vertices     Number of vertices
 * @param state          Random number generator state
 * @param dev_buffer     Device buffer to write results per block
 */
__global__ void get_vertex_id_per_block(vertex* dev_vertices, int n_vertices, curandState *state, int *dev_buffer)
{
    // This is the random choice for each thread
    __shared__ int tmp_data[2*N_TRDS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Get the state of random numbers
    curandState localState = state[tid];
    // Counter of alive vertices and the random choice per vertex
    int n_alive = 0, rand_vrt_id = -1, n_steps = 0;
    // Each thread will check the data counting
    while(tid < n_vertices) {
        if(dev_vertices[tid].enabled) {
            n_alive++;
        }
        tid += gridDim.x * blockDim.x;
    }
    // Saving the number of alive vertices for each thread
    tmp_data[N_TRDS+threadIdx.x]=n_alive;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    // At this point all threads within a block have counted alives
    // Now we pick a random number and each thread will take those steps
    if(n_alive > 0) {
        n_steps = (int) (curand_uniform_double(&localState) * (n_alive-1));

        int k = 0;
        while(tid < n_vertices) {
            if(dev_vertices[tid].enabled) {
                if(k == n_steps) {
                    rand_vrt_id = dev_vertices[tid].id;
                    break;
                }
                k++;
            }
            tid += gridDim.x * blockDim.x;
        }
        tmp_data[threadIdx.x] = rand_vrt_id;
    } else {
        tmp_data[threadIdx.x] = -1;
    }
    __syncthreads();

    if(threadIdx.x == 0) {
        // Counting number of alive vertices per block
        int n_alives_per_block=0;
        for(int i = 0; i < N_TRDS; i++) {
            n_alives_per_block+=tmp_data[N_TRDS+i];
        }

        if(n_alives_per_block > 0) {
            double p_random = curand_uniform_double(&localState);
            //printf("%d--->%.16f\n", blockIdx.x, p_random);
            double p_cummulative=0.0;
            for(int i = 0; i < N_TRDS; i++) {
                if(tmp_data[i] != -1 && p_cummulative<= p_random
                    && p_random < (p_cummulative+(tmp_data[N_TRDS+i]/(double)n_alives_per_block))) {
                    tmp_data[0] = tmp_data[i];
                    tmp_data[N_TRDS+0] = tmp_data[N_TRDS+i];
                    break;
                }
                p_cummulative+=tmp_data[N_TRDS+i]/(double)n_alives_per_block;
            }
            dev_buffer[blockIdx.x] = tmp_data[0];
            dev_buffer[N_BLKS+blockIdx.x] = n_alives_per_block;
        } else {
            dev_buffer[blockIdx.x] = -1;
            dev_buffer[N_BLKS+blockIdx.x] = 0;
        }
    }
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    state[tid] = localState;
}

/**
 * Get vertex id using Monte Carlo reduction
 *
 * @param dev_vertices         Pointer to vertex device array
 * @param n_vertices           Number of vertices
 * @param state                Random number generator state
 * @param dev_buffer           Device buffer to write partial results
 * @param dev_vrt_id_candidate Chosen vertex
 * @param MC_k                 Monte Carlo constant k = 1 / (kBT)
 */
__global__ void get_vertex_id2(vertex* dev_vertices, int n_vertices, curandState *state,
                 int *dev_buffer, int *dev_vrt_id_candidate, double MC_k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0) {
        for(int i = 0; i < 2*N_BLKS;i++) {
                printf("%d ", dev_buffer[i]);
        }printf("\n");
        // Get the state of random numbers
        curandState localState = state[tid];
        int n_alives_total=0;
        for(int i = 0; i < N_BLKS; i++) {
            n_alives_total += dev_buffer[N_BLKS+i];
        }
        for(int i = 0; i < 2*N_BLKS;i++) {
                printf("%d ", dev_buffer[i]);
        }printf("\n");
        double p_random = curand_uniform_double(&localState);
        double p_cummulative=0.0, p_next = 0.0;
        for(int i = 0; i < N_BLKS; i++) {

            p_next = ((double)dev_buffer[N_BLKS+i])/n_alives_total;
            if(dev_buffer[i] != -1 && p_cummulative < p_random && p_random <= (p_cummulative+p_next)) {
                dev_vrt_id_candidate[0] = dev_buffer[i];
                break;
            }
            p_cummulative += p_next;
        }

        printf("Chose vertex %d for nucleation\n", dev_vrt_id_candidate[0]);

        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////
        printf("%d\n", dev_vrt_id_candidate[0]);
        vertex *vrt = &dev_vertices[dev_vrt_id_candidate[0]];
        vrt->nucleate = false;
        // MC rule
        if(vrt->DeltaE < 0) {
            vrt->nucleate = true;
            printf("[GREEN] Random vertex %d at [%.16f %.16f]\n", vrt->id, vrt->pos.x, vrt->pos.y);
            printf("Nucleation allowed with Delta E < 0\n");
            printf("Delta E = %.16f\nDelta^2 E = %.16f\n", vrt->DeltaE, vrt->nucleation_factor);
        } else {
            if(vrt->nucleation_factor < 0) {
                double p = curand_uniform_double(&localState);
                double threshold = exp(-MC_k * vrt->DeltaE);
                if(p < threshold) {
                    vrt->nucleate = true;
                    printf("[YELLOW1] Random vertex %d at [%.16f %.16f]\n", vrt->id, vrt->pos.x, vrt->pos.y);
                    printf("Nucleation allowed by some probability\n");
                    printf("Accepted (+/-).\np = %.16f\nthreshold = %.16f\nDelta E = %.16f\nDelta^2 E = %.16f\n",
                        p, threshold, vrt->DeltaE, vrt->nucleation_factor);
                    //printf("Delta E = %.16f\nDelta^2 E = %.16f\n", vrt->DeltaE, vrt->nucleation_factor);
                } else {
                    printf("[YELLOW2] Random vertex %d rejected (+/-).\np = %.16f\nthreshold = %.16f\nDelta E = %.16f\nDelta^2 E = %.16f\n",
                        vrt->id, p, threshold, vrt->DeltaE, vrt->nucleation_factor);
                    dev_vrt_id_candidate[0] = -1;
                }
            } else {
                printf("[RED] Random vertex %d rejected (+/+).\nDelta E = %.16f\nDelta^2 E = %.16f\n", vrt->id,
                    vrt->DeltaE, vrt->nucleation_factor);
                dev_vrt_id_candidate[0] = -1;
            }
        }
        state[tid] = localState;
    }
}

/* Brackets has been added to indicate what "data structure" need to be updated.
 *
 *                                      D
 *                                      |
 *        D                             |
 *        |                            [A]
 *    g2  |  g1                   [g2] / \  [g1]
 *        A       == Build =>         /[g]\
 *      /   \                       [B]---[C]
 *     / g3  \                      /       \
 *    E       F                    /  [g3]   \
 *                               [E]         [F]
 *
 *
 *
 * @param dev_grains           Pointer to grain device array
 * @param n_grains             Number of grains
 * @param dev_boundaries       Pointer to boundary device array
 * @param n_boundaries         Number of boundaries
 * @param dev_vertices         Pointer to vertex device array
 * @param n_vertices           Number of vertices
 * @param dev_vrt_id_candidate Chosen vertex
 * @param dev_vrt_ids          Free vertex ids
 * @param dev_bnd_ids          Free boundary ids
 * @param dev_grn_id           Free grain id
 * @param energy_eps           Epsilon for grain boundary energy function
 * @param energy_scaling       Scaling for grain boundary energy
 * @param DOMAIN_BOUND         Numerical domain bound
 */
__global__ void nucleate(grain* dev_grains, int n_grains,
                         boundary* dev_boundaries, int n_boundaries,
                         vertex* dev_vertices, int n_vertices, int* dev_vrt_id_candidate,
                         int* dev_vrt_ids, int* dev_bnd_ids, int* dev_grn_id,
                         double energy_eps, double energy_scaling, const double DOMAIN_BOUND)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Check that only one thread executes this code
    if(tid == 0) {
        // Check that all the selected data (vertices ids, boundaries ids
        // grains id and vertex to be replaced) are different values
        // which means that past kernels computed valid values.
        bool diff_vrt = (dev_vrt_ids[0] != dev_vrt_ids[1]);
        bool diff_bnd = (dev_bnd_ids[0] != dev_bnd_ids[1] &&
                         dev_bnd_ids[1] != dev_bnd_ids[2] &&
                         dev_bnd_ids[2] != dev_bnd_ids[0]);
        bool diff_grn = (dev_grn_id[0] != -1);
        bool diff_vrt_candidate = dev_vrt_id_candidate[0] != -1;
        if(!diff_vrt_candidate) {
            printf("In %s: Attempt to nucleate failed: no candidate was given\n", __func__);
            return;
        }
        if(!diff_bnd || !diff_vrt || !diff_grn) {
            printf("In %s: Attempt to nucleate failed: no free memory\n", __func__);
            return;
        }

        vertex *A = &dev_vertices[dev_vrt_id_candidate[0]];
        if(!A->nucleate) {
            printf("In %s: Vertex %d was chosen randomly but will not nucleate.\n",__func__, A->id);
            return;
        }

        printf("Nucleating at vertex %d with position (%.16f, %.16f)\n", A->id, A->pos.x, A->pos.y);

        // STAGE 1: Recover available ids for nucleation.
        // Recover vertices available ids to build verticec B and C
        vertex *B = &dev_vertices[dev_vrt_ids[0]];
        vertex *C = &dev_vertices[dev_vrt_ids[1]];
        // Recover boundaries available ids to build A-B, B-C and C-A
        boundary *AB = &dev_boundaries[dev_bnd_ids[0]];
        boundary *BC = &dev_boundaries[dev_bnd_ids[1]];
        boundary *CA = &dev_boundaries[dev_bnd_ids[2]];
        // Recover grain ids
        grain *g = &dev_grains[dev_grn_id[0]];

        // Recover information of neighbor structure
        vector2 Apos, Bpos, Cpos;
        vertex *D, *E, *F;
        boundary *AD, *AE, *AF;
        grain *g1, *g2, *g3;
        double DeltaE;
        generate_grain_for_nucleation(A, energy_scaling, energy_eps, Apos, Bpos, Cpos,
            D, E, F, AD, AE, AF, g1, g2, g3, DeltaE, DOMAIN_BOUND, true);

        printf("Nucleation data:\n");
        printf("\tNeighbors: [%d %d %d]\n", D->id, E->id, F->id);
        printf("\tVertex %d position: (%.16f, %.16f)\n", D->id, D->pos.x, D->pos.y);
        printf("\tVertex %d position: (%.16f, %.16f)\n", E->id, E->pos.x, E->pos.y);
        printf("\tVertex %d position: (%.16f, %.16f)\n", F->id, F->pos.x, F->pos.y);

        // This computation is just for debugging
        grain *tmp_gic[2] = {NULL, NULL};
        grains_intersect(AD->ini, AD->end, tmp_gic);
        printf("\tFormer adjacent grains: [%d %d %d]\n", g1->id, g2->id, g3->id);
        printf("\tFormer boundary AD: [%d %d] with grains [%d %d]\n",AD->ini->id, AD->end->id, tmp_gic[0]->id, tmp_gic[1]->id);
        grains_intersect(AE->ini, AE->end, tmp_gic);
        printf("\tFormer boundary AE: [%d %d] with grains [%d %d]\n", AE->ini->id, AE->end->id, tmp_gic[0]->id, tmp_gic[1]->id);
        grains_intersect(AF->ini, AF->end, tmp_gic);
        printf("\tFormer boundary AF: [%d %d] with grains [%d %d]\n", AF->ini->id, AF->end->id, tmp_gic[0]->id, tmp_gic[1]->id);
        printf("\tFree vertices %d %d\n\tFree boundaries: %d %d %d\n\tFree grain %d\n",
            B->id, C->id, AB->id, BC->id, CA->id, g->id);



        // STAGE 4: Initialize recovered memory
        // Init vertices B and C again.
        // Notice that this effectively clear boundaries list and grain list
        init_vertex(B, Bpos, B->id);
        init_vertex(C, Cpos, C->id);
        A->pos = Apos;
        printf("\tVertex %d position: (%.16f, %.16f)\n", A->id, A->pos.x, A->pos.y);
        printf("\tVertex %d position: (%.16f, %.16f)\n", B->id, B->pos.x, B->pos.y);
        printf("\tVertex %d position: (%.16f, %.16f)\n", C->id, C->pos.x, C->pos.y);
        // Initialize data for boundaries, fix grain lists and boundaries lists
        // Init boundaries information with recovered vertices.
        // Vertices' list of boundaries is updated.
        for(int i = 0; i < 3; i++) {
            A->boundaries[i] = NULL;
        }
        init_boundary(AB, A, B, AB->id);
        init_boundary(BC, B, C, BC->id);
        init_boundary(CA, C, A, CA->id);
        vertex_add_boundary(A, AD);
        // Init grain and add the vertex information
        nucleate_grain(g);
        g->orientation = A->nucl_ori;
        grain_add_vertex(g, A);
        grain_add_vertex(g, B);
        grain_add_vertex(g, C);
        printf("Nucleating grain...\n");
        printf("\tNew boundary AB [%d %d]\n", AB->ini->id, AB->end->id);
        printf("\tNew boundary BC [%d %d]\n", BC->ini->id, BC->end->id);
        printf("\tNew boundary CA [%d %d]\n", CA->ini->id, CA->end->id);
        printf("\tNew grain g(%d) of size %d: [%d %d %d]\n",
             g->id, g->vlen, g->vertices[0]->id, g->vertices[1]->id, g->vertices[2]->id);

        printf("\tOld vertices list for grain g1(%d)\n\t", g1->id);
         for(int i = 0; i < g1->vlen; i++) {
            printf("%d ", g1->vertices[i]->id);
        }printf("\n");
        printf("\tOld vertices list for grain g2(%d)\n\t", g2->id);
         for(int i = 0; i < g2->vlen; i++) {
            printf("%d ", g2->vertices[i]->id);
        }printf("\n");
        printf("\tOld vertices list for grain g3(%d)\n\t", g3->id);
         for(int i = 0; i < g3->vlen; i++) {
            printf("%d ", g3->vertices[i]->id);
        }printf("\n");

        printf("\tGrain set for vertex A(%d) before: [%d %d %d]\n", A->id,
            A->grains[0]->id, A->grains[1]->id, A->grains[2]->id);

        // Build grain list for A
        for(int i = 0; i < 3; i++) {
            if(A->grains[i] == g3) {
                A->grains[i] = g;
                break;
            }
        }

        // Build grain list for B and C
        vertex_add_grain(B, g);
        vertex_add_grain(B, g2);
        vertex_add_grain(B, g3);
        vertex_add_grain(C, g);
        vertex_add_grain(C, g3);
        vertex_add_grain(C, g1);
        printf("\tGrain set for vertex A(%d) After: [%d %d %d]\n", A->id,
            A->grains[0]->id, A->grains[1]->id, A->grains[2]->id);
        printf("\tGrain set for vertex B(%d) After: [%d %d %d]\n", B->id,
            B->grains[0]->id, B->grains[1]->id, B->grains[2]->id);
        printf("\tGrain set for vertex C(%d) After: [%d %d %d]\n", C->id,
            C->grains[0]->id, C->grains[1]->id, C->grains[2]->id);
        // Fix vertices list for g1, g2, g3
        grain_add_vertex(g1, C, A);
        printf("\tNew vertices list for grain g1(%d)\n\t", g1->id);
        for(int i = 0; i < g1->vlen; i++) {
            printf("%d ", g1->vertices[i]->id);
        }printf("\n");
        // TODO: Check if this reference AE is still valid
        if(AE->ini == A) {
            grain_add_vertex(g2, B, AE->end);
        } else if(AE->end == A) {
            grain_add_vertex(g2, B, AE->ini);
        } else {
            printf("In nucleation: We lost reference to vertex E.\n");
        }
        printf("\tNew vertices list for grain g2(%d)\n\t", g2->id);
        for(int i = 0; i < g2->vlen; i++) {
            printf("%d ", g2->vertices[i]->id);
        }printf("\n");
        // g3 is a special case, we first remove A and then we add B and C counterclockwise
        grain_add_vertex(g3, C, A);
        grain_add_vertex(g3, B, C);
        grain_remove_vertex(g3, A);
        printf("\tNew vertices list for grain g3(%d)\n\t", g3->id);
        for(int i = 0; i < g3->vlen; i++) {
            printf("%d ", g3->vertices[i]->id);
        }printf("\n");
        vertex_invert_boundary_order(B);
        vertex_invert_boundary_order(C);
        // Check neighbor vertices of A
        printf("\tVertices neighbors of %d: ", A->id);
        for(int i = 0; i < 3; i++) {
            printf("[%d %d] ", A->boundaries[i]->ini->id, A->boundaries[i]->end->id);
        } printf("\n");
        // Fix boundaries list for B, C, D and F
        // AE is now BE, AF is now CF
        if(AE->ini == A) {
            AE->ini = B;
        } else if(AE->end == A) {
            AE->end = B;
        } else {
            printf("In nucleation: Boundary AE has no vertex A\n");
        }
        B->boundaries[2] = AE;
        printf("\tVertices neighbors of %d: ", B->id);
        for(int i = 0; i < 3; i++) {
            printf("[%d %d] ", B->boundaries[i]->ini->id, B->boundaries[i]->end->id);
        } printf("\n");
        if(AF->ini == A) {
            AF->ini = C;
        } else {
            AF->end = C;
        }
        C->boundaries[2] = AF;
        printf("\tVertices neighbors of %d: ", C->id);
        for(int i = 0; i < 3; i++) {
            printf("[%d %d] ", C->boundaries[i]->ini->id, C->boundaries[i]->end->id);
        } printf("\n");
    }
}

/**
 * Auxiliar function to set some selected grain to be different from the rest
 * with a given stored energy and setting it as nucleated. Useful for experiments.
 *
 * @param dev_grains Pointer to grain device array
 * @param n_grains   Number of grains
 * @param grain_id   Grain id to be nucleated
 * @param SE         Stored Energy value for selected grain
 */
__global__ void set_stored_energy(grain *dev_grains, int n_grains, int grain_id, double SE) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0) {
        if(grain_id < n_grains) {
            dev_grains[grain_id].SE = SE;
            dev_grains[grain_id].type = NUCLEATED;
            printf("In %s: Grain %d has SE = %.16f\n", __func__, grain_id, dev_grains[grain_id].SE);
        }
        else {
            printf("In %s: Grain %d out of range.\n", __func__, grain_id);
        }
    }
}

#endif // NUCLEATION_CU
