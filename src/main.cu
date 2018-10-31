/**************************************
 *                                    *
 *                                    *
 *  Stored Energy Vertex Code (SEVC)  *
 *                                    *
 *                                    *
 **************************************/

#include "geometry.h"
#include "options.h"
#include "calculus.h"
#include "topological.h"
#include "nucleation2.cu"
#include "io.h"
#include "translation.h"
#include <curand_kernel.h>

/**
 * Main program
 */
int main(int argc, char const *argv[])
{
    // Prepare output folders and load initial configuraion
    options opt;
    bool result = load_config_file(argc, argv, &opt);
    if(!result) { return 1; }
    result = create_folders(&opt);
    if(!result) { return 1; }
    // Temporal variables
    double dt = opt.dt;
    // Current simulation time
    double t = 0;
    // Current iteration
    int n = 0;
    // Current removal progress
    int prog = 0;
    // Number of vertices, boundaries and grains
    int n_vertices;
    int n_boundaries;
    int n_grains = opt.n_grains;
    int min_grains;
    if(opt.leave_percent < 0) {
        min_grains = -1;
    } else {
        min_grains = floor(opt.leave_percent*n_grains);
    }
    int n_grains_output = 100000;
    // This is hardcoded to finish with 20000 grains
    min_grains = 5000;
    if(min_grains > 0) {
        printf("Saving up to %d grains.\n", min_grains);
    }
    // Vertices, boundaries and grains data
    vertex *vertices, *dev_vertices;
    boundary *boundaries, *dev_boundaries;
    grain *grains, *dev_grains;
    // Arrays for management of enabled structs
    int *dev_vrt_ids, *dev_bnd_ids, *dev_grn_id;

    // Allocate host memory and load initial condition
    result = load_from_files(opt, vertices, n_vertices, boundaries,
                                  n_boundaries, grains, opt.domain_bound);
    if(!result) { return 1; };
    // Send data to GPU
    host_to_device(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                   dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
    // Reserve memory for enabled structs
    HERR(cudaMalloc(&dev_vrt_ids, sizeof(int) * 2));
    HERR(cudaMalloc(&dev_bnd_ids, sizeof(int) * 3));
    HERR(cudaMalloc(&dev_grn_id, sizeof(int)));
    // Current number of grains
    int curr_n_grains = n_grains;
    // Buffer to store the vertex nucleation candidate
    int* dev_vrt_id_candidate;
    HERR(cudaMalloc(&dev_vrt_id_candidate, sizeof(int) * opt.n_blocks));
    int* vrt_id_candidate = (int*) malloc(sizeof(int) * opt.n_blocks);

    // Init random generator
    curandState *dev_states;
    HERR(cudaMalloc((void **) &dev_states , opt.n_blocks * opt.n_threads * sizeof(curandState)));
    rand_setup<<<opt.n_blocks, opt.n_threads>>>(dev_states);
    CERR();

    // Buffer for gather results of get_vertex_id
    int *dev_get_vertex_buffer;
    HERR(cudaMalloc((void **) &dev_get_vertex_buffer, sizeof(int) * opt.n_blocks * 2));

    // Buffers for computing number of grains
    int *n_grains_buffer, *dev_n_grains_buffer;
    n_grains_buffer = (int*) malloc(opt.n_blocks * sizeof(int));
    HERR(cudaMalloc((void **) &dev_n_grains_buffer, sizeof(int) * opt.n_blocks));

    // Buffers for computing nucleated area
    double *nucl_buffer, *dev_nucl_buffer;
    nucl_buffer = (double*) malloc(opt.n_blocks * sizeof(double));
    HERR(cudaMalloc((void **) &dev_nucl_buffer, sizeof(double) * opt.n_blocks));

    // Buffers for polling system
    bool *polling_buffer, *dev_polling_buffer, *candidate_buffer;
    polling_buffer = (bool*) malloc(opt.n_blocks * sizeof(bool));
    HERR(cudaMalloc((void **) &dev_polling_buffer, sizeof(bool) * opt.n_blocks));
    HERR(cudaMalloc((void **) &candidate_buffer, sizeof(bool) * n_boundaries));

    // Buffers for counting two three sided grains
    int *twothree_buffer, *dev_twothree_buffer;
    twothree_buffer = (int*) malloc(opt.n_blocks * sizeof(int));
    HERR(cudaMalloc((void **) &dev_twothree_buffer, sizeof(int) * opt.n_blocks));

    // Area of nucleated grains
    double nucleated_area = 0;

    // Enable nucleation from a certain point
    bool allow_nucleation = false;

    /* Special experiment. We initially choose a grain and set its SE to 0 */
    //set_stored_energy<<<1,1>>>(dev_grains, n_grains, 1355, 0.0);
    //cudaDeviceSynchronize();

    // Main loop
    while(((t <= opt.tmax && opt.tmax > 0) || (n <=opt.nmax && opt.nmax > 0)) && (curr_n_grains >= min_grains && min_grains > 0))
    {
        printf("n = %d, t = %f\n", n, t);
        // Compute boundary arclengths
        compute_boundary_arclengths<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries, opt.domain_bound);
        CERR();
        // Compute grain areas
        compute_grain_areas<<<opt.n_blocks, opt.n_threads>>>(dev_grains, n_grains, opt.domain_bound);
        CERR();
        // Compute boundary energies given the corrected misorientations of previous timestep
        compute_boundary_energies<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries, opt.GB_eps, opt.GB_scaling);
        CERR();
        // Compute vertex energies which depends on previous updated quantities
        compute_vertex_energies<<<opt.n_blocks, opt.n_threads>>>(dev_vertices, n_vertices);
        CERR();
        /******************** Begin Nucleation ********************/
        // Detect when we reach for first time the number of grains
        if(curr_n_grains <= 5*n_grains/6) {
            allow_nucleation = true;
        }

        if(opt.do_nucleation && (n % opt.nucleation_gap) == 0 && allow_nucleation) {
            // Compute for each vertex the nucleation factor given for each vertex
            // a random orientation for the new grain and a random size of nucleation
            compute_nucleation_factor<<<opt.n_blocks, opt.n_threads>>>(dev_vertices, n_vertices, opt.GB_eps, opt.GB_scaling, opt.domain_bound, dev_states);
            CERR();
            cudaDeviceSynchronize();
            get_vertex_id_per_block<<<opt.n_blocks, opt.n_threads>>>(dev_vertices, n_vertices, dev_states, dev_get_vertex_buffer);
            CERR();
            get_vertex_id2<<<1,1>>>(dev_vertices, n_vertices, dev_states, dev_get_vertex_buffer, dev_vrt_id_candidate, opt.MC_k);
            CERR();
            cudaDeviceSynchronize();
            /*int mybuffer[N_BLKS*2], myvertex[N_BLKS];
            HERR(cudaMemcpy(mybuffer, dev_get_vertex_buffer, sizeof(int)*2*N_BLKS, cudaMemcpyDeviceToHost));
            HERR(cudaMemcpy(myvertex, dev_vrt_id_candidate, sizeof(int)*N_BLKS, cudaMemcpyDeviceToHost));
            printf("From host %d\n", myvertex[0]);
            for(int i = 0; i < 2*N_BLKS;i++) {
                printf("%d ", mybuffer[i]);
            }printf("\n");
            cudaDeviceSynchronize();*/

            get_nucleation_ids<<<1,1>>>(dev_grains, n_grains, dev_boundaries, n_boundaries, dev_vertices, n_vertices, dev_vrt_ids, dev_bnd_ids, dev_grn_id);
            CERR();
            cudaDeviceSynchronize();
            nucleate<<<1,1>>>(dev_grains, n_grains, dev_boundaries, n_boundaries,
                              dev_vertices, n_vertices, dev_vrt_id_candidate, dev_vrt_ids,
                              dev_bnd_ids, dev_grn_id, opt.GB_eps, opt.GB_scaling, opt.domain_bound);
            CERR();
            cudaDeviceSynchronize();

            // Compute boundary arclengths
            compute_boundary_arclengths<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries, opt.domain_bound);
            CERR();
            // Compute grain areas
            compute_grain_areas<<<opt.n_blocks, opt.n_threads>>>(dev_grains, n_grains, opt.domain_bound);
            CERR();
            // Compute boundary energies after neighbor changing with nucleation
            compute_boundary_energies<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries, opt.GB_eps, opt.GB_scaling);
            CERR();
            // Compute all the individual vertices energies
            compute_vertex_energies<<<opt.n_blocks, opt.n_threads>>>(dev_vertices, n_vertices);
            CERR();
        }
        /********************** End Nucleation **********************/

        if(opt.debug_lvl >= 1) {
            printf("After Nucleation\n");
            device_to_host(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                           dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            result = check_data(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains);
            host_to_device(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                           dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            if(!result) {
                printf("Error during nucleation\n");
                break;
            }
            cudaDeviceSynchronize();
        }

        // Compute dihedral angles
        compute_dihedral_angles<<<opt.n_blocks, opt.n_threads>>>(dev_vertices, n_vertices, opt.domain_bound);
        CERR();
        // Compute vertices velocities
        compute_vertex_velocities<<<opt.n_blocks, opt.n_threads>>>(dev_vertices, n_vertices, opt.domain_bound);
        CERR();
        // Compute dA/dt
        compute_grain_dAdts<<<opt.n_blocks, opt.n_threads>>>(dev_grains, n_grains, opt.domain_bound);
        CERR();

        /********************** Handle Topological Transitions **********************/

        // Compute boundaries extinction time
        compute_t_ext<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries, dt, opt.domain_bound);
        CERR();
        //cudaDeviceSynchronize();
        // Label boundaries with 0 <= t_ext <= dt as candidates
        label_boundaries<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries, dt);
        CERR();
        // The polling system
        polling_system(dev_vertices, n_vertices, dev_boundaries, n_boundaries, polling_buffer, dev_polling_buffer,
                                candidate_buffer, opt.n_blocks, opt.n_threads);
        CERR();
        check_conflictive_boundaries<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries);
        CERR();
        unlabel_conflictive_boundaries<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries);
        CERR();
        // Remove degenerated configuration of two 3 sided grains sharing a boundary about to flip
        for(int _i = 0; _i < 20; _i++) {
            int n_two_three_sided_grains = two_three_sided_grains(dev_boundaries, n_boundaries,
                                                                  opt.n_blocks, opt.n_threads,
                                                                  twothree_buffer, dev_twothree_buffer);
            printf("%d configurations of two three sided grains\n", n_two_three_sided_grains);
            if(n_two_three_sided_grains == 0)
                break;
            remove_two_grains_configurations<<<1, 1>>>(dev_boundaries, n_boundaries, opt.debug_lvl);
            CERR();
            fix_grain_lists(dev_grains, n_grains, dev_vertices, opt.n_blocks, opt.n_threads, opt.debug_lvl);
            if(opt.debug_lvl >= 1) {
                printf("After remove two grains configuration in subiter %d.\n", _i);
                device_to_host(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                               dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
                result = check_data(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains);
                host_to_device(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                               dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
                if(!result) {
                    printf("Error during remove two grains configuration.\n");
                    break;
                }
                cudaDeviceSynchronize();
            }
        }
        // Clear candidate state of grains
        clear_grains_state<<<opt.n_blocks, opt.n_threads>>>(dev_grains, n_grains);
        CERR();
        // Label 3 sided grains with boundaries marked to flip
        label_3sided_grains<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries, dev_grains, n_grains, dt);
        CERR();
        // Remove grains
        //remove_grains<<<opt.n_blocks, opt.n_threads>>>(dev_grains, n_grains, opt.domain_bound);
        remove_grains<<<1,1>>>(dev_grains, n_grains, opt.domain_bound);
        CERR();
        // Fix grains lists
        fix_grain_lists(dev_grains, n_grains, dev_vertices, opt.n_blocks, opt.n_threads, opt.debug_lvl);
        CERR();
        if(opt.debug_lvl >= 1) { cudaDeviceSynchronize(); }
        // Compute grain areas
        compute_grain_areas<<<opt.n_blocks, opt.n_threads>>>(dev_grains, n_grains, opt.domain_bound);
        CERR();
        // Check errors
        if(opt.debug_lvl >= 1) {
            printf("After Grain Removal\n");
            device_to_host(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                           dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            result = check_data(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains);
            host_to_device(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                           dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            if(!result) {
                printf("Error during Grain Removal\n");
                break;
            }
        }
        // Unlabel boundaries related to newly created 3 sided grains
        unlabel_boundaries_st2<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries);
        CERR();
        // Unlabel boundaries related to 4 sided grains which may have two flippings
        //unlabel_boundary_4sided_grains<<<opt.n_blocks, opt.n_threads>>>(dev_grains, n_grains);
        CERR();
        if(opt.debug_lvl >= 1) { cudaDeviceSynchronize(); }
        // Apply flippings
        apply_flippings<<<opt.n_blocks, opt.n_threads>>>(dev_boundaries, n_boundaries, opt.GB_eps, opt.domain_bound);
        CERR();
        cudaDeviceSynchronize();
        //update_vertices_list<<<opt.n_blocks, opt.n_threads>>>(dev_vertices, n_vertices);
        // Fix grains lists
        fix_grain_lists(dev_grains, n_grains, dev_vertices, opt.n_blocks, opt.n_threads, opt.debug_lvl);
        CERR();
        if(opt.debug_lvl >= 1) { cudaDeviceSynchronize(); }
        if(opt.debug_lvl >= 1) {
            printf("After Flipping\n");
            device_to_host(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                           dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            result = check_data(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains);
            host_to_device(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                           dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            if(!result) {
                printf("Error during flipping\n");
                break;
            }
        }


        /********************** End Handle Topological Transitions **********************/

        // Update positions
        update_vertices_positions<<<opt.n_blocks, opt.n_threads>>>(dev_vertices, n_vertices, dt, opt.domain_bound);
        CERR();
        // Get current number of grains in system
        curr_n_grains = current_n_grains(dev_grains, n_grains, opt.n_blocks, opt.n_threads, n_grains_buffer,
            dev_n_grains_buffer);
        CERR();
        // Get current percentage of nucleated area
        nucleated_area = nucleated_fraction_area(dev_grains, n_grains, opt.n_blocks, opt.n_threads, nucl_buffer,
            dev_nucl_buffer, opt.domain_bound);
        CERR();
        printf("Number of grains: %d\n", curr_n_grains);
        printf("Nucleated area %%: %.16f\n", nucleated_area);

        // Export data after some iterations
        if(n % opt.snap_gap == 0 || (curr_n_grains <= min_grains && min_grains > 0)) {
            //if(curr_n_grains <= 20000) {
                printf("Saving at n=%d, t=%.16f, n_grains=%d, nucl_area=%.16f\n", n, t, curr_n_grains, nucleated_area);
                device_to_host(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                               dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
                result = check_data(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains);
                if(!result) {
                    printf("Error while saving output file. Exiting...\n");
                    break;
                } else {
                    printf("Valid data. Saving...\n");
                }
                export_data(vertices, n_vertices, boundaries, n_boundaries,
                       grains, n_grains, prog, t, opt);
                host_to_device(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                               dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            /*} else {
                printf("I can't save yet, Im in n grains %d\n", curr_n_grains);
            }*/
            prog++;
        }

        // Export data after some iterations
        /*if(nucleated_area >= prog * 5 || (curr_n_grains <= min_grains && min_grains > 0)) {
            printf("Saving at n=%d, t=%.16f, n_grains=%d, nucl_area=%.16f\n", n, t, curr_n_grains, nucleated_area);
            device_to_host(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                           dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            result = check_data(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains);
            if(!result) {
                printf("Error while saving output file. Exiting...\n");
                break;
            } else {
                printf("Valid data. Saving...\n");
            }
            export_data(vertices, n_vertices, boundaries, n_boundaries,
                   grains, n_grains, prog, t, opt);
            host_to_device(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                           dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            prog++;
        }*/

        /*// Save each 20000 removed grains
        if(curr_n_grains <= n_grains_output) {
            printf("Saving at n=%d, t=%.16f, n_grains=%d, nucl_area=%.16f\n", n, t, curr_n_grains, nucleated_area);
            device_to_host(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                           dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            result = check_data(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains);
            if(!result) {
                printf("Error while saving output file. Exiting...\n");
                break;
            } else {
                printf("Valid data. Saving...\n");
            }
            export_data(vertices, n_vertices, boundaries, n_boundaries,
                   grains, n_grains, prog+n_grains_output, t, opt);
            host_to_device(vertices, n_vertices, boundaries, n_boundaries, grains, n_grains,
                           dev_vertices, dev_boundaries, dev_grains, opt.n_blocks, opt.n_threads);
            n_grains_output -= 20000;
        }*/

        n++;
        t = n*dt;

        if(curr_n_grains <= min_grains && min_grains > 0) {
            break;
        }
    }

    printf("\nPerformed %d iterations.\n", n-1);
    curr_n_grains = current_n_grains(dev_grains, n_grains, opt.n_blocks, opt.n_threads, n_grains_buffer, dev_n_grains_buffer);
    printf("Number of grains: %d\n", curr_n_grains);
    // Free n_grains buffers
    cudaFree(dev_get_vertex_buffer);
    cudaFree(dev_n_grains_buffer);
    cudaFree(dev_nucl_buffer);
    cudaFree(dev_polling_buffer);
    cudaFree(candidate_buffer);
    cudaFree(dev_twothree_buffer);
    // Free device enabled struct bool
    cudaFree(dev_vrt_ids);
    cudaFree(dev_bnd_ids);
    cudaFree(dev_grn_id);
    cudaFree(dev_vrt_id_candidate);
    // Free random state
    cudaFree(dev_states);
    // Free host memory
    free(n_grains_buffer);
    free(nucl_buffer);
    free(polling_buffer);
    free(twothree_buffer);
    free(vrt_id_candidate);
    free(vertices);
    free(boundaries);
    free(grains);
    printf("Done.\n");
    return 0;
}