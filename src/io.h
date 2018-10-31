#ifndef IO_H
#define IO_H

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include "geometry.h"
#include "options.h"

/**
 * Input Output Module.
 * Manages initial condition loading and export results
 */

/**
 * Create folders to export data
 * @param opt Options struct
 */
inline bool create_folders(options *opt) {
    struct stat st = {0};
    char command[MAX_FNAME_SIZE];
    if(opt->output_folder[0] == '\0') {
        time_t t = time(NULL);
        struct tm tm = *localtime(&t);
        // Create default folder with current date and time
        strftime(opt->output_folder, MAX_FNAME_SIZE,"%Y%m%d_%H%M%S", &tm);
        if(opt->debug_lvl >= 0) {
            printf("No folder was found in .ini\n");
        }
    }
    else if (stat(opt->output_folder, &st) != -1) {
        strcpy(command, "rm -rf ");
        strcat(command, opt->output_folder);
        printf("Folder already exists. Overwritting...\n");
        system(command);
        if(opt->debug_lvl >= 0) {
            printf("%s\n", command);
        }
    }
    // Create structure
    mkdir(opt->output_folder, 0775);
    strcpy(command, opt->output_folder);
    strcat(command, "/DATA");
    mkdir(command, 0775);
    strcpy(opt->output_folder, command);
    if(opt->debug_lvl >= 0) {
        printf("mkdir %s\n", opt->output_folder);
    }
    return true;
}

/**
 * Load initial condition from files
 *
 * @param  opt               Options configuration file
 * @param  vertices          Pointer to array of vertices structs
 * @param  n_vertices        Number of vertices counter
 * @param  boundaries        Pointer to array of boundaries structs
 * @param  n_boundaries      Number of boundaries counter
 * @param  grains            Pointer to array of grains structs
 * @param  DOMAIN_BOUND      Numerical domain bound
 * @return                   True if all the files were loaded, false otherwise
 */
inline bool load_from_files(options opt, vertex* &vertices, int &n_vertices,
                            boundary* &boundaries, int &n_boundaries,
                            grain* &grains, const double DOMAIN_BOUND) {
    if(opt.debug_lvl >= 1) {
        printf("Loading data ...\n");
    }
    // Fill vertices information
    n_vertices = 0;
    n_boundaries = 0;
    vector2 vec;

    FILE *vertex_file = fopen(opt.vertex_fname, "r");
    if(vertex_file == NULL)  {
        perror(opt.vertex_fname);
        return false;
    }
    int vertex_size = 1;
    vertices = (vertex*) malloc(vertex_size * sizeof(vertex));
    while(fscanf(vertex_file,"%lf %lf\n",&(vec.x),&(vec.y))!=EOF){
        if(n_vertices == vertex_size){
            vertex_size *= 2;
            vertices = (vertex*) realloc(vertices, vertex_size * sizeof(vertex));
        }
        vec.x = vec.x * DOMAIN_BOUND;
        vec.y = vec.y * DOMAIN_BOUND;
        init_vertex(&vertices[n_vertices], vec, n_vertices);
        n_vertices++;
    }
    fclose(vertex_file);
    if(opt.debug_lvl >= 1) {
        printf("%d vertices loaded.\n", n_vertices);
    }
    // Fill neighbors information
    FILE *boundary_file = fopen(opt.boundary_fname, "r");
    if(boundary_file == NULL) {
        perror(opt.boundary_fname);
        free(vertices);
        return false;
    }
    int a, b;
    int boundary_size = 1;
    int *bnds_ini = (int*) malloc(boundary_size * sizeof(int));
    int *bnds_end = (int*) malloc(boundary_size * sizeof(int));
    while(fscanf(boundary_file, "%d %d\n", &a, &b) != EOF) {
        if(n_boundaries == boundary_size) {
            boundary_size *= 2;
            bnds_ini = (int*) realloc(bnds_ini, boundary_size * sizeof(int));
            bnds_end = (int*) realloc(bnds_end, boundary_size * sizeof(int));
        }
        bnds_ini[n_boundaries] = a;
        bnds_end[n_boundaries] = b;
        n_boundaries += 1;
    }

    if(opt.debug_lvl >= 1) {
        printf("%d boundaries loaded.\n", n_boundaries);
    }
    boundaries = (boundary*) malloc(boundary_size * sizeof(boundary));
    for(int i = 0; i < n_boundaries; i++) {
        init_boundary(&boundaries[i], &vertices[bnds_ini[i]], &vertices[bnds_end[i]], i);
    }
    free(bnds_ini);
    free(bnds_end);
    fclose(boundary_file);

    // Set all the vertices's boundaries clockwise.
    for(int j = 0; j < n_vertices; j++){
        vertex_set_boundaries_clockwise(&vertices[j], DOMAIN_BOUND);
    }

    // Generate grain structure
    int lengrains = 0;
    grains = (grain*) malloc(opt.n_grains * sizeof(grain));

    bool *considerated = (bool*) malloc(3*n_vertices*sizeof(bool));
    for(int k = 0; k < 3*n_vertices; k++) { considerated[k] = false; }
    // Check each junction to be added to three grains
    for(int j = 0; j < n_vertices; j++) {
        vertex *start_vrt = &vertices[j];
        if(start_vrt->enabled) {
            for(int g = 0; g < 3; g++) {
                if(!considerated[j*3 + g]) {
                    grain grn;
                    init_grain(&grn, lengrains);
                    vertex *current_vrt = start_vrt;
                    int current_side = g;
                    do {
                        int j_idx = current_vrt - vertices;
                        //printf("j_idx = %d\n", j_idx);
                        if(considerated[j_idx * 3 + current_side] || !current_vrt->enabled) {
                            printf("%d\n", considerated[j_idx * 3 + current_side]);
                            printf("%d\n", current_vrt->enabled);
                            printf("Illegal grains were detected!\n");
                            exit(1);
                        }
                        considerated[j_idx * 3 + current_side] = true;
                        grain_add_vertex(&grn, current_vrt);
                        boundary *bnd = current_vrt->boundaries[current_side];
                        if(bnd->ini == current_vrt) {
                            current_vrt = bnd->end;
                        } else {
                            current_vrt = bnd->ini;
                        }
                        int i = 0; while(1){
                            if(current_vrt->boundaries[i] == bnd) break;
                            i++;
                        }
                        current_side = (i+1) % 3;
                    } while(current_vrt != start_vrt);
                    grains[lengrains] = grn;
                    lengrains++;
                }
            }
        }
    }

    if(lengrains != opt.n_grains) {
        printf("In %s: Loaded grains (%d) are not equal to expected grains (%d)\n",
         __func__, lengrains, opt.n_grains);
        exit(1);
    }

    // Set all the vertices's boundaries counterclockwise.
    for(int j = 0; j < n_vertices; j++){
        vertex_set_boundaries_clockwise(&vertices[j], DOMAIN_BOUND);
        vertex_invert_boundary_order(&vertices[j]);
    }
    printf("%d grains loaded.\n", lengrains);
    free(considerated);

    for(int i = 0; i < opt.n_grains; i++) {
        for(int j = 0; j < grains[i].vlen; j++) {
            vertex_add_grain(grains[i].vertices[j], &grains[i]);
        }
    }
    // Store orientations and energies
    FILE *ori_file = fopen(opt.orientation_fname, "r");
    if(ori_file == NULL) {
        perror(opt.orientation_fname);
        return false;
    }
    FILE *SE_file = fopen(opt.SE_fname, "r");
    if(ori_file == NULL) {
        perror(opt.SE_fname);
        return false;
    }
    printf("Added grains to vertices\n");
    int i = 0;
    while(fscanf(ori_file, "%lf\n", &(grains[i++].orientation)) != EOF) {}
    // Fix orientations to [0, 2pi]
    for(int j = 0; j < opt.n_grains; j++) {
        grains[j].orientation = fix_orientation(grains[j].orientation);
    }
    i = 0;

    // Load SE from file and find min and max values
    double min_SE = 999999, max_SE = 0;
    while(fscanf(SE_file, "%lf\n", &(grains[i++].SE)) != EOF) {}
    fclose(SE_file);

    // Fix SE according to option file
    // If we don't get SE_min and SE_max from option file we can scale the data
    if(opt.SE_min == 999 && opt.SE_max == 999) {
        for(int j = 0; j < opt.n_grains; j++) {
            grains[j].SE *= opt.SE_eps;
        }
    }
    // If we get the same values of SE_min and SE_max we set to max
    // We can also scale here
    else if(opt.SE_min == opt.SE_max) {
        for(int j = 0; j < opt.n_grains; j++) {
            grains[j].SE = opt.SE_max * opt.SE_eps;
        }
    }
    // Otherwise, we do not scale but rather change the interval
    // Notice that this assumes that te original distribution is in [0,1]
    else {
        double fix_m = (opt.SE_max - opt.SE_min);
        double fix_b = opt.SE_min;
        for(int j = 0; j < opt.n_grains; j++) {
            grains[j].SE = fix_m * grains[j].SE + fix_b;
        }
    }

    for(int j = 0; j < opt.n_grains; j++) {
        if(grains[j].SE < min_SE) {
            min_SE = grains[j].SE;
        }
        if(grains[j].SE > max_SE) {
            max_SE = grains[j].SE;
        }
    }
    fclose(ori_file);

    printf("Effective Min SE: %.16f\n", min_SE);
    printf("Effective Max SE: %.16f\n", max_SE);
    printf("Data loaded.\n");
    return true;
}


/**
 * Export data to output files. Output folder is already specified.
 *
 * @param vertices     Pointer to array of vertices structs
 * @param n_vertices   Number of vertices
 * @param boundaries   Pointer to array of boundaries structs
 * @param n_boundaries Number of boundaries
 * @param grains       Pointer to array of grains structs
 * @param n_grains     Number of grains
 * @param n            Number of iteration (used for output filename)
 * @param time         Current time of simulation
 * @param opt          Options struct
 */
inline bool export_data(vertex* vertices, int n_vertices, boundary* boundaries, int n_boundaries,
                        grain* grains, int n_grains, int n, double time, options opt) {
    // Store vertex info
    char fname[MAX_FNAME_SIZE];
    char ff[12];
    sprintf(ff, "/%06d.txt", n);
    strcpy(fname, opt.output_folder);
    strcat(fname, ff);

    FILE* output_file = fopen(fname, "w");

    if(output_file == NULL) {
        perror(fname);
        return false;
    }
    int curr_n_vertices = 0;
    int curr_n_boundaries = 0;
    int curr_n_grains = 0;
    for(int i = 0; i < n_vertices; i++) { if(vertices[i].enabled) curr_n_vertices++; }
    for(int i = 0; i < n_boundaries; i++) { if(boundaries[i].enabled) curr_n_boundaries++; }
    for(int i = 0; i < n_grains; i++) { if(grains[i].enabled) curr_n_grains++; }

    fprintf(output_file, "TIME %.16f\n", time);
    fprintf(output_file, "DELTA T %.16f\n", opt.dt);
    fprintf(output_file, "DELTA TAU %.16f\n", opt.dtau);
    fprintf(output_file, "MAX TIME %.16f\n", opt.tmax);
    fprintf(output_file, "NUCLEATION %d\n", opt.do_nucleation);
    fprintf(output_file, "DOMAIN BOUND %.16f\n", opt.domain_bound);
    fprintf(output_file, "GB EPSILON %.16f\n", opt.GB_eps);
    fprintf(output_file, "GB SCALING %.16f\n", opt.GB_scaling);
    fprintf(output_file, "SE EPSILON %.16f\n", opt.SE_eps);
    fprintf(output_file, "SE MIN %.16f\n", opt.SE_min);
    fprintf(output_file, "SE MAX %.16f\n", opt.SE_max);
    fprintf(output_file, "NUCLEATION SCALING %.16f\n", opt.nucleation_scaling);
    fprintf(output_file, "MAX VERTICES PER GRN %d\n", MAX_VRT_PER_GRN);
    fprintf(output_file, "VERTICES %d\n", n_vertices);
    fprintf(output_file, "BOUNDARIES %d\n", n_boundaries);
    fprintf(output_file, "GRAINS %d\n", n_grains);
    fprintf(output_file, "CURR VERTICES %d\n", curr_n_vertices);
    fprintf(output_file, "CURR BOUNDARIES %d\n", curr_n_boundaries);
    fprintf(output_file, "CURR GRAINS %d\n", curr_n_grains);
    fprintf(output_file, "#VERTEX ENABLED ENERGY X Y VX VY VXbnd VXgrn VYbnd VYgrn VG1 G2 G3 N1 N2 N3\n");
    for(int i = 0; i < n_vertices; i++) {
        vertex* vrt = &vertices[i];
        fprintf(output_file,"%d %d %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %d %d %d ",
                vertices[i].id, vertices[i].enabled, vertices[i].energy,
                vertices[i].pos.x, vertices[i].pos.y,
                vertices[i].vel.x, vertices[i].vel.y,
                vertices[i].vel_bnd_x, vertices[i].vel_grn_x,
                vertices[i].vel_bnd_y, vertices[i].vel_grn_y,
                vertices[i].grains[0]->id,
                vertices[i].grains[1]->id,
                vertices[i].grains[2]->id);
        for(int j = 0; j < 3; j++) {
            if(vertices[i].boundaries[j]->ini == vrt) {
                if(j == 2)
                    fprintf(output_file, "%d\n", vertices[i].boundaries[j]->end->id);
                else
                    fprintf(output_file, "%d ", vertices[i].boundaries[j]->end->id);
            } else {
                if(j == 2)
                    fprintf(output_file, "%d\n", vertices[i].boundaries[j]->ini->id);
                else
                    fprintf(output_file, "%d ", vertices[i].boundaries[j]->ini->id);
            }
        }
    }

    // Store boundary info
    fprintf(output_file, "#BOUNDARY ENABLED INI END ENERGY ARCLEN EXTTIME CANDIDATE\n");
    for(int i = 0; i < n_boundaries; i++) {
        fprintf(output_file,"%d %d %d %d %.16f %.16f %.16f %d\n", boundaries[i].id,
                boundaries[i].enabled, boundaries[i].ini->id, boundaries[i].end->id,
                boundaries[i].energy, boundaries[i].arclength, boundaries[i].t_ext,
                boundaries[i].candidate);
    }

    // Store grains info
    fprintf(output_file, "#GRAIN ID ENABLED NUCLEATED ORIENT SE AREA dAdt VLEN VERTICES\n");
    for(int i = 0; i < n_grains; i++) {
        fprintf(output_file, "%d %d %d %.16f %.16f %.16f %.16f %d ",
            grains[i].id, grains[i].enabled, grains[i].type,
            grains[i].orientation, grains[i].SE, grains[i].area,
            grains[i].dAdt, grains[i].vlen);
        for(int j = 0; j < MAX_VRT_PER_GRN; j++) {
            if(j < grains[i].vlen) {
                fprintf(output_file, "%d ", grains[i].vertices[j]->id);
            } else if(j == MAX_VRT_PER_GRN - 1)
                fprintf(output_file, "%d", -1);
            else {
                fprintf(output_file, "%d ", -1);
            }
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);
    return true;
}

#endif // IO_H