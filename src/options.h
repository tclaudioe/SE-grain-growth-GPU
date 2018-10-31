#ifndef OPTIONS_H
#define OPTIONS_H

#include <stdio.h>
#include "geometry.h"

/**
 * Load Options and Program Help Module.
 */

#define MAX_BUFFER_SIZE 1000
#define MAX_VAR_SIZE 50
#define MAX_FNAME_SIZE 256

struct options {
    /* Temporal vars */
    double dt;                   // Delta t
    double dtau;                 // Delta t for nucleation
    int nucleation_gap;          // Ratio between dtau and dt
    double tmax;                 // Max simulation time
    int nmax;                    // Max iterations
    int snap_gap;                // Save each snap_gap
    double leave_percent;        // Max percentage of grains
    /* Structure vars */
    int n_grains;                               // Number of initial grains
    char vertex_fname[MAX_FNAME_SIZE];          // Filenames of initial condition
    char boundary_fname[MAX_FNAME_SIZE];        //
    char orientation_fname[MAX_FNAME_SIZE];     //
    char SE_fname[MAX_FNAME_SIZE];
    double GB_eps;                      // Energy epsilon for GB energy function
    double GB_scaling;                  // Scale the GB energy by some factor
    double SE_eps;                      // Scale the SE by some epsilon
    double SE_min;                      // Maximum value of SE
    double SE_max;                      // Minimum value of SE
    double nucleation_scaling;          // Portion of boundary to assign lengths of grain
    double domain_bound;                // Domain bounds [0, domain_bound]^2
    bool do_nucleation;                 // Perform nucleation algorithm
    /* MC vars */
    double MC_k;                        // Monte Carlo coef in exp(-coef*DeltaE)
    /* System vars */
    int n_blocks;
    int n_threads;
    int debug_lvl;
    /* Output vars */
    char output_folder[MAX_FNAME_SIZE];
};

extern options opt;


/**
 * Remove the last character of a string.
 * @param str String to be modified.
 */
void fix_string(char* str) {
    int len = strlen(str);
    str[len-1] = '\0';
}

/**
 * Load configuration file from the path given by argv.
 * Must search first the flag -c to properly load.
 *
 * @param  argc Number of program parameters
 * @param  argv Parameters
 * @param  opt  Options struct
 * @return      True if successfully loaded, False otherwise
 */
bool load_config_file(int argc, const char* argv[], options *opt) {
    // Set default values
    opt->tmax = -1;
    opt->nmax = -1;
    opt->GB_eps = 0.02;
    opt->GB_scaling = 1.0;
    opt->SE_eps = 1.0;
    opt->SE_min = 999;
    opt->SE_max = 999;
    opt->MC_k = 1.0;
    opt->nucleation_scaling = 0.9;
    opt->domain_bound = 1.0;
    opt->do_nucleation = true;
    opt->nucleation_gap = 1;
    opt->debug_lvl = 0;
    opt->output_folder[0] = '\0';
    opt->leave_percent = -1;

    for(int i = 0; i < argc; i++) {
        if(!strcmp(argv[i], "-c") && i+1 < argc) {
            printf("Loading %s ...\n", argv[i+1]);
            FILE *file = fopen(argv[i+1], "r");
            if(file == NULL) {
                perror(argv[i+1]);
                return false;
            }
            // Holds whole line
            char buffer[MAX_BUFFER_SIZE];
            // Holds the variable name
            char var[MAX_VAR_SIZE];
            while(fgets(buffer, MAX_BUFFER_SIZE, file)) {
                if(buffer[0] == '#' || buffer[0] == '[') { continue; }
                char* pch = strchr(buffer, '=');
                if(pch != NULL) {
                    strncpy(var, buffer,  pch - buffer);
                    var[pch-buffer] = '\0';
                    //printf("%s\n", var);
                    if(!strcmp(var, "dt")) {
                        opt->dt = atof(pch+1);
                    } else if(!strcmp(var, "dtau")) {
                        opt->dtau = atof(pch+1);
                    } else if(!strcmp(var, "tmax")) {
                        opt->tmax = atof(pch+1);
                    } else if(!strcmp(var, "nmax")) {
                        opt->nmax = atoi(pch+1);
                    } else if(!strcmp(var, "snap_gap")) {
                        opt->snap_gap = atoi(pch+1);
                    } else if(!strcmp(var, "leave_percent")) {
                        opt->leave_percent = atof(pch+1);
                    } else if(!strcmp(var, "n_grains")) {
                        opt->n_grains = atoi(pch+1);
                    } else if(!strcmp(var, "vertex_fname")) {
                        strcpy(opt->vertex_fname, pch+1);
                        fix_string(opt->vertex_fname);
                    } else if(!strcmp(var, "boundary_fname")) {
                        strcpy(opt->boundary_fname, pch+1);
                        fix_string(opt->boundary_fname);
                    } else if(!strcmp(var, "orientation_fname")) {
                        strcpy(opt->orientation_fname, pch+1);
                        fix_string(opt->orientation_fname);
                    } else if(!strcmp(var, "SE_fname")) {
                        strcpy(opt->SE_fname, pch+1);
                        fix_string(opt->SE_fname);
                    } else if(!strcmp(var, "n_blocks")) {
                        opt->n_blocks = atoi(pch+1);
                    } else if(!strcmp(var, "n_threads")) {
                        opt->n_threads = atoi(pch+1);
                    } else if(!strcmp(var, "debug_lvl")) {
                        opt->debug_lvl = atoi(pch+1);
                    } else if(!strcmp(var, "folder")) {
                        strcpy(opt->output_folder, pch+1);
                        fix_string(opt->output_folder);
                    } else if(!strcmp(var, "GB_eps")) {
                        opt->GB_eps = atof(pch+1);
                    } else if(!strcmp(var, "GB_scaling")) {
                        opt->GB_scaling = atof(pch+1);
                    } else if(!strcmp(var, "SE_eps")) {
                        opt->SE_eps = atof(pch+1);
                    } else if(!strcmp(var, "SE_min")) {
                        opt->SE_min = atof(pch+1);
                    } else if(!strcmp(var, "SE_max")) {
                        opt->SE_max = atof(pch+1);
                    } else if(!strcmp(var, "MC_k")) {
                        opt->MC_k = atof(pch+1);
                    } else if(!strcmp(var, "nucleation_scaling")) {
                        opt->nucleation_scaling = atof(pch+1);
                    } else if(!strcmp(var, "domain_bound")) {
                        opt->domain_bound = atof(pch+1);
                    } else if(!strcmp(var, "do_nucleation")) {
                        if(!strcmp(pch+1, "true\n")) {
                            opt->do_nucleation = true;
                        } else {
                            opt->do_nucleation = false;
                        }
                    }
                }
            }
            fclose(file);
            break;
        }
    }

    //
    opt->nucleation_gap = int(opt->dtau / opt->dt);
    if(opt->tmax > 0 && opt->nmax == -1) {
        printf("Running until t = %.16f\n", opt->tmax);
    } else if(opt->tmax == -1 && opt->nmax > 0) {
        printf("Running until iteration #%d\n", opt->nmax);
    } else {
        opt->tmax = 1.0;
        opt->nmax = -1;
        printf("Running until t = %.16f\n", opt->tmax);
    }
    printf("Grain Boundary Energy Info:\n");
    printf("\tUsing GB epsilon: %.16f\n", opt->GB_eps);
    printf("\tGB energy scaling: %.16f\n", opt->GB_scaling);
    printf("Stored Energy Info:\n");
    printf("\tSE scaling: %.16f\n", opt->SE_eps);
    if(opt->SE_min == 999 && opt->SE_max == 999) {
        printf("\tMin and max values of SE will be obtained from data\n");
    } else if(opt->SE_min == opt->SE_max) {
        printf("\tMin SE = Max SE: %.16f\n", opt->SE_max);
    } else {
        printf("\tMin SE: %.16f\n\tMax SE: %.16f\n", opt->SE_min, opt->SE_max);
    }
    printf("\tMC k: %.16f\n", opt->MC_k);

    if(opt->do_nucleation) {
        printf("Perform nucleation: True\n");
        printf("\tNucleating each %d steps\n", opt->nucleation_gap);
        printf("\tNucleation scaling: %.16f\n", opt->nucleation_scaling);
    } else {
        printf("Perform nucleation: False\n");
    }
    printf("Domain bound [0 x %.2f][0 x %.2f]\n", opt->domain_bound, opt->domain_bound);
    printf("Configuration file loaded.\n");
    return true;
}

#endif /* OPTIONS_H */