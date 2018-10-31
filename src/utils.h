#ifndef UTILS_H
#define UTILS_H

#define N_BLKS 32
#define N_TRDS 256

#include <stdio.h>
#include "geometry.h"

/**
 * Utilities Module. Handling Error, Check data consistency.
 */

/*
    Error...
*/
inline void HandleError(cudaError_t err, const char *file, int line){
    // As from the book, "CUDA by example".
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HERR( err ) (HandleError( err, __FILE__, __LINE__ ))

/*
    Check faillure at using pointers to null address
*/
inline void CheckNull(void *ptr, const char *file, int line){
    if(ptr==NULL){
        printf("NULL pointer in %s at line %d\n", file, line);
        exit(EXIT_FAILURE);
    }
}
#define CNULL(ptr) (CheckNull((void *)(ptr),__FILE__, __LINE__ ))

inline void CheckError(const char *file, int line){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CERR(err) (CheckError(__FILE__, __LINE__ ))

/**
 * Print grain information to screen.
 *
 * @param grn Pointer to grain struct
 */
inline void print_grain(const grain *grn) {
    printf("Grain Info:\n\tId: %d\n", grn->id);
    printf("\tNumber of vertices: %d\n\tVertices: ", grn->vlen);
    for(int i = 0; i < grn->vlen; i++) {
        printf("%d ", grn->vertices[i]->id);
    }
    printf("\n\tArea: %.16f\n", grn->area);
    printf("\tOrientation: %.16f\n", grn->orientation);
    printf("\tStored Energy: %.16f\n", grn->SE);
    if(grn->enabled)
        printf("\tEnabled: true\n");
    else
        printf("\tEnabled: false\n");
    if(grn->type == NUCLEATED)
        printf("\tNucleated: true\n");
    else
        printf("\tNucleated: false\n");
    if(grn->candidate)
        printf("\tCandidate for removal: true\n");
    else
        printf("\tCandidate for removal: false\n");
}

/**
 * Host function to check data consistency
 *
 * @param vertices       Host vertices device array
 * @param n_vertices     Number of vertices
 * @param boundaries     Host boundaries device array
 * @param n_boundaries   Number of boundaries
 * @param grains         Host grains device array
 * @param n_grains       Number of grains
 * @return               True if the data is valid, false otherwise
 */
inline bool check_data(vertex* vertices, int n_vertices,
                       boundary *boundaries, int n_boundaries,
                       grain* grains, int n_grains) {
    bool is_valid = true;

    /* Check boundary stats */
    for(int i = 0; i < n_boundaries; i++) {
        boundary *bnd = &boundaries[i];
        if(!bnd->enabled) continue;
        if(bnd->ini == bnd->end) {
            printf("Boundary [ %d %d ] (id %d) starts and ends at the same junction!\n",
                   bnd->ini->id, bnd->end->id, bnd->id);
            is_valid = false;
        }
        if(!bnd->ini->enabled || !bnd->end->enabled) {
            printf("Boundary [ %d %d ] (id %d) has disabled vertex!\n",
                   bnd->ini->id, bnd->end->id, bnd->id);
            printf("vrt %d (%d), vrt %d (%d)\n", bnd->ini->id, bnd->ini->enabled,
                bnd->end->id, bnd->end->enabled);
            is_valid = false;
        }

        if(bnd->ini->boundaries[0] != bnd &&
           bnd->ini->boundaries[1] != bnd &&
           bnd->ini->boundaries[2] != bnd) {
            printf("Boundary [ %d %d ] (id %d) is not referenced by vertex %d\n",
                   bnd->ini->id, bnd->end->id, bnd->id, bnd->ini->id);
            printf("\tBoundaries associated from %d:\n\t", bnd->ini->id);
            for(int p = 0; p < 3; p++) {
                printf("[ %d %d ] ", bnd->ini->boundaries[p]->ini->id,
                   bnd->ini->boundaries[p]->end->id);
            }
            printf("\n");
            is_valid = false;
        }
        if(bnd->end->boundaries[0] != bnd &&
           bnd->end->boundaries[1] != bnd &&
           bnd->end->boundaries[2] != bnd) {
            printf("Boundary [ %d %d ] (id %d) is not referenced by vertex %d\n",
                   bnd->ini->id, bnd->end->id, bnd->id, bnd->end->id);
            printf("\tBoundaries associated from %d:\n\t", bnd->end->id);
            for(int p = 0; p < 3; p++) {
                printf("[ %d %d ] ", bnd->end->boundaries[p]->ini->id,
                   bnd->end->boundaries[p]->end->id);
            }
            printf("\n");
            is_valid = false;
        }
        // Check enabled neighbor boundaries
        if(!bnd->end->boundaries[0]->enabled ||
           !bnd->end->boundaries[1]->enabled ||
           !bnd->end->boundaries[2]->enabled) {
            printf("Boundary [ %d %d ] has a disabled boundary\n",
                bnd->ini->id, bnd->end->id);
            is_valid = false;
        }
        if(!bnd->ini->boundaries[0]->enabled ||
           !bnd->ini->boundaries[1]->enabled ||
           !bnd->ini->boundaries[2]->enabled) {
            printf("Boundary [ %d %d ] has a disabled boundary\n",
                bnd->ini->id, bnd->end->id);
            is_valid = false;
        }
        // Check bad grains in common
        grain *gic[2];
        grains_intersect(bnd->ini, bnd->end, gic);
        if(gic[0] == gic[1]) {
            printf("Boundary [ %d %d ] has repeated grain in common\n",
                bnd->ini->id, bnd->end->id);
            is_valid = false;
        }

        // Check if two grains have size 3
        if(gic[0]->vlen == 3 && gic[1]->vlen == 3) {
            printf("Error, Boundary [ %d %d ] has two grains with three sides.\nGrains %d, %d\n",
                bnd->ini->id, bnd->end->id, gic[0]->id, gic[1]->id);
            //is_valid = false;
        }
    }

    /* Check vertices stats */
    for(int i = 0; i < n_vertices; i++) {
        vertex *vrt = &vertices[i];
        if(!vrt->enabled) continue;
        // Check repeated boundaries
        if((vrt->boundaries[0] == vrt->boundaries[1]) ||
           (vrt->boundaries[1] == vrt->boundaries[2]) ||
           (vrt->boundaries[2] == vrt->boundaries[0])) {
            printf("Vertex %d has repated boundary pointers!\n", vrt->id);
            is_valid = false;
        }
        // Check repeated vertices neighbors
        int ids[3];
        for(int j = 0; j < 3; j++) {
            if(vrt->boundaries[j]->ini == vrt) {
                ids[j] = vrt->boundaries[j]->end->id;
            } else {
                ids[j] = vrt->boundaries[j]->ini->id;
            }
        }
        if(ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0]) {
            printf("Vertex %d has repeated neighbors! (by ids)\n", vrt->id);
        }
        // Check repeated grains
        if((vrt->grains[0] == vrt->grains[1]) ||
           (vrt->grains[1] == vrt->grains[2]) ||
           (vrt->grains[2] == vrt->grains[0])) {
            printf("Vertex %d has repeated grain!\n", vrt->id);
            is_valid = false;
        }
        // Check references to disabled boundaries
        for(int j = 0; j < 3; j++) {
            if(!vrt->boundaries[j]->enabled) {
                printf("Vertex %d points to disabled boundary!\n", vrt->id);
                is_valid = false;
            }
        }
        // Check references to disabled grains
        for(int j = 0; j < 3; j++) {
            if(!vrt->grains[j]->enabled) {
                printf("Vertex %d points to disabled grain!\n", vrt->id);
                is_valid = false;
            }
        }
        // Check bad boundary formation
        for(int j = 0; j < 3; j++) {
            if(vrt->boundaries[j]->ini == vrt->boundaries[j]->end) {
                printf("Vertex %d has malformed boundary\n", vrt->id);
                is_valid = false;
                if(vrt->boundaries[j]->ini == vrt) {
                    printf("Vertex %d points to itself\n", vrt->id);
                }
            }
        }
        // Check repeated neighboring vertices
        vertex *arr[3];
        for(int j = 0; j < 3; j++) {
            if(vrt->boundaries[j]->ini == vrt) {
                arr[j] = vrt->boundaries[j]->end;
            } else {
                arr[j] = vrt->boundaries[j]->ini;
            }
        }
        if((arr[0] == arr[1]) || (arr[1] == arr[2]) || (arr[2] == arr[0])) {
            printf("Vertex %d has repeated neighbors! (by pointers)\n", vrt->id);
            printf("\t[%d %d %d]\n", arr[0]->id, arr[1]->id, arr[2]->id);
            is_valid = false;
        }
        // Check inverse references
        for(int j = 0; j < 3; j++) {
            if(vrt->boundaries[j]->ini != vrt && vrt->boundaries[j]->end != vrt) {
                printf("Vertex %d is not referenced by one of its boundaries (%d)", vrt->id, j);
                printf("\n\tBoundary is [ %d %d ]\n", vrt->boundaries[j]->ini->id,
                    vrt->boundaries[j]->end->id);
                is_valid = false;
            }
        }
        // Check vertex in grain
        for(int j = 0; j < 3; j++) {
            bool not_found = true;
            for(int k = 0; k < vrt->grains[j]->vlen; k++) {
                if(vrt->grains[j]->vertices[k] == vrt) {
                    not_found = false;
                }
            }
            if(not_found) {
                is_valid = false;
                printf("Vertex %d is not in list of associated grains!\n", vrt->id);
                for(int l = 0; l < 3; l++) {
                    printf("\tAssociated vertices:\n\tgrn %d: ", vrt->grains[l]->id);
                    for(int k = 0; k < vrt->grains[l]->vlen; k++) {
                       printf("%d ", vrt->grains[l]->vertices[k]->id);
                    }
                    printf("\n");
                }
            }
        }
    }
    /* Check grains stats */
    double total_area = 0;
    for(int i = 0; i < n_grains; i++) {
        grain *grn = &grains[i];
        if(!grn->enabled) continue;
        // Check if grain is convex
        /*if(grain_get_convexity(grn)) {
            printf("Grain %d is not counterclock_wise!\n", grn->id);
            print_grain(grn);
            is_valid = false;
        }*/
        if(grn->area < 0) {
            printf("Grain %d has negative area: %.16f!\n", grn->id, grn->area);
            //is_valid = false;
        }
        total_area += grn->area;
        for(int j = 0; j < grn->vlen; j++) {
            if(!grn->vertices[j]->enabled) {
                printf("Grain %d has disabled vertex %d\n", grn->id, grn->vertices[j]->id);
                is_valid = false;
            }
        }

        for(int j = 0; j < grn->vlen; j++) {
            bool found_grain = false;
            for(int k = 0; k < 3; k++) {
                if(grn->vertices[j]->grains[k] == grn) {
                    found_grain = true;
                }
            }
            if(!found_grain) {
                printf("Grain %d is not referenced by vertex\n", grn->id);
                is_valid = false;
                for(int h = 0; h < grn->vlen; h++) {
                    printf("Grains from vertex %d: %d %d %d\n", grn->vertices[h]->id,
                        grn->vertices[h]->grains[0]->id, grn->vertices[h]->grains[1]->id,
                        grn->vertices[h]->grains[2]->id);
                }
                break;
            }
        }

        int cnt = 0;
        for(int j = 0; j < MAX_VRT_PER_GRN; j++) {
            if(grn->vertices[j] != NULL)
                cnt++;
        }
        if(cnt != grn->vlen) {
            printf("Grain %d number of vertices mismatch with declared vlen\n", grn->id);
            for(int j = 0; j < MAX_VRT_PER_GRN; j++) {
                if(grn->vertices[j] != NULL)
                    printf("%p ", grn->vertices[j]);
                else
                    printf("-1 ");
            }
            printf("\n");
            printf("%d versus vlen %d\n", cnt, grn->vlen);
            is_valid = false;
        }
        if(grn->candidate) {
            printf("Warning: grain %d has three sides and candidate!\n", grn->id);
            print_grain(grn);
        }
    }

    double eps = 1e-12;
    if(total_area >= 1.0+eps || total_area <= 1.0-eps || total_area < 0) {
        printf("Total grain area is incorrect: %.16f, should be 1.0\n", total_area);
        //is_valid = false;
    } else {
        printf("Total grain area is correct: %.16f, should be 1.0\n", total_area);
    }
    return is_valid;
}

/**
 * Print vertex information to screen.
 *
 * @param vrt Pointer to vertex struct
 */
__device__ __host__ inline void print_vertex(const vertex *vrt) {
    int b[3];
    for(int i = 0; i < 3; i++) {
        if(vrt->boundaries[i]->ini == vrt) {
            b[i] = vrt->boundaries[i]->end->id;
            //printf("%d ", vrt->boundaries[i]->end->id);
        } else {
            b[i] = vrt->boundaries[i]->ini->id;
            //printf("%d ", vrt->boundaries[i]->ini->id);
        }
    }
    printf("Vertex Info:\n\tId: %d\n\tPosition: %f %f\n\tVelocity: %f %f\n\tEnergy: %f\n\tEnabled: %d\n\tAssociated vertices: %d %d %d\n\tAssociated grains: %d %d %d\n",
     vrt->id, vrt->pos.x, vrt->pos.y, vrt->vel.x, vrt->vel.y, vrt->energy, vrt->enabled,
     b[0], b[1], b[2], vrt->grains[0]->id, vrt->grains[1]->id, vrt->grains[2]->id);
}

/**
 * Print boundary information to screen.
 *
 * @param bnd Pointer to boundary struct
 */
inline void print_boundary(const boundary *bnd) {
    printf("Boundary Info:\n\tVertices: %d %d\n", bnd->ini->id, bnd->end->id);
    printf("\tArclength: %f\n", bnd->arclength);
    printf("\tEnergy: %f\n", bnd->energy);
    if(bnd->enabled)
        printf("\tEnabled: true\n");
    else
        printf("\tEnabled: false\n");
    if(bnd->candidate)
        printf("\tCandidate: true\n");
    else
        printf("\tCandidate: false\n");
}

/**
 *  Print full boundary information according to flipping requirements
 *
 * @param bnd Pointer to boundary struct
 */
inline void print_full_boundary(const boundary *bnd) {
    printf("Boundary Info:\n\tVertices: %d %d\n", bnd->ini->id, bnd->end->id);
    print_vertex(bnd->ini);
    print_vertex(bnd->end);
    printf("Extinction time: %.16f\n", bnd->t_ext);
    printf("Connected vertices (by boundary):\n");
    for(int i = 0; i < 3; i++) {
        printf("\t[%d %d]\t", bnd->ini->boundaries[i]->ini->id, bnd->ini->boundaries[i]->end->id);
        if(bnd->ini->boundaries[i]->ini == bnd->ini) {
            print_vertex(bnd->ini->boundaries[i]->end);
        } else {
            print_vertex(bnd->ini->boundaries[i]->ini);
        }
    }
    printf("\n");
    for(int i = 0; i < 3; i++) {
        printf("\t[%d %d]\t", bnd->end->boundaries[i]->ini->id, bnd->end->boundaries[i]->end->id);
        if(bnd->end->boundaries[i]->ini == bnd->end) {
            print_vertex(bnd->end->boundaries[i]->end);
        } else {
            print_vertex(bnd->end->boundaries[i]->ini);
        }
    }
    printf("\n");
    printf("\tArclength: %f\n", bnd->arclength);
    printf("\tEnergy: %f\n", bnd->energy);
    if(bnd->enabled)
        printf("\tEnabled: true\n");
    else
        printf("\tEnabled: false\n");
}

/**
 * Print flipping scheme
 * @param bnd Pointer to boundary struct
 */
inline void print_flip_scheme(const boundary *bnd) {
    vertex *A, *B, *C, *D, *E, *F;
    A = bnd->ini; B = bnd->end;
    printf("Boundary [ %d %d ]\n", A->id, B->id);
    grain *gic[2], *gnic[2];
    grains_symdiff(A, B, gnic);
    grains_intersect(A, B, gic);
    printf("Grains in common:\n");
    print_grain(gic[0]);
    print_grain(gic[1]);
    printf("Grains not in common:\n");
    print_grain(gnic[0]);
    print_grain(gnic[1]);
    printf("%d %d %d %d\n", gic[0]->id, gic[1]->id, gnic[0]->id, gnic[1]->id);
    // Get the boundary A-C
    boundary *AC, *BE, *AD, *BF;
    int i, j, m, n;
    for(i = 0; i < 3; i++) {
        if(A->boundaries[i] == bnd) {
            m = (i + 2) % 3;
            i = (i + 1) % 3;
            AC = A->boundaries[i];
            AD = A->boundaries[m];
            break;
        }
    }
    // Get the boundary B-E
    for(j = 0; j < 3; j++) {
        if(B->boundaries[j] == bnd) {
            n = (j + 2) % 3;
            j = (j + 1) % 3;
            BE = B->boundaries[j];
            BF = B->boundaries[n];
            break;
        }
    }
    if(AC->ini == A) { C = AC->end;} else { C = AC->ini; }
    if(BE->ini == B) { E = BE->end;} else { E = BE->ini; }
    if(AD->ini == A) { D = AD->end;} else { D = AD->ini; }
    if(BF->ini == B) { F = BF->end;} else { F = BF->ini; }
    printf("Vrts: A=%d B=%d C=%d D=%d E=%d F=%d\n", A->id, B->id, C->id, D->id, E->id, F->id);
    grain *G1, *G2, *G3, *G4;
    printf("GIC\n");
    for(int k = 0; k < 2; k++) {
        printf("k=%d\n",k);
        int A_pos = grain_contains_vertex(gic[k], A);
        printf("%d, A=%d\n", A_pos, A->id);
        int C_pos = grain_contains_vertex(gic[k], C);
        printf("%d, C=%d\n", C_pos, C->id);
        int B_pos = grain_contains_vertex(gic[k], B);
        printf("%d, B=%d\n", B_pos, B->id);
        int E_pos = grain_contains_vertex(gic[k], E);
        printf("%d, E=%d\n", E_pos, E->id);
        if(A_pos >= 0 && C_pos >= 0) {
            G2 = gic[k];
            printf("Nice1\n");
        }
        if(B_pos >= 0 && E_pos >= 0) {
            G4 = gic[k];
            printf("Nice2\n");
        }
    }
    // Iterate over grains not in common
    printf("GNIC\n");
    for(int k = 0; k < 2; k++) {
        printf("k=%d\n",k);
        int A_pos = grain_contains_vertex(gnic[k], A);
        printf("%d, A=%d\n", A_pos, A->id);
        int B_pos = grain_contains_vertex(gnic[k], B);
        printf("%d, B=%d\n", B_pos, B->id);
        if(A_pos >= 0) {
            G3 = gnic[k];
        }
        if(B_pos >= 0) {
            G1 = gnic[k];
        }
    }

    // Simulate flipping
    printf("Boundary [%d %d] swaps to [%d %d]\n",
            A->boundaries[i]->ini->id, A->boundaries[i]->end->id,
            BE->ini->id, BE->end->id);
    printf("Boundary [%d %d] swaps to [%d %d]\n",
            B->boundaries[j]->ini->id, B->boundaries[j]->end->id,
            AC->ini->id, AC->end->id);
    if(AC->ini == A) {
        printf("Boundary [%d %d] swaps to [%d %d]\n",
            B->boundaries[j]->ini->id, B->boundaries[j]->end->id,
            B->id, AC->end->id);
    } else {
        printf("Boundary [%d %d] swaps to [%d %d]\n",
            B->boundaries[j]->ini->id, B->boundaries[j]->end->id,
            AC->ini->id, B->id);
    }
    if(BE->ini == B) {
        printf("Boundary [%d %d] swaps to [%d %d]\n",
            A->boundaries[i]->ini->id, A->boundaries[i]->end->id,
            A->id, BE->end->id);
    } else {
        printf("Boundary [%d %d] swaps to [%d %d]\n",
            A->boundaries[i]->ini->id, A->boundaries[i]->end->id,
            BE->ini->id, A->id);
    }




    printf("%d %d %d %d\n", G2->id, G4->id, G1->id, G3->id);
    printf("      %04d        %04d\n", C->id, F->id);
    printf("       \\    %04d    / \n", G2->id);
    printf("        \\          / \n");
    printf("%04d     %04d  %04d     %04d\n", G3->id, A->id, B->id, G1->id);
    printf("        /          \\ \n");
    printf("       /    %04d    \\ \n", G4->id);
    printf("      %04d        %04d\n", D->id, E->id);
}

/**
 * Print data structure
 * @param vertices     Vertices host array
 * @param n_vertices   Number of vertices
 * @param boundaries   Boundaries host array
 * @param n_boundaries Number of boundaries
 * @param grains       Grains host array
 * @param n_grains     Number of grains
 */
inline void print_data(vertex* vertices, int n_vertices,
                       boundary *boundaries, int n_boundaries,
                       grain* grains, int n_grains) {
    for(int i = 0; i < n_grains; i++) {
        print_grain(&grains[i]);
    }
    for(int i = 0; i < n_vertices; i++) {
        print_vertex(&vertices[i]);
    }
    for(int i = 0; i < n_boundaries; i++) {
        print_boundary(&boundaries[i]);
    }
}

#endif // UTILS_H