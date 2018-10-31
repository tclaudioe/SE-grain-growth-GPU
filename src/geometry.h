#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <assert.h>
#include <stdio.h>
#include "options.h"

// Maximum number of vertices per grain
#define MAX_VRT_PER_GRN 100
// Upper bound for extinction time
#define MAX_T_EXT 100000
// Matrix-free check for x-component or y-component
#define XCOMP 0
#define YCOMP 1

// Define type of change to be made to grains
typedef enum {ADD, REMOVE, REPLACE, NONE} fix_t;
// Define type of grain
typedef enum {COMMON, NUCLEATED} gtype_t;

struct vertex;
struct boundary;
struct grain;

// Modification type to rebuild grain list
struct mod_t {
    int grn_id;                 // Grain to be modified
    int vrt_id;                 // Vertex to be modified
    int vrt_prev_id;            // Reference vertex that must exists
    fix_t fix;                  // Type of modification
};

// Simple 2D vector
struct vector2 {
    double x;
    double y;
};

struct vertex {
    int id;                         // Unique id
    vector2 pos;                    // Position (x,y)
    vector2 vel;                    // Velocity (x,y)
    double energy;                  // Local energy
    bool enabled;                   // Is the vertex logically enabled?
    boundary *boundaries[3];        // Related boundaries
    grain *grains[3];               // Related grains
    mod_t mod[4];                   // Modifications of grains
    boundary *voted;                // Vote for boundary with lowest t_ext
    double nucl_ori;                // Random chosen orientation for nucleating grain
    double nucl_r;                  // Random length of grain along boundaries for nucleating grain
    double DeltaE;                  // Energy being added when nucleating here
    double nucleation_factor;       // Nucleation threshold Delta E^2
    bool nucleate;                  // Vertex marked for nucleation
    double vel_bnd_x;               // x-component of tangential velocity
    double vel_grn_x;               // x-component of stored energy velocity
    double vel_bnd_y;               // y-component of tangential velocity
    double vel_grn_y;               // y-component of stored energy velocity
    double angles[3];               // Dihedral angles
};

struct boundary {
    int id;                         // Unique id
    vertex *ini;                    // Pointer to initial vertex
    vertex *end;                    // Pointer to final vertex
    double energy;                  // Grain boundary energy
    double arclength;               // Arclength
    double t_ext;                   // Extinction time
    bool enabled;                   // Is the boundary logically enabled?
    bool candidate;                 // Is the boundary candidate to flip?
    int n_votes;                    // Number of votes received from its vertices
    int near_conflictive_bnd;       // Id of near boundary to flip
};

struct grain {
    int id;                                 // Unique id
    vertex *vertices[MAX_VRT_PER_GRN];      // List of points to vertices
    int vlen;                               // Number of vertices (sides)
    double area;                            // Area
    double dAdt;                            // Area rate of change
    double orientation;                     // Orientation [0, 2pi]
    double SE;                              // Stored energy
    bool enabled;                           // Is the grain logically enabled?
    bool candidate;                         // Is the grain candidate to be removed?
    bool fix;                               // Grain has to be fixed
    gtype_t type;                           // Type of grain (common or nucleated)
};


__device__ __host__ inline void grains_intersect(vertex *a, vertex *b, grain** grain_ids);
__device__ __host__ inline void adjust_origin_for_points(double *xpts, double *ypts, int plen, const double DOMAIN_BOUND);

/**
 * Finds the position of the maximum element of an unsorted array.
 *
 * @param  array to look for the maximum element
 * @param  size of array
 * @return
 */
__device__ __host__ inline int argmax(double* array, int size) {
    int pos = 0;
    for(int i = 0; i < size; i++) {
        if(array[pos] < array[i]) {
            pos = i;
        }
    }
    return pos;
}

/**
 * Finds the mininum of an unsorted array.
 *
 * @param  array to look for the maximum element
 * @param  size of array
 * @return
 */
__device__ __host__ inline double min(double* array, int size) {
    double value = array[0];
    for(int i = 0; i < size; i++) {
        if(value > array[i]) {
            value = array[i];
        }
    }
    return value;
}

/**
 * Apply mod to a coordinate in order to keep it inside the periodic domain.
 *
 * @param    a              Coordinate to be applied the mod
 * @param    DOMAIN_BOUND   Numerical domain bound
 * @return                  Coordinate inside the periodic domain
 */
__device__ __host__ inline double dom_mod(double a, const double DOMAIN_BOUND) {
    if (a < 0) return a + DOMAIN_BOUND;
    else if (a >= DOMAIN_BOUND) return a - DOMAIN_BOUND;
    return a;
}

/**
 * Gives the minor distance from a coordinate to another on the wrapping domain
 *
 * @param    dest           Destination coordinate
 * @param    orig           Origin coordinate
 * @param    DOMAIN_BOUND   Numerical domain bound
 * @return      Wrap distance between two coordinates
 */
__device__ __host__ inline double wrap_dist(double dest, double orig, const double DOMAIN_BOUND) {
    double delta = dest - orig;
    if(delta > +(DOMAIN_BOUND/2.0)) return (delta - DOMAIN_BOUND);
    if(delta < -(DOMAIN_BOUND/2.0)) return (DOMAIN_BOUND + delta);
    return delta;
}

/**
 * Compute the difference vector between two points, element wise
 *
 * @param    dest           Destination vector
 * @param    orig           Origin vector
 * @param    DOMAIN_BOUND   Numerical domain bound
 * @return      Wrap distance per coordinate
 */
__device__ __host__ inline vector2 vector2_delta_to(const vector2 dest, const vector2 orig, const double DOMAIN_BOUND) {
    vector2 res;
    res.x = wrap_dist(dest.x, orig.x, DOMAIN_BOUND);
    res.y = wrap_dist(dest.y, orig.y, DOMAIN_BOUND);
    return res;
}

/**
 * Vector sum element wise
 *
 * @param  a Input vector
 * @param  b Input vector
 * @return   Output vector a + b
 */
__device__ __host__ inline vector2 vector2_sum(const vector2 a, const vector2 b){
    vector2 res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
}

/**
 * Vector product with scalar
 *
 * @param  a Input vector
 * @param  b Input real scalar
 * @return   Output vector a * w
 */
__device__ __host__ inline vector2 vector2_prod(const vector2 a, const double w) {
    vector2 res;
    res.x = a.x * w;
    res.y = a.y * w;
    return res;
}

/**
 * Squared magnitude of vector
 *
 * @param  a Input vector
 * @return   Squared magnitude of vector
 */
__device__ __host__ inline double vector2_mag2(const vector2 a){
    double p = max(abs(a.x), abs(a.y));
    if(p == 0)
        return 0;
    double q = min(abs(a.x), abs(a.y));
    return (p*p) * (1.0 + (q/p)*(q/p));
}

/**
 * Magnitude of vector
 *
 * @param  a Input vector
 * @return   Magnitude of vector
 */
__device__ __host__ inline double vector2_mag(const vector2 a){
    double p = max(abs(a.x), abs(a.y));
    if(p == 0)
        return 0;
    double q = min(abs(a.x), abs(a.y));
    return p * sqrt(1.0 + (q/p)*(q/p));
}

/**
 * Dot product between two vectors
 *
 * @param  a Input vector
 * @param  b Input vector
 * @return   <a,b>
 */
__device__ __host__ inline double vector2_dot(const vector2 a, const vector2 b){
    return (a.x*b.x+a.y*b.y);
}

/**
 * Compute the integer part of a vector element wise
 *
 * @param  v   Input vector
 * @param  num Numerator
 * @param  den Denominator
 * @return     Integer part of vector
 */
__device__ __host__ inline vector2 vector2_portion(const vector2 v, int num, int den){
    vector2 res;
    res.x = (num*v.x)/den;
    res.y = (num*v.y)/den;
    return res;
}

/**
 * Compute the floating part of a vector element wise
 *
 * @param  v   Input vector
 * @param  prt Part to be computed
 * @return     Floating part
 */
__device__ __host__ inline vector2 vector2_float_portion(const vector2 v, double prt){
    vector2 res;
    if(prt<0.0) prt=0.0;
    if(prt>1.0) prt=1.0;
    res.x = (double)(prt*v.x);
    res.y = (double)(prt*v.y);
    return res;
}


/**
 * Compute a unitary vector using the angle and atan2 method
 *
 * @param  v Input vector
 * @return   Unitary vector
 */
__device__ __host__ inline vector2 vector2_unitary(const vector2 v){
    vector2 ret;
    double mag = vector2_mag(v);

    if(mag==0){
        ret.x = 0;
        ret.y = 0;
    } else {
        double angle = atan2(v.y,v.x);
        ret.x = cos(angle);
        ret.y = sin(angle);
    }
    return ret;
}


/**
 * Adjust a vector to the domain
 * @param  v            Input vector
 * @param  DOMAIN_BOUND Numerical domain bound
 * @return              Vector adjusted to periodic domain
 */
__device__ __host__ inline vector2 vector2_adjust(const vector2 v, const double DOMAIN_BOUND) {
    vector2 aux;
    aux.x = dom_mod(v.x, DOMAIN_BOUND);
    aux.y = dom_mod(v.y, DOMAIN_BOUND);
    return aux;
}


/**
 * Compute midpoint between two points
 *
 * @param  v1 Origin point
 * @param  v2 Destination point
 * @return    Midpoint adjusted to periodic domain
 */
__device__ __host__ inline vector2 vector2_mean(const vector2 v1, const vector2 v2, const double DOMAIN_BOUND) {
    vector2 v1_cpy = v1;
    vector2 delta = vector2_delta_to(v2, v1, DOMAIN_BOUND);
    v1_cpy.x += delta.x/2;
    v1_cpy.y += delta.y/2;
    return vector2_adjust(v1_cpy, DOMAIN_BOUND);
}


/**
 * Compute length of line segment given positions in a periodic domain.
 *
 * @param  ini            Initial point of line
 * @param  end            Initial point of line
 * @param  DOMAIN_BOUND   Numerical domain bound
 * @return                Length of line segment
 */
__device__ __host__ inline double compute_arclength(const vector2 ini, const vector2 end, const double DOMAIN_BOUND) {
    return vector2_mag(vector2_delta_to(ini, end, DOMAIN_BOUND));
}


/******************** VERTICES FUNCTIONS ********************/

__device__ __host__ inline void reset_mod_list(vertex *vrt);

/**
 * Init vertex struct with information of position and unique identifier.
 * For consistency, the associated grains and boundaries are initialized as null pointers.
 * Vertex energy also is initialized as zero. Enabled by default.
 *
 * @param vrt Pointer to vertex being created
 * @param pos Position of vertex in periodic domain
 * @param id  Unique identifier
 */
__device__ __host__ inline void init_vertex(vertex *vrt, const vector2 pos, int id) {
    vrt->id = id;
    vrt->pos = pos;
    vrt->vel.x = 0.0;
    vrt->vel.y = 0.0;
    vrt->energy = 0.0;
    vrt->enabled = true;
    vrt->nucleate = false;
    for(int i = 0; i < 3; i++) {
        vrt->grains[i] = NULL;
        vrt->boundaries[i] = NULL;
    }
    // Clean info for polling algorithm
    vrt->voted = NULL;
    reset_mod_list(vrt);
}

/**
 * Reset the list of modification metadata. Pointers to
 * vertex and previous vertex, type of fix and grain are null.
 *
 * @param vrt Pointer to vertex
 */
__device__ __host__ inline void reset_mod_list(vertex *vrt) {
    for(int i = 0; i < 4; i++) {
        vrt->mod[i].vrt_id = -1;
        vrt->mod[i].vrt_prev_id = -1;
        vrt->mod[i].grn_id = -1;
        vrt->mod[i].fix = NONE;
    }
}

/**
 * Add boundary to vertex. The boundary is placed at the first place in array
 * where a null pointer exists.
 *
 * @param vrt Pointer to vertex to be modified
 * @param bnd Pointer to boundary to be added
 */
__device__ __host__ inline void vertex_add_boundary(vertex *vrt, boundary *bnd) {
    if(vrt->boundaries[0] == NULL) {
        vrt->boundaries[0] = bnd;
    } else if(vrt->boundaries[1] == NULL) {
        vrt->boundaries[1] = bnd;
    } else if(vrt->boundaries[2] == NULL) {
        vrt->boundaries[2] = bnd;
    } else {
        printf("In %s: There are not free slots for assign boundary %d into vertex %d\n",
         __func__, bnd->id, vrt->id);
    }
}

/**
 * Add grain to vertex information. The reference are added so an internal
 * array points to each grain. The new reference is added in the next
 * null pointer.
 *
 * @param vrt Pointer to vertex
 * @param grn Grain reference
 */
__device__ __host__ inline void vertex_add_grain(vertex *vrt, grain *grn) {
    if(vrt->grains[0] == NULL) {
        vrt->grains[0] = grn;
    } else if(vrt->grains[1] == NULL) {
        vrt->grains[1] = grn;
    } else if(vrt->grains[2] == NULL) {
        vrt->grains[2] = grn;
    } else {
        printf("In %s: There are not free slots for assign grain %d into vertex %d\n",
            __func__, grn->id, vrt->id);
    }
}

/**
 * Invert the order of the boundaries of a vertex
 *
 * @param vrt Pointer to vertex struct
 */
__device__ __host__ inline void vertex_invert_boundary_order(vertex *vrt) {
    // The order can be reversed easily, just swapping two boundaries
    boundary *bnd = vrt->boundaries[1];
    vrt->boundaries[1] = vrt->boundaries[0];
    vrt->boundaries[0] = bnd;
}

/**
 * Set boundaries clockwise
 *
 * @param vrt             Vertex which boundaries will be sorted
 * @param  DOMAIN_BOUND   Numerical domain bound
 */
__device__ __host__ inline void vertex_set_boundaries_clockwise(vertex *vrt, const double DOMAIN_BOUND) {
    double angle[3];
    for(int i = 0; i < 3; i++) {
        vector2 delta;
        if(vrt->boundaries[i]->ini == vrt) {
            delta = vector2_delta_to(vrt->boundaries[i]->end->pos, vrt->pos, DOMAIN_BOUND);
        } else {
            delta = vector2_delta_to(vrt->boundaries[i]->ini->pos, vrt->pos, DOMAIN_BOUND);
        }
        angle[i] = atan2(delta.y, delta.x);
    }
    for(int i = 0; i < 2; i++) {
        for(int p = 1; p < 3-i; p++) {
            if(angle[p-1] < angle [p]) {
                double aux = angle[p-1];
                angle[p-1] = angle[p];
                angle[p] = aux;
                boundary *auxb = vrt->boundaries[p-1];
                vrt->boundaries[p-1] = vrt->boundaries[p];
                vrt->boundaries[p] = auxb;
            }
        }
    }
}

/**
 * Compute vertex energy as:
 * $E_i(t) = \sum_{j=0}^{3} \gamma_{i,j} \frac{||l(s,t)||}{2} + SE_{i,j}*\frac{A_{i,j}}{ns(g)}$
 * This function assumes that all the quantities (arclength, energy, areas and SE) are updated.
 * Notice that assuming boundary energies equal to 1 and stored energy 0 we recover
 * the classic vertex model.
 *
 * @param vrt Pointer to vertex struct
 * @return    Vertex energy
 */
__device__ __host__ inline double vertex_compute_energy(vertex *vrt) {
    double energy = 0.0;
    for(int i = 0; i < 3; i++) {
        energy += (vrt->boundaries[i]->arclength * vrt->boundaries[i]->energy * 0.5);
        energy += (vrt->grains[i]->SE * vrt->grains[i]->area / vrt->grains[i]->vlen);
    }
    return energy;
}
/******************** BOUNDARIES FUNCTIONS ********************/

/**
 * Init boundary struct with vertices information. Arclength is set to 0,
 * thus must be calculated before operating. By default the boundary is enabled.
 *
 * @param bnd Pointer to boundary being created
 * @param ini Pointer to vertex where parametrization starts
 * @param end Pointer to vertex where parametrization ends
 * @param id  Unique identifier
 */
__device__ __host__ inline void init_boundary(boundary *bnd, vertex *ini,
                                              vertex *end, int id) {
    bnd->ini = ini;
    vertex_add_boundary(ini, bnd);
    bnd->end = end;
    vertex_add_boundary(end, bnd);
    bnd->id = id;
    bnd->arclength = 0;
    bnd->t_ext = MAX_T_EXT;
    bnd->enabled = true;
    bnd->candidate = false;
    bnd->n_votes = 0;
    bnd->near_conflictive_bnd = -1;
}

/**
 * Check if boundary meets the requirement to be candidate for flipping.
 *
 * @param  bnd Pointer to boundary.
 * @param  dt  Simulation $\Delta t$.
 * @return     True if the boundary will flip, false otherwise.
 */
__device__ __host__ inline bool boundary_is_candidate(boundary *bnd, double dt) {
    if(bnd->t_ext >= 0 && bnd->t_ext <= dt)
        return true;
    return false;
}

/**
 * Compute boundary arclength $AL = ||l(s,t)||$ and stores it in boundary struct.
 *
 * @param  bnd Pointer to boundary struct
 * @param  DOMAIN_BOUND   Numerical domain bound
 * @return     Boundary arclength
 */
__device__ __host__ inline double boundary_compute_arclength(boundary *bnd, const double DOMAIN_BOUND) {
    return compute_arclength(bnd->ini->pos, bnd->end->pos, DOMAIN_BOUND);
}

/**
 * Fix orientation to correct domain [0, 2pi].
 *
 * @param  alpha Grain orientation as a random value between [0, 1]
 * @return       Orientation in [0, 2pi]
 */
__device__ __host__ inline double fix_orientation(const double alpha) {
    return alpha * 2 * M_PI;
}

/**
 * Grain boundary energy function:
 * $\gamma_{i,j}(\Delta \alpha) = 1 + \frac{\varepsilon}{2}(1-\cos^3(\Delta \alpha))$.
 *
 * @param  d_alpha Misorientation
 * @param  eps     Epsilon of energy function
 * @return         Boundary energy
 */
__device__ __host__ inline double boundary_energy_func(const double d_alpha, double eps) {
    return (1 + 0.5*eps*(1 - pow(cos(4*d_alpha), 3)));
}

/**
 * Compute boundary energy using some specified function.
 *
 * @param  bnd Pointer to boundary struct
 * @param  eps Epsilon of energy function
 * @return     Boundary energy
 */
__device__ __host__ inline double boundary_compute_energy(boundary *bnd, double eps) {
    grain *gic[2];
    grains_intersect(bnd->ini, bnd->end, gic);
    double d_alpha = gic[0]->orientation - gic[1]->orientation;
    return boundary_energy_func(d_alpha, eps);
}

/******************** GRAINS FUNCTIONS ********************/

__device__ __host__ inline void grain_clean_vertices(grain *grn);

/**
 * Init grain struct setting an unique identifier and 0 vertices.
 * A static array of MAX_VRT_PER_GRN pointers to vertices is created,
 * each pointer is NULL. By default the grain is enabled.
 *
 * @param grn       Pointer to grain being created
 * @param lengrains Counter of grains created, used for set id
 */
__device__ __host__ inline void init_grain(grain *grn, const int lengrains) {
    grn->id = lengrains;
    grain_clean_vertices(grn);
    grn->vlen = 0;
    grn->area = 0;
    grn->SE = 0;
    grn->orientation = 0;
    grn->enabled = true;
    grn->candidate = false;
    grn->fix = false;
    grn->type = COMMON;
}

/**
 * Init grain with old id and type NUCLEATED.
 *
 * @param grn Pointer to grain being nucleated.
 */
__device__ __host__ inline void nucleate_grain(grain *grn) {
    grain_clean_vertices(grn);
    grn->vlen = 0;
    grn->area = 0;
    grn->SE = 0;
    grn->orientation = 0;
    grn->enabled = true;
    grn->candidate = false;
    grn->fix = false;
    grn->type = NUCLEATED;
}

/**
 * Add vertex reference to grain. The counter of vertices increases.
 *
 * @param grn Pointer to grain struct
 * @param vrt Pointer to vertex struct
 */
__device__ __host__ inline void grain_add_vertex(grain *grn, vertex *vrt) {
    if(grn->vlen == MAX_VRT_PER_GRN) {
        printf("In %s: Maximum number of vertices reached for grain %d.\n", __func__, grn->id);
    }
    grn->vertices[grn->vlen] = vrt;
    grn->vlen++;
}

/**
 * Add vertex reference to a grain with respect to a certain vertex.
 *
 * @param grn      Pointer to grain struct
 * @param vrt      Pointer to vertex struct
 * @param vrt_prev Pointer to reference vertex struct
 */
__device__ __host__ inline void grain_add_vertex(grain *grn, vertex *vrt, vertex *vrt_prev) {
    int vlen = grn->vlen;
    int pos;
    // Check where the reference vertex is and add new vertex after it
    for(int i = 0; i < vlen; i++) {
        if(grn->vertices[i] == vrt_prev) {
            pos = i+1;
            break;
        }
    }
    // Fix vertex lists to make room
    for(int i = vlen; i > pos; i--) {
        grn->vertices[i] = grn->vertices[i-1];
    }
    grn->vertices[pos] = vrt;
    grn->vlen++;
}

/**
 * Remove a vertex from grain. The counter of vertices decreases.
 *
 * @param  grn Pointer to grain struct
 * @param  vrt Vertex to be removed
 */
__device__ __host__ inline void grain_remove_vertex(grain *grn, vertex *vrt) {
    for(int i = 0; i < grn->vlen; i++) {
        if(grn->vertices[i] == vrt) {
            for(int j = i; j < grn->vlen-1; j++) {
                grn->vertices[j] = grn->vertices[j+1];
            }
            grn->vlen--;
            grn->vertices[grn->vlen] = NULL;
            return;
        }
    }
    printf("In %s: Vertex %d was not found in grain %d.\n",
                __func__, vrt->id, grn->id);
}

/**
 * Clean vertex list. This method effectively set number of vertices to zero.
 *
 * @param grn Pointer to grain struct
 */
__device__ __host__ inline void grain_clean_vertices(grain *grn) {
    for(int i = 0; i < MAX_VRT_PER_GRN; i++) {
        grn->vertices[i] = NULL;
    }
}

/**
 * Replace a vertex by another wone.
 *
 * @param grn     Pointer to grain struct
 * @param old_vrt Vertex to be replaced
 * @param new_vrt Vertex replacing
 */
__device__ __host__ inline void grain_replace_vertex(grain *grn, vertex *old_vrt, vertex *new_vrt) {
    int vlen = grn->vlen;
    for(int i = 0; i < vlen; i++) {
        if(grn->vertices[i] == old_vrt) {
            grn->vertices[i] = new_vrt;
            return;
        }
    }
}

/**
 * Checks if the vertex pointer by vrt is in the grain list.
 * If it's not found, returns negative index.
 *
 * @param grn Pointer to grain struct
 * @param vrt Pointer to vertex struct
 */
__device__ __host__ inline int grain_contains_vertex(grain *grn, vertex *vrt) {
    for(int i = 0; i < grn->vlen; i++) {
        if(grn->vertices[i] == vrt) {
            return i;
        }
    }
    return -1;
}

/**
 * Check if grain contains a certain configuration of two consecutive vertices.
 * Return value depends on which is the order of the vertices.
 * If the configuration is not found, return -1.
 *
 * @param  grn Pointer to grain struct
 * @param  A   Pointer to vertex struct
 * @param  B   Pointer to vertex struct
 * @return     1 if the vertices are found in the order [A,B],
 *             2 if the vertices are found in the order [B,A],
 *             -1 otherwise
 */
__device__ __host__ inline int grain_contains_consecutive_vertices(grain *grn, vertex *A, vertex *B) {
    for(int i = 0; i < grn->vlen; i++) {
        if(grn->vertices[i] == A && grn->vertices[(i+1) % grn->vlen] == B) {
            return 1;
        }
        if(grn->vertices[i] == B && grn->vertices[(i+1) % grn->vlen] == A) {
            return 2;
        }
    }
    return -1;
}

/**
 * Check if a grain contains a certain boundary. This method does not ask for boundary id,
 * instead it looks for the vertices forming that boundary.
 *
 * @param  grn Pointer to grain struct
 * @param  bnd Pointer to boundary struct
 * @return     Same as @grain_contains_consecutive_vertices
 */
__device__ __host__ inline int grain_contains_boundary(grain *grn, boundary *bnd) {
    return grain_contains_consecutive_vertices(grn, bnd->ini, bnd->end);
}

/**
 * Compute area of a polygon given arrays of components x and y.
 *
 * @param  X              List of x positions
 * @param  Y              List of y positions
 * @param  vlen           Number of vertices in polygon
 * @param  DOMAIN_BOUND   Numerical domain bound
 * @return                Area of polygon using Green formula
 */
__device__ __host__ inline double compute_area(double *X, double *Y, int vlen, const double DOMAIN_BOUND) {
    double area = 0;
    adjust_origin_for_points(X, Y, vlen, DOMAIN_BOUND);
    for(int k = 0; k < vlen; k++) {
        area += X[k]*Y[(k+1) % vlen] - Y[k]*X[(k+1) % vlen];
    }
    area *= 0.5;
    return area;
}

/**
 * Compute rate of change of polygon area given arrays of components x and y
 * and components of velocities. The formula of dA/dt follows
 * the derivation with respect to time of Green formula.
 *
 * @param  X    List of x positions
 * @param  Y    List of y positions
 * @param  Vx   List of x velocities
 * @param  Vy   List of y velocities
 * @param  vlen Number of vertices in polygon
 * @return      dA/dt of polygon
 */
__device__ __host__ inline double compute_dAdt(double *X, double *Y, double *Vx, double *Vy, const int vlen, const double DOMAIN_BOUND) {
    double dAdt = 0;
    adjust_origin_for_points(X, Y, vlen, DOMAIN_BOUND);
    for(int k = 0; k < vlen; k++) {
        dAdt += (Vx[k]*Y[(k+1)%vlen]+X[k]*Vy[(k+1)%vlen]);
        dAdt -= (Vy[k]*X[(k+1)%vlen] + Y[k]*Vx[(k+1)%vlen]);
    }
    return dAdt;
}

/**
 * Wrapper for generic polygon area computation. The grain positions
 * of each vertex is arranged in a way that we can compute the area easily.
 *
 * @param  grn            Pointer to grain struct
 * @param  DOMAIN_BOUND   Numerical domain bound
 * @return                Grain area
 */
__device__ __host__ inline double grain_compute_area(grain *grn, const double DOMAIN_BOUND) {
    double X[MAX_VRT_PER_GRN], Y[MAX_VRT_PER_GRN];
    for(int i = 0; i < grn->vlen; i++) {
        X[i] = grn->vertices[i]->pos.x;
        Y[i] = grn->vertices[i]->pos.y;
    }
    return compute_area(X, Y, grn->vlen, DOMAIN_BOUND);
}

/**
 * Compute instant area rate of change dA/dt.
 *
 * @param  grn            Pointer to grain struct
 * @param  DOMAIN_BOUND   Numerical domain bound
 * @return                dA/dt
 */
__device__ __host__ inline double grain_compute_dAdt(grain *grn, const double DOMAIN_BOUND) {
    double X[MAX_VRT_PER_GRN], Y[MAX_VRT_PER_GRN];
    double Vx[MAX_VRT_PER_GRN], Vy[MAX_VRT_PER_GRN];
    for(int i = 0; i < grn->vlen; i++) {
        X[i] = grn->vertices[i]->pos.x;
        Vx[i] = grn->vertices[i]->vel.x;
        Y[i] = grn->vertices[i]->pos.y;
        Vy[i] = grn->vertices[i]->vel.y;
    }
    return compute_dAdt(X, Y, Vx, Vy, grn->vlen, DOMAIN_BOUND);
}

/**
 * Intersect two sets of grains from given vertices.
 * Each vertex has a list of grains of the form:
 * [g1, g2, g3], [g1, g2, g4]
 * The algorithm checks each grain in first list. If it is present in
 * the second list, it is added to grain_ids list. The grain list
 * it is expected to be size 2 to return [g1, g2]
 *
 * @param a         Vertex a
 * @param b         Vertex b
 * @param grain_ids Array of pointers to grains
 */
__device__ __host__ inline void grains_intersect(vertex *a, vertex *b, grain** grain_ids) {
    int k = 0;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            if(a->grains[i] == b->grains[j]) {
                grain_ids[k++] = a->grains[i];
                if(k == 2)
                    return;
            }
        }
    }
}

/**
 * Union of two sets of grains from given vertices.
 * Each vertex has a list of grains of the form:
 * [g1, g2, g3], [g1, g2, g4]
 * The union is initialized as all the grains of vertex a.
 * After that each grain of vertex b is checked,
 * if the grain is not present in list it is added and algorithm ends.
 * The return list is expected to be of size 4 to obtain [g1, g2, g3, g4]
 *
 * @param a         Vertex a
 * @param b         Vertex b
 * @param grain_ids Array of pointers to grains
 */
__device__ __host__ void grains_union(vertex *a, vertex *b, grain** grain_ids) {
    for(int i = 0; i < 3; i++) {
        grain_ids[i] = a->grains[i];
    }
    for(int i = 0; i < 3; i++) {
        if((b->grains[i] != grain_ids[0]) &&
           (b->grains[i] != grain_ids[1]) &&
           (b->grains[i] != grain_ids[2])) {
            grain_ids[3] = b->grains[i];
            break;
        }
    }
}

/**
 * Symmetric difference of grains lists, useful to find the grains
 * that are not shared between a boundary.
 * The return list is expected to be of size 2
 *
 * @param a         Vertex a
 * @param b         Vertex b
 * @param grain_ids Array of pointers to grains
 */
__device__ __host__ void grains_symdiff(vertex *a, vertex *b, grain** grain_ids) {
    grain *gic[2], *gun[4];
    grains_intersect(a, b, gic);
    grains_union(a, b, gun);
    int k = 0;
    for(int i = 0; i < 4; i++) {
        if((gun[i] != gic[0]) && (gun[i] != gic[1])) {
            grain_ids[k++] = gun[i];
            if(k == 2)
                return;
        }
    }
}

/**
 * Adjust the points on a curve to new coordinates with the origin on the first one.
 *
 * @param xpts           Array of x-coordinates
 * @param ypts           Array of y-coordinates
 * @param plen           Number of points in the curve
 * @param DOMAIN_BOUND   Numerical domain bound
 */
__device__ __host__ inline void adjust_origin_for_points(double *xpts, double *ypts, int plen, const double DOMAIN_BOUND){
    if(plen > 0){
        vector2 prev;
        prev.x = xpts[0]; prev.y = ypts[0];
        xpts[0] = 0; ypts[0] = 0;
        for(int i = 1; i < plen; i++){
            // Calculate the delta with the previous point
            vector2 current;
            current.x = xpts[i];
            current.y = ypts[i];
            vector2 delta = vector2_delta_to(current, prev, DOMAIN_BOUND);
            // Save the point before changing it.
            prev.x = xpts[i];
            prev.y=  ypts[i];
            // Update the current point to the new axis.
            vector2 adjusted_prev;
            adjusted_prev.x = xpts[i-1];
            adjusted_prev.y = ypts[i-1];
            current = vector2_sum(adjusted_prev, delta);
            xpts[i] = current.x;
            ypts[i] = current.y;
        }
    }
}

#endif // GEOMETRY_H