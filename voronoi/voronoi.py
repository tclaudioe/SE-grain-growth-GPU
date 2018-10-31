# Modified from the code of Alejandro Sazo.
# This program generates synthetic Voronoi data that wraps arround the borders.


from scipy.spatial import Voronoi
import numpy as np
import sys

VERTEX_OUTPUT_FORMAT = "%.16f %.16f\n"


def get_voronoi_data(N, resolution=2**32):
    # | Repeat until a suitable combination of vertices is reached.
    attempt = 0
    while 1:
        attempt += 1
        sample = np.random.random((N, 2))

        # Copy original sample 8 quadrants around the main one
        # to create a bigger sample
        big_sample = np.concatenate((
                                    sample,
                                    sample + [-1., -1.],
                                    sample + [0., -1.],
                                    sample + [1., -1.],
                                    sample + [1., 0.],
                                    sample + [1., 1.],
                                    sample + [0., 1.],
                                    sample + [-1., 1.],
                                    sample + [-1., 0.]
                                    ), axis=0)
        # Apply Voronoi Diagram to sample
        vor = Voronoi(big_sample)

        # Get the true indexes that convert indexes of vertices to unique indexes of vertices
        # (Those that repeat themselves).
        vertices_dict = dict() # Transform from adjusted position to true_vertexes.
        true_indexes = []
        true_vertexes = []
        are_inside = [] # Registers it the vertices are inside or not.
        for i in range(vor.vertices.shape[0]):
            x = vor.vertices[i,0]
            y = vor.vertices[i,1]
            # Register if the vertex is inside the main quadrant or not.
            are_inside.append(x>=0.0 and x<1.0 and y>=0.0 and y<1.0)
            # Use the resolution to merge analogous vertex possitions.
            xx = (round(x*resolution)%resolution)/resolution
            yy = (round(y*resolution)%resolution)/resolution
            equiv = (xx, yy)
            if equiv not in vertices_dict:
                vertices_dict[equiv] = len(true_vertexes)
                true_vertexes.append(equiv)
            true_indexes.append(vertices_dict[equiv])

        # Calculate the true ridge vertices, deleting analogous.
        new_ridge_vertices = set()
        for (ini,end) in vor.ridge_vertices:
            # Discard the ridges that are totally outside of the main quadrant.
            if are_inside[ini] or are_inside[end]:
                # Check that both start and end are vertices inside the 9 quadrants and
                # that the true vertice at the start and at the end aren't the same, to prevent
                # the case of merging vertexes.
                if ini!=-1 and end!=-1 and true_indexes[ini] != true_indexes[end]:
                    # Sort the true indexes of the pointed vertices so they can be compared well.
                    combi= sorted([true_indexes[ini],true_indexes[end]])
                    if (combi[0],combi[1]) not in new_ridge_vertices:
                        new_ridge_vertices.add((combi[0],combi[1]))

        # For each true vertex count the number of ridges:
        count = [0 for i in range(len(true_vertexes))]
        for (a,b) in new_ridge_vertices:
            count[a] += 1
            count[b] += 1

        # The only adminisible values should be 0 and 3.
        failed = False
        for i in count:
            if i!=0 and i!=3:
                failed = True
                break

        if failed:
            print("Attempt %d failed!"%(attempt,))
            continue

        # Delete the vertices that doesn't have any ridge associated
        true_true_indexes = []
        true_true_vertexes = []
        for k in range(len(count)):
            if count[k] == 3:
                true_true_indexes.append(len(true_true_vertexes))
                true_true_vertexes.append(true_vertexes[k])
            else:
                true_true_indexes.append(-1)

        # Check that there aren't two vertices on the same position
        if len(set([VERTEX_OUTPUT_FORMAT%(x,y) for (x,y) in true_true_vertexes])) < len(true_true_vertexes):
            print("Attempt %d failed due to same position vertices!"%(attempt,))
            continue

        # Readjust the ridges.
        true_ridge_vertices = []
        for (a,b) in new_ridge_vertices:
            indxa = true_true_indexes[a]
            indxb = true_true_indexes[b]
            assert (indxa!=-1 and indxb !=-1)
            true_ridge_vertices.append((indxa,indxb))

        return true_true_vertexes, true_ridge_vertices


def help():
    print("Usage:\n\t"+sys.argv[0]+" <N> <seed> <file vertices> <file ridges> <file orientations> <file_SE> [--sedist DIST]")

def main():
    if len(sys.argv) != 7 and len(sys.argv) != 9:
        help()
        return

    N = int(sys.argv[1])
    seed = int(sys.argv[2])
    file_vertices = sys.argv[3]
    file_ridges = sys.argv[4]
    file_orientations = sys.argv[5]
    file_SE = sys.argv[6]

    if len(sys.argv) == 9:
        if "--sedist" == sys.argv[7]:
            dist = sys.argv[8]
        else:
            help()
    else:
        dist = 'uniform'

    np.random.seed(seed)
    vrtx, ridgs = get_voronoi_data(N)

    print("Vertices: %d"%(len(vrtx),))
    print("Ridges: %d"%(len(ridgs),))

    # Generate orientations
    orientations = np.random.rand(N)
    with open(file_orientations, "w") as ori_file:
        for alpha in orientations:
            ori_file.write("%.16f\n" % alpha)
    print("Orientations: %d" % (len(orientations)))
    # Generate stored energy using triangular probability dist
    if dist == 'triangular':
        print("Using triangular distribution for SE")
        SE = np.random.triangular(3,6,6,N)
    else:
        # Generate with uniform
        print("Using uniform distribution for SE")
        SE = np.random.rand(N)
    with open(file_SE, "w") as SE_file:
        for se in SE:
            SE_file.write("%.16f\n" % se)
    print("SE %s: %d" % (dist,len(SE)))
    # Print to files.
    with open(file_vertices,"w") as vrtx_file:
        for (x,y) in vrtx:
            vrtx_file.write(VERTEX_OUTPUT_FORMAT%(x,y))

    with open(file_ridges,"w") as ridg_file:
        for (a,b) in ridgs:
            ridg_file.write("%d %d\n"%(a,b))

if __name__ == '__main__':
    main()