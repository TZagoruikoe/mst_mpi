#include "defs.h"
#include <vector>
#include <limits>
#include <iostream>
#include <map>
#include <set>
#include <cassert>
#include <deque>

#ifdef USE_MPI
#include <mpi.h>
#include <omp.h>

typedef std::vector<std::vector<edge_id_t > > result_t;
result_t really_result;

typedef vertex_id_t component_id_t;

typedef struct {
    double weight;
    int edge;
    int index;
} weight_edge_id_t;

void for_custom_operation(void* inputBuffer, void* outputBuffer, int* len, MPI_Datatype* datatype) {
    weight_edge_id_t* input = (weight_edge_id_t*)inputBuffer;
    weight_edge_id_t* output = (weight_edge_id_t*)outputBuffer;
 
    for(int i = 0; i < *len; i++) {
        if(input[i].weight < output[i].weight || input[i].weight == output[i].weight && 
           input[i].index < output[i].index) {
            output[i].weight = input[i].weight;
            output[i].edge = input[i].edge;
            output[i].index = input[i].index;
        }
    }
}

//Copy from example mst_reference_mpi.cpp
edge_id_t edge_to_global(edge_id_t edge, graph_t *G) {
    int rank = G->rank;
    int size = G->nproc;
    edge_id_t g_edge = 0;
    for(int i = 0; i < rank && i < size; ++i)
    {
        g_edge += G->num_edges_of_any_process[i];
    }
    return (g_edge + edge);
}

//Copy from example mst_reference_mpi.cpp
extern "C" void init_mst(graph_t *G) { 
    G->num_edges_of_any_process = (edge_id_t*) malloc (G->nproc * sizeof(edge_id_t));
    edge_id_t * edges_to_send = (edge_id_t*) malloc (G->nproc * sizeof(edge_id_t));
    
    for(int i = 0; i < G->nproc; ++i)
	edges_to_send[i] = G->local_m;
    
    MPI_Alltoall(edges_to_send, 1, MPI_UINT64_T, G->num_edges_of_any_process, 1, MPI_UINT64_T, MPI_COMM_WORLD);
    if(edges_to_send) free(edges_to_send); 
}   

extern "C" void* MST(graph_t *G) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Datatype mpi_struct_w_e_i;
    int mpi_length[3] = {1, 1, 1};

    MPI_Aint displacements[3];
    weight_edge_id_t dummy_w_e_i;
    MPI_Aint base_address;

    MPI_Get_address(&dummy_w_e_i, &base_address);
    MPI_Get_address(&dummy_w_e_i.weight, &displacements[0]);
    MPI_Get_address(&dummy_w_e_i.edge, &displacements[1]);
    MPI_Get_address(&dummy_w_e_i.index, &displacements[2]);

    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);

    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
    MPI_Type_create_struct(3, mpi_length, displacements, types, &mpi_struct_w_e_i);
    MPI_Type_commit(&mpi_struct_w_e_i);

    MPI_Op operation;
    MPI_Op_create(&for_custom_operation, 1, &operation);

    vertex_id_t local_n = G->local_n;

    std::set<edge_id_t> result{};
    bool changed = true;

    std::map<vertex_id_t, component_id_t> components;
    std::map<component_id_t, weight_edge_id_t> comp_transform;
    std::map<component_id_t, component_id_t> shadow_comp_transform;

    for (vertex_id_t i = 0; i < local_n; i++) {
        vertex_id_t i_global = VERTEX_TO_GLOBAL(i, G->n, size, rank);
        components[i_global] = i_global;
    }

    for (vertex_id_t u_local = 0; u_local < local_n; u_local++) {
        for (edge_id_t e = G->rowsIndices[u_local]; e < G->rowsIndices[u_local + 1]; e++) {
            vertex_id_t v_global = G->endV[e];
            if (VERTEX_OWNER(v_global, G->n, size) != rank) {
                components[v_global] = v_global;
            }
        }
    }

    while(changed) {
        changed = false;
        comp_transform.clear();
        shadow_comp_transform.clear();

        for (vertex_id_t u_local = 0; u_local < local_n; u_local++) {
            vertex_id_t u_global = VERTEX_TO_GLOBAL(u_local, G->n, size, rank);
            vertex_id_t u_component = components[u_global];
            for (edge_id_t e = G->rowsIndices[u_local]; e < G->rowsIndices[u_local + 1]; e++) {
                vertex_id_t v_global = G->endV[e];
                vertex_id_t v_component = components[v_global];
                if (u_component != v_component && (comp_transform.find(u_component) == comp_transform.end() || G->weights[e] < comp_transform[u_component].weight ||
                    G->weights[e] == comp_transform[u_component].weight && v_component < comp_transform[u_component].index)) {
                    comp_transform[u_component] = {G->weights[e], edge_to_global(e, G), v_component};
                }
            }
        }

        for (auto it = components.begin(); it != components.end(); ++it) {
            if (comp_transform.find(components[it->first]) != comp_transform.end()) {
                shadow_comp_transform[components[it->first]] = comp_transform[components[it->first]].index;
            }
            else {
                shadow_comp_transform[components[it->first]] = -1;
            }
        }

        int fragment_size = G->n / size;
        std::vector<component_id_t> comp_transform_keys;

        for (auto it = comp_transform.begin(); it != comp_transform.end(); ++it) {
            comp_transform_keys.push_back(it->first);
        }

        for (int frag_id = G->n / fragment_size - (G->n % fragment_size == 0); frag_id >= 0; frag_id--) {
            std::vector<weight_edge_id_t> element_w_e_i(fragment_size);

#pragma omp parallel 
{
    #pragma omp for
            for (int i = 0; i < fragment_size; i++) {
                element_w_e_i[i].weight = std::numeric_limits<double>::max();
                element_w_e_i[i].index = -1;
            }

    #pragma omp for
            for (auto it = comp_transform_keys.begin(); it != comp_transform_keys.end(); ++it) {
                if (frag_id * fragment_size <= *it && *it < (frag_id + 1) * fragment_size) {
                    element_w_e_i[*it - frag_id * fragment_size] = comp_transform[*it];
                } 
            }
}

            MPI_Allreduce(MPI_IN_PLACE, element_w_e_i.data(), fragment_size, mpi_struct_w_e_i, operation, MPI_COMM_WORLD);

            std::map<component_id_t, component_id_t> fragment_transform;
            std::set<component_id_t> missing_index;
            std::vector<int> missing_index_value;

            for(component_id_t i = 0; i < fragment_size; i++) {
                if (element_w_e_i[i].index >= (frag_id + 1) * fragment_size) {
                    missing_index.insert(element_w_e_i[i].index);
                }
                fragment_transform[frag_id * fragment_size + i] = element_w_e_i[i].index;
            }

            for(component_id_t i : missing_index) {
                if (shadow_comp_transform.find(i) != shadow_comp_transform.end()) {
                    missing_index_value.push_back(shadow_comp_transform[i]);
                }
                else {
                    missing_index_value.push_back(-1);
                }
            }

            MPI_Allreduce(MPI_IN_PLACE, missing_index_value.data(), missing_index_value.size(), MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            int k = 0;

            for(component_id_t i : missing_index) {
                fragment_transform[i] = missing_index_value[k];
                k++;
            }

            for(component_id_t i = frag_id * fragment_size; i < (frag_id + 1) * fragment_size; i++) {
                if (fragment_transform[i] != -1) {
                    if (fragment_transform[i] < i || fragment_transform[fragment_transform[i]] != i) {
                        if (rank == 0) {
                            result.insert(element_w_e_i[i - frag_id * fragment_size].edge);
                        }
                        
                        changed = true;
                    } 
                }
            }

            std::map<component_id_t, std::set<component_id_t> > traversing_graph;

            for(auto it = fragment_transform.begin(); it != fragment_transform.end(); ++it) {
                if (fragment_transform[it->first] != -1) {
                    traversing_graph[it->first].insert(fragment_transform[it->first]);
                    traversing_graph[fragment_transform[it->first]].insert(it->first);
                }
            }

            for(auto it = traversing_graph.begin(); it != traversing_graph.end(); ++it) {
                fragment_transform[it->first] = it->first;
            }

            for (auto it = traversing_graph.begin(); it != traversing_graph.end(); ++it) {
                if (fragment_transform[it->first] == it->first) {
                    std::vector<component_id_t> queue;
                    std::vector<component_id_t> clone_queue;
                    queue.push_back(it->first);
                    while(!queue.empty()) {
                        int queue_size = queue.size();
#pragma omp parallel
{
                        std::vector<component_id_t> local_queue;
    #pragma omp for nowait
                        for (int j = 0; j < queue_size; j++) {
                            component_id_t cur;
                            cur = queue[j];
                            for(component_id_t tmp : traversing_graph[cur]) {
                                if(fragment_transform[tmp] != it->first) {
                                    fragment_transform[tmp] = it->first;
                                    local_queue.push_back(tmp);
                                }
                            }
                        }
    #pragma omp critical
    {
                        clone_queue.insert(clone_queue.end(), local_queue.begin(), local_queue.end());
    }
}
                        queue.clear();
                        std::swap(queue, clone_queue);
                    }
                }
            }

            for (auto it = shadow_comp_transform.begin(); it != shadow_comp_transform.end(); ++it) {
                if (fragment_transform.find(it->first) != fragment_transform.end()) {
                    shadow_comp_transform[it->first] = fragment_transform[it->first];
                }
                else if (it->first >= (frag_id + 1) * fragment_size && fragment_transform.find(it->second) != fragment_transform.end()) {
                    shadow_comp_transform[it->first] = fragment_transform[it->second];
                }
            }
        }  

        for (auto it = components.begin(); it != components.end(); ++it) {
            if (shadow_comp_transform.find(components[it->first]) != shadow_comp_transform.end()) {
                components[it->first] = shadow_comp_transform[components[it->first]];
            }
        }
    }

    MPI_Op_free(&operation);
    MPI_Type_free(&mpi_struct_w_e_i);

    std::vector<edge_id_t> tmp(result.begin(), result.end());
    really_result.clear();
    really_result.push_back(tmp);

    return &really_result;
}

//Copy from example mst_reference.cpp
extern "C" void convert_to_output(graph_t *G, void* result, forest_t *trees_output) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        result_t &trees_mst = *reinterpret_cast<result_t*>(result);
        trees_output->p_edge_list = (edge_id_t *)malloc(trees_mst.size()*2 * sizeof(edge_id_t));
        edge_id_t number_of_edges = 0;
        for (vertex_id_t i = 0; i < trees_mst.size(); i++) number_of_edges += trees_mst[i].size();
        trees_output->edge_id = (edge_id_t *)malloc(number_of_edges * sizeof(edge_id_t));
        trees_output->p_edge_list[0] = 0;
        trees_output->p_edge_list[1] = trees_mst[0].size();
        for (vertex_id_t i = 1; i < trees_mst.size(); i++) {
            trees_output->p_edge_list[2*i] = trees_output->p_edge_list[2*i-1];
            trees_output->p_edge_list[2*i +1] = trees_output->p_edge_list[2*i-1] + trees_mst[i].size();
        }
    
        int k = 0;
        for (vertex_id_t i = 0; i < trees_mst.size(); i++) {
            for (edge_id_t j = 0; j < trees_mst[i].size(); j++) {
                trees_output->edge_id[k] = trees_mst[i][j];
                k++;
            }
        }
        trees_output->numTrees = trees_mst.size();
        trees_output->numEdges = number_of_edges;
    }
}

extern "C" void finalize_mst(graph_t *G)
{
}
#else 

/* Write your own shared-memory MST implementation */

extern "C" void init_mst(graph_t *G)
{   
}   

extern "C" void* MST(graph_t *G)
{
    return 0;
}

extern "C" void convert_to_output(graph_t *G, void* result, forest_t *trees_output)
{
}

extern "C" void finalize_mst(graph_t *G)
{
}

#endif
