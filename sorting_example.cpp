#include <random>
#include <vector>

#include "JanusSort.hpp"

#define PRINT_ROOT(msg) if (rank == 0) std::cout << msg << std::endl;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  double start, finish, loc_elapsed, elapsed;

  // Crear elementos de entrada aleatorios
  PRINT_ROOT("Crear elementos de entrada aleatorios");
  std::mt19937_64 generator;
  int data_seed = 3469931 + rank;
  generator.seed(data_seed);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::vector<double> data;
  int num_elementos = atoi(argv[1]);
  PRINT_ROOT("Datos a ordenar JANUS SORT");
  for (int i = 0; i < num_elementos; ++i) {
    data.push_back(dist(generator));
    //PRINT_ROOT(data[i])
  }
  {
    /* JanusSort */

    // Sort data descending
    auto data1 = data;

    //Experimento 1
    MPI_Barrier(comm);
    start = MPI_Wtime();
    PRINT_ROOT("Algoritmo de ordenamiento JanusSort con MPI_Comm. " <<"RBC::Communicators usados internamente.");
    JanusSort::sort(comm, data1, MPI_DOUBLE, std::greater<double>());
    PRINT_ROOT("Ordenación terminada");
    finish = MPI_Wtime();
    loc_elapsed = finish-start;
    MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm); 
    PRINT_ROOT("Elapsed time = %e\n"<<elapsed);
    
    //Experimento 2
    MPI_Barrier(comm);
    start = MPI_Wtime();
    PRINT_ROOT("Algoritmo de ordenamiento JanusSort con RBC::Comm.");
    RBC::Comm rcomm;
    RBC::Create_Comm_from_MPI(comm, &rcomm);
    auto data2 = data;
    JanusSort::sort(rcomm, data2, MPI_DOUBLE, std::greater<double>());
    finish = MPI_Wtime();
    loc_elapsed = finish-start;
    MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm); 
    PRINT_ROOT("Elapsed time = %e\n"<<elapsed);
    PRINT_ROOT("Ordenación terminada");

    //Experimento 3
    MPI_Barrier(comm);
    start = MPI_Wtime();
    PRINT_ROOT("Algoritmo de ordenamiento JanusSort con RBC::Comm. " << "MPI communicators y MPI collectives son usados.");
    RBC::Comm rcomm1;
    RBC::Create_Comm_from_MPI(comm, &rcomm1, true, true);
    auto data3 = data;
    JanusSort::sort(rcomm1, data3, MPI_DOUBLE);
    finish = MPI_Wtime();
    loc_elapsed = finish-start;
    MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm); 
    PRINT_ROOT("Elapsed time = %e\n"<<elapsed);
    PRINT_ROOT("Ordenación terminada");
    
  }

  // Finalize the MPI environment
  MPI_Finalize();
}
