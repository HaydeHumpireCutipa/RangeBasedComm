#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "../../include/RBC/RBC.hpp"
#include "Constants.hpp"
#include "DataExchange.hpp"
#include "PivotSelection.hpp"
#include "QSInterval.hpp"
#include "SequentialSort.hpp"
#include "TbSplitter.hpp"

namespace JanusSort {

template <typename T>
class Sorter { //Algoritmo Quicksort
 public:

  Sorter(MPI_Datatype mpi_type, int seed, int64_t min_samples = 64,
         bool barriers = false, bool split_MPI_comm = false,
         bool use_MPI_collectives = false, bool add_pivot = false)
    : m_mpi_type(mpi_type),
      m_seed(seed),
      m_barriers(barriers),
      m_split_mpi_comm(split_MPI_comm),
      m_use_mpi_collectives(use_MPI_collectives),
      m_add_pivots(add_pivot),
      m_min_samples(min_samples) {
  }

  ~Sorter() { }

  void sort(MPI_Comm mpi_comm, std::vector<T>& data_vec, int64_t global_elements = -1) {
    sort(mpi_comm, data_vec, global_elements, std::less<T>());
  }


  template <class Compare>
  void sort(MPI_Comm mpi_comm, std::vector<T>& data, Compare&& comp, int64_t global_elements) {
    RBC::Comm comm;
    RBC::Create_Comm_from_MPI(mpi_comm, &comm, m_use_mpi_collectives, m_split_mpi_comm);
    MPI_Barrier(mpi_comm);
    sort_range(comm, data, std::forward<Compare>(comp), global_elements);
  }


  template <class Compare>
  void sort_range(RBC::Comm comm, std::vector<T>& data, Compare&& comp, int64_t global_elements) {
    assert(!comm.isEmpty());
    double total_start = getTime();
    // m_parent_comm = comm.GetMpiComm();
    this->m_data = &data;
    int size, rank;
    RBC::Comm_size(comm, &size);
    RBC::Comm_rank(comm, &rank);
    m_generator.seed(m_seed);
    m_sample_generator.seed(m_seed + rank);

    QSInterval_SQS<T> ival;
    assert(global_elements >= -1);
    if (global_elements == -1) {
      m_buffer = nullptr;
      //Los elementos locales y global_end  seran cambiados despues del intercambio de datos
      ival = QSInterval_SQS<T>(&data, m_buffer, -1, -1, 0, data.size(), comm, 0, 0, m_mpi_type, m_seed, m_min_samples, m_add_pivots, true, false);
    } else {
      int64_t split_size = global_elements / size;
      int64_t extra_elements = global_elements % size;
      m_buffer = new T[data.size()];
      ival = QSInterval_SQS<T>(&data, m_buffer, split_size, extra_elements, 0,
                               data.size(), comm, 0, 0, m_mpi_type, m_seed,
                               m_min_samples, m_add_pivots, true);
    }

    auto sorter = ips4o::make_sorter<T*>(std::forward<Compare>(comp));

    /* Recursive */
    quickSort(ival, std::forward<Compare>(comp));

    delete[] m_buffer;

    /* Casos base */
    double start, end;
    if (m_barriers)
      RBC::Barrier(comm);
    start = getTime();
    ordenar2IntervalosPE(std::forward<Compare>(comp), sorter);
    end = getTime();
    t_sort_two = end - start;

    start = getTime();
    ordenarIntervalosLocales(sorter);
    end = getTime();
    t_sort_local = end - start;

    double total_end = getTime();
    t_runtime = (total_end - total_start);
  }

 
  int getDepth() {
    return m_profundidad;
  }

  void getTimers(std::vector<std::string>& timer_names, std::vector<double>& max_timers, RBC::Comm comm) {
    std::vector<double> timers;
    int size, rank;
    RBC::Comm_size(comm, &size);
    RBC::Comm_rank(comm, &rank);
    if (m_barriers) {
      timers.push_back(t_pivot);
      timer_names.push_back("pivot");
      timers.push_back(t_partition);
      timer_names.push_back("partition");
      timers.push_back(t_calculate);
      timer_names.push_back("calculate");
      timers.push_back(t_exchange);
      timer_names.push_back("exchange");

      timers.push_back(t_sort_local + t_sort_two);
      timer_names.push_back("base_cases");
      double sum = 0.0;
      for (size_t i = 0; i < timers.size(); i++)
        sum += timers[i];
      timers.push_back(sum);
      timer_names.push_back("sum");
    }

    timers.push_back(m_profundidad);
    timer_names.push_back("depth");
    timers.push_back(t_create_comms);
    timer_names.push_back("create_comms");

    for (size_t i = 0; i < timers.size(); i++) {
      double time = 0.0;
      RBC::Reduce(&timers[i], &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      max_timers.push_back(time);
    }
  }

 private:

  //Ejecutar el algoritmo Quicksort en el interval QSInterval dado
  template <class Compare>
  void quickSort(QSInterval_SQS<T>& ival, Compare&& comp) {
    m_profundidad++;

    if (isBaseCase(ival)) //Caso base para finalizar le recursión
      return;

    // Selección de pivot
    T pivot;
    int64_t split_idx;
    double t_start, t_end;
    t_start = startTime(ival.m_comm);
    bool zero_global_elements;
    getPivot(ival, pivot, split_idx, std::forward<Compare>(comp), zero_global_elements);
    t_end = getTime();
    t_pivot += (t_end - t_start);

    if (zero_global_elements)
      return;

    //Particionamiento
    t_start = startTime(ival.m_comm);
    int64_t bound1, bound2;
    particionarData(ival, pivot, split_idx, &bound1, &bound2, std::forward<Compare>(comp));
    t_end = getTime();
    t_partition += (t_end - t_start);

    //Calcular cuanta data debe ser intercambiada
    t_start = startTime(ival.m_comm);
    calculateExchangeData(ival, bound1, split_idx, bound2);
    t_end = getTime();
    t_calculate += (t_end - t_start);

    //Intercambiar data
    t_start = startTime(ival.m_comm);
    exchangeData(ival);
    t_end = getTime();
    t_exchange += (t_end - t_start);
    t_vec_exchange.push_back(t_end - t_start);

    //Crear QSIntervals para el siguiente nivel de recursión
    int64_t mid, offset;
    int left_size;
    bool janus;
    calculateSplit(ival, left_size, offset, janus, mid);

    RBC::Comm comm_left, comm_right;
    if (m_use_mpi_collectives)
      t_start = startTime_barrier(ival.m_comm);
    else
      t_start = startTime(ival.m_comm);
    crearNuevosComunicadores(ival, left_size, janus, &comm_left, &comm_right);
    t_end = getTime();
    t_create_comms += (t_end - t_start);

    QSInterval_SQS<T> ival_left, ival_right;
    createIntervals(ival, offset, left_size, janus, mid, comm_left, comm_right, ival_left, ival_right);

    bool sort_left = false, sort_right = false;
    if (ival.m_rank <= ival_left.m_end_pe)
      sort_left = true;
    if (ival.m_rank >= ival_left.m_end_pe) {
      if (ival.m_rank > ival_left.m_end_pe || janus)
        sort_right = true;
    }

    /* Llamar recursivamente */
    if (sort_left && sort_right) {
      janusQuickSort(ival_left, ival_right, std::forward<Compare>(comp));
    } else if (sort_left) {
      quickSort(ival_left, std::forward<Compare>(comp));
    } else if (sort_right) {
      quickSort(ival_right, std::forward<Compare>(comp));
    }
  }

  //Ejecutar el algoritmo QuickSort como janus PE
  template <class Compare>
  void janusQuickSort(QSInterval_SQS<T>& ival_left, QSInterval_SQS<T>& ival_right, Compare&& comp) {
    m_profundidad++;

    if (isBaseCase(ival_left)) { //Finalizacion de recursión en caso base
      quickSort(ival_right, std::forward<Compare>(comp));
      return;
    }
    if (isBaseCase(ival_right)) {
      quickSort(ival_left, std::forward<Compare>(comp));
      return;
    }

    //Seleccion de pivot
    T pivot_left, pivot_right;
    int64_t split_idx_left, split_idx_right;
    double t_start, t_end;
    t_start = startTimeJanus(ival_left.m_comm, ival_right.m_comm);
    getPivotJanus(ival_left, ival_right, pivot_left, pivot_right, split_idx_left, split_idx_right, std::forward<Compare>(comp));
    t_end = getTime();
    t_pivot += (t_end - t_start);

    //Particionamiento
    t_start = startTimeJanus(ival_left.m_comm, ival_right.m_comm);
    int64_t bound1_left, bound2_left, bound1_right, bound2_right;
    particionarData(ival_left, pivot_left, split_idx_left, &bound1_left, &bound2_left, std::forward<Compare>(comp));
    particionarData(ival_right, pivot_right, split_idx_right, &bound1_right, &bound2_right, std::forward<Compare>(comp));
    t_end = getTime();
    t_partition += (t_end - t_start);

    //Calcular cuanta data debe ser intercambiada
    t_start = startTimeJanus(ival_left.m_comm, ival_right.m_comm);
    calculateExchangeDataJanus(ival_left, ival_right, bound1_left, split_idx_left,
                               bound2_left, bound1_right, split_idx_right, bound2_right);
    t_end = getTime();
    t_calculate += (t_end - t_start);

    //Intercambiar data
    t_start = startTimeJanus(ival_left.m_comm, ival_right.m_comm);
    exchangeDataJanus(ival_left, ival_right);
    t_end = getTime();
    t_exchange += (t_end - t_start);
    t_vec_exchange.push_back(t_end - t_start);

    //Crear intervalos QS para el siguiente nivel de recursión
    int64_t mid_left, mid_right, offset_left, offset_right;
    int left_size_left, left_size_right;
    bool janus_left, janus_right;
    calculateSplit(ival_left, left_size_left, offset_left, janus_left, mid_left);
    calculateSplit(ival_right, left_size_right, offset_right, janus_right, mid_right);
    RBC::Comm left1, right1, left2, right2;
    if (m_use_mpi_collectives)
      t_start = startTimeJanus_barrier(ival_left.m_comm, ival_right.m_comm);
    else
      t_start = startTimeJanus(ival_left.m_comm, ival_right.m_comm);
    crearNuevosComunicadoresJanus(ival_left, ival_right, left_size_left, janus_left,
                                left_size_right, janus_right, &left1, &right1, &left2, &right2);
    t_end = getTime();
    t_create_comms += (t_end - t_start);

    QSInterval_SQS<T> ival_left_left, ival_right_left,
      ival_left_right, ival_right_right;
    createIntervals(ival_left, offset_left, left_size_left,
                    janus_left, mid_left, left1, right1,
                    ival_left_left, ival_right_left);
    createIntervals(ival_right, offset_right, left_size_right,
                    janus_right, mid_right, left2, right2,
                    ival_left_right, ival_right_right);

    bool sort_left = false, sort_right = false;
    QSInterval_SQS<T>* left_i, * right_i;
    // Calcular nuevo intervalo izquierdo y si necesita ser ordenado
    if (ival_right_left.m_numero_pes == 1) {
      agregarIntervaloLocal(ival_right_left);
      left_i = &ival_left_left;
      if (ival_left_left.m_numero_pes == ival_left.m_numero_pes)
        sort_left = true;
    } else {
      left_i = &ival_right_left;
      sort_left = true;
    }
    // Calcular nuevo intervalo derecho y si necesita ser ordenado
    if (ival_left_right.m_numero_pes == 1) {
      agregarIntervaloLocal(ival_left_right);
      right_i = &ival_right_right;
      if (ival_right_right.m_numero_pes == ival_right.m_numero_pes)
        sort_right = true;
    } else {
      right_i = &ival_left_right;
      sort_right = true;
    }

    //Llamar recursivamente
    if (sort_left && sort_right) {
      janusQuickSort(*left_i, *right_i, std::forward<Compare>(comp));
    } else if (sort_left) {
      quickSort(*left_i, std::forward<Compare>(comp));
    } else if (sort_right) {
      quickSort(*right_i, std::forward<Compare>(comp));
    }
  }

  //Verificar casos base, true:caso base, false: caso no base
  bool isBaseCase(QSInterval_SQS<T>& ival) {
    if (ival.m_rank == -1)
      return true;
    if (ival.m_numero_pes == 2) {
      agregar2IntervalosPE(ival);
      return true;
    }
    if (ival.m_numero_pes == 1) {
      agregarIntervaloLocal(ival);
      return true;
    }
    return false;
  }

  //Retornar tiempo actual : MPI_Wtime
  double getTime() {
    return MPI_Wtime();
  }

  double startTime(RBC::Comm& comm) {
    if (!m_barriers)
      return getTime();
    RBC::Request req;
    RBC::Ibarrier(comm, &req);
    RBC::Wait(&req, MPI_STATUS_IGNORE);
    return getTime();
  }

  double startTime_barrier(RBC::Comm& comm) {
    RBC::Request req;
    RBC::Ibarrier(comm, &req);
    RBC::Wait(&req, MPI_STATUS_IGNORE);
    return getTime();
  }

  double startTimeJanus(RBC::Comm& left_comm, RBC::Comm& right_comm) {
    if (!m_barriers)
      return getTime();
    RBC::Request req[2];
    RBC::Ibarrier(left_comm, &req[0]);
    RBC::Ibarrier(right_comm, &req[1]);
    RBC::Waitall(2, req, MPI_STATUS_IGNORE);
    return getTime();
  }

  double startTimeJanus_barrier(RBC::Comm& left_comm, RBC::Comm& right_comm) {
    RBC::Request req[2];
    RBC::Ibarrier(left_comm, &req[0]);
    RBC::Ibarrier(right_comm, &req[1]);
    RBC::Waitall(2, req, MPI_STATUS_IGNORE);
    return getTime();
  }

  //Seleccionar un elemento del intervalo como pivot
  template <class Compare>
  void getPivot(QSInterval_SQS<T> const& ival, T& pivot, int64_t& split_idx, Compare&& comp, bool& zero_global_elements) {
    return PivotSelection_SQS<T>::getPivot(ival, pivot, split_idx, std::forward<Compare>(comp),
                                           m_generator, m_sample_generator, zero_global_elements);
  }

  //Seleccionar un elemento como el pivot de ambos intervalos
  template <class Compare>
  void getPivotJanus(QSInterval_SQS<T> const& ival_left,
                     QSInterval_SQS<T> const& ival_right, T& pivot_left,
                     T& pivot_right, int64_t& split_idx_left, int64_t& split_idx_right,
                     Compare&& comp) {
    PivotSelection_SQS<T>::getPivotJanus(ival_left, ival_right, pivot_left, pivot_right,
                                         split_idx_left, split_idx_right, std::forward<Compare>(comp),
                                         m_generator, m_sample_generator);
  }

/*
 * Particionar los datos separadamente para los elementos con el indice mas pequeño que less_idx y los elementos con
 * el indice mas grande que less_idx, retorna los indices del primer elemento de las particiones derecha
 * @param index1 First element of the first partition with large elements
 * @param index2 First element of the second partition with large elements
 */

  template <class Compare>
  void particionarData(QSInterval_SQS<T> const& ival, T pivot, int64_t less_idx,
                     int64_t* index1, int64_t* index2, Compare&& comp) {
    int64_t start1 = ival.m_local_start, end1 = less_idx,
      start2 = less_idx, end2 = ival.m_local_end;
    *index1 = partitionSequence(m_data->data(), pivot, start1, end1, true,
                                std::forward<Compare>(comp));
    *index2 = partitionSequence(m_data->data(), pivot, start2, end2, false,
                                std::forward<Compare>(comp));
  }

/**
 * Particionar los datos con indice [start, end)
 * @param less_equal If true, compare to the pivot with <=, else compare with >
 * @return Indice del primer elemento mas grande
 */
  template <class Compare>
  int64_t partitionSequence(T* data_ptr, T pivot, int64_t start, int64_t end,
                            bool less_equal, Compare&& comp) {
    T* bound;
    if (less_equal) {
      bound = std::partition(data_ptr + start, data_ptr + end, [pivot, &comp](T x) {
          return !comp(pivot, x)  /*x <= pivot*/;
        });
    } else {
      bound = std::partition(data_ptr + start, data_ptr + end, [pivot, &comp](T x) {
          return comp(x, pivot);
        });
    }
    return bound - data_ptr;
  }

/*
 * Prefix sum of small/large elements and broadcast of global small/large elements
 */
  void calculateExchangeData(QSInterval_SQS<T>& ival, int64_t bound1,
                             int64_t split, int64_t bound2) {
    elementsCalculation(ival, bound1, split, bound2);
    int64_t in[2] = { ival.m_local_small_elements, ival.m_local_large_elements };
    int64_t presum[2], global[2];
    RBC::Request request;
    RBC::IscanAndBcast(&in[0], &presum[0], &global[0], 2, MPI_LONG_LONG,
                       MPI_SUM, ival.m_comm, &request, Constants::CALC_EXCH);
    RBC::Wait(&request, MPI_STATUS_IGNORE);

    assignPresum(ival, presum, global);

    if (!ival.m_evenly_distributed) {
      ival.m_split_size = ival.m_global_elements / ival.m_numero_pes;
      ival.m_extra_elements = ival.m_global_elements % ival.m_numero_pes;
      int64_t buf_size = std::max(ival.m_elementos_locales, ival.getLocalElements());
      m_buffer = new T[buf_size];
      ival.m_buffer = m_buffer;
    }
  }

  void elementsCalculation(QSInterval_SQS<T>& ival, int64_t bound1, int64_t split, int64_t bound2) {
    ival.m_bound1 = bound1;
    ival.m_bound2 = bound2;
    ival.m_split = split;
    ival.m_local_small_elements = (bound1 - ival.m_local_start) + (bound2 - split);
    ival.m_local_large_elements = ival.m_elementos_locales - ival.m_local_small_elements;
  }

  void assignPresum(QSInterval_SQS<T>& ival, int64_t presum[2], int64_t global[2]) {
    ival.m_presum_small = presum[0] - ival.m_local_small_elements;
    ival.m_presum_large = presum[1] - ival.m_local_large_elements;
    ival.m_global_small_elements = global[0];
    ival.m_global_large_elements = global[1];
    ival.m_global_elements = ival.m_global_small_elements + ival.m_global_large_elements;
  }

  void calculateExchangeDataJanus(QSInterval_SQS<T>& ival_left,
                                  QSInterval_SQS<T>& ival_right,
                                  int64_t bound1_left, int64_t split_left,
                                  int64_t bound2_left, int64_t bound1_right,
                                  int64_t split_right, int64_t bound2_right) {
    elementsCalculation(ival_left, bound1_left, split_left, bound2_left);
    elementsCalculation(ival_right, bound1_right, split_right, bound2_right);

    int64_t in_left[2] = { ival_left.m_local_small_elements, ival_left.m_local_large_elements };
    int64_t in_right[2] = { ival_right.m_local_small_elements, ival_right.m_local_large_elements };
    int64_t presum_left[2], presum_right[2], global_left[2], global_right[2];
    RBC::Request requests[2];
    RBC::IscanAndBcast(&in_left[0], &presum_left[0], &global_left[0], 2, MPI_LONG_LONG,
                       MPI_SUM, ival_left.m_comm, &requests[1], Constants::CALC_EXCH);
    RBC::IscanAndBcast(&in_right[0], &presum_right[0], &global_right[0], 2, MPI_LONG_LONG,
                       MPI_SUM, ival_right.m_comm, &requests[0], Constants::CALC_EXCH);
    RBC::Waitall(2, requests, MPI_STATUSES_IGNORE);

    assignPresum(ival_left, presum_left, global_left);
    assignPresum(ival_right, presum_right, global_right);
  }

  //Intercambio de datos con otros PEs
  void exchangeData(QSInterval_SQS<T>& ival) {
    DataExchange_SQS<T>::exchangeData(ival);

    if (!ival.m_evenly_distributed) {
      ival.m_elementos_locales = ival.getLocalElements();
      ival.m_local_end = ival.m_local_start + ival.m_elementos_locales;
    }
  }

/*
 * Exchange the data with other PEs on both intervals simultaneously
 */
  void exchangeDataJanus(QSInterval_SQS<T>& left, QSInterval_SQS<T>& right) {
    DataExchange_SQS<T>::exchangeDataJanus(left, right);
  }

/*
 * Calculate how the PEs should be split into two groups
 */
  void calculateSplit(QSInterval_SQS<T>& ival, int& left_size, int64_t& offset,
                      bool& janus, int64_t& mid) {
    assert(ival.m_global_small_elements != 0);
    int64_t last_small_element = ival.m_missing_first_pe + ival.m_global_small_elements - 1;

    left_size = ival.getRankFromIndex(last_small_element) + 1;
    offset = ival.getOffsetFromIndex(last_small_element);

    if (offset + 1 == ival.getSplitSize(left_size - 1))
      janus = false;
    else
      janus = true;

    if (ival.m_rank < left_size - 1) {
      mid = ival.m_local_end;
    } else if (ival.m_rank > left_size - 1) {
      mid = ival.m_local_start;
    } else {
      mid = offset + 1;
    }
  }

/*
 * Splits the communicator into two new, left and right
 */
  void crearNuevosComunicadores(QSInterval_SQS<T>& ival, int64_t left_size,
                              bool janus, RBC::Comm* left, RBC::Comm* right) {
    int size;
    RBC::Comm_size(ival.m_comm, &size);
    int left_end = left_size - 1;
    int right_start = left_size;
    if (janus)
      right_start--;
    int right_end = std::min(static_cast<int64_t>(size - 1), ival.m_global_elements - 1);
    right_end = std::max(right_start, right_end);
    RBC::Split_Comm(ival.m_comm, 0, left_end, right_start, right_end,
                    left, right);
    RBC::Comm_free(ival.m_comm);
  }

  void crearNuevosComunicadoresJanus(QSInterval_SQS<T>& ival_left,
                                   QSInterval_SQS<T>& ival_right, int64_t left_size_left,
                                   bool janus_left, int64_t left_size_right, bool janus_right,
                                   RBC::Comm* left_1, RBC::Comm* right_1,
                                   RBC::Comm* left_2, RBC::Comm* right_2) {
    if (ival_left.m_blocking_priority) {
      crearNuevosComunicadores(ival_left, left_size_left, janus_left, left_1, right_1);
      crearNuevosComunicadores(ival_right, left_size_right, janus_right, left_2, right_2);
    } else {
      crearNuevosComunicadores(ival_right, left_size_right, janus_right, left_2, right_2);
      crearNuevosComunicadores(ival_left, left_size_left, janus_left, left_1, right_1);
    }
  }

  //Crear QSIntervals para el siguiente nivel de recursión
  void createIntervals(QSInterval_SQS<T>& ival, int64_t offset, int left_size,
                       bool janus,
                       int64_t mid, RBC::Comm& comm_left, RBC::Comm& comm_right,
                       QSInterval_SQS<T>& ival_left,
                       QSInterval_SQS<T>& ival_right) {
    int64_t missing_last_left, missing_first_right;
    if (janus) {
      missing_last_left = ival.getSplitSize(left_size - 1) - (offset + 1);
      missing_first_right = offset + 1;
    } else {
      missing_last_left = 0;
      missing_first_right = 0;
    }

    int64_t start = ival.m_local_start;
    int64_t end = ival.m_local_end;
    int64_t extra_elements_left, extra_elements_right,
      split_size_left, split_size_right;
    if (left_size <= ival.m_extra_elements) {
      extra_elements_left = 0;
      split_size_left = ival.m_split_size + 1;
      extra_elements_right = ival.m_extra_elements - left_size;
      if (janus)
        extra_elements_right++;
    } else {
      extra_elements_left = ival.m_extra_elements;
      split_size_left = ival.m_split_size;
      extra_elements_right = 0;
    }
    split_size_right = ival.m_split_size;

    ival_left = QSInterval_SQS<T>(ival.m_data, ival.m_buffer, split_size_left, extra_elements_left,
                                  start, mid, comm_left, ival.m_missing_first_pe, missing_last_left,
                                  m_mpi_type, ival.m_seed, ival.m_min_samples, ival.m_add_pivot, true);
    ival_right = QSInterval_SQS<T>(ival.m_data, ival.m_buffer, split_size_right, extra_elements_right,
                                   mid, end, comm_right, missing_first_right, ival.m_missing_last_pe,
                                   m_mpi_type, ival.m_seed + 1, ival.m_min_samples, ival.m_add_pivot, false);
  }

  //Agregar un intervalo con dos PEs (Caso base)
  void agregar2IntervalosPE(QSInterval_SQS<T> const& ival) {
    m_two_pe_intervals.push_back(ival);
  }

  //Agregar un intervalo que puede ser ordenado localmente (caso base)
  void agregarIntervaloLocal(QSInterval_SQS<T>& ival) {
    m_intervalos_locales.push_back(ival);
  }


  //Ordenar los intervalos guardados con exactamente dos PEs
  template <class Compare, class Sorter>
  void ordenar2IntervalosPE(Compare&& comp, Sorter& sorter) {
    bc2_elements = SequentialSort_SQS<T>::ordenar2IntervalosPE(std::forward<Compare>(comp), sorter, m_two_pe_intervals);
    for (size_t i = 0; i < m_two_pe_intervals.size(); i++)
      RBC::Comm_free(m_two_pe_intervals[i].m_comm);
  }


  //Ordenar todos los intervalos locales
  template <class Sorter>
  void ordenarIntervalosLocales(Sorter& sorter) {
    SequentialSort_SQS<T>::ordenarIntervalosLocales(sorter, m_intervalos_locales);
    bc1_elements = 0.0;
    for (size_t i = 0; i < m_intervalos_locales.size(); i++) {
      bc1_elements += m_intervalos_locales[i].m_elementos_locales;
      RBC::Comm_free(m_intervalos_locales[i].m_comm);
    }
  }

  MPI_Datatype m_mpi_type;
  int m_profundidad = 0, m_seed;
  double t_pivot = 0.0, t_calculate = 0.0, t_exchange = 0.0, t_partition = 0.0, t_sort_two = 0.0, t_sort_local = 0.0, t_create_comms = 0.0, t_runtime, bc1_elements, bc2_elements;
  std::vector<double> t_vec_exchange, exchange_times { 0.0, 0.0, 0.0, 0.0 };
  T* m_buffer;
  std::vector<T>* m_data;
  std::mt19937_64 m_generator, m_sample_generator; //m_generator esta sincronizado entre procesos, //m_sample_generator es un generador aleatorio no sincronizado
  std::vector<QSInterval_SQS<T> > m_intervalos_locales, m_two_pe_intervals;
  bool m_barriers, m_split_mpi_comm, m_use_mpi_collectives, m_add_pivots;
  int64_t m_min_samples;
};
}