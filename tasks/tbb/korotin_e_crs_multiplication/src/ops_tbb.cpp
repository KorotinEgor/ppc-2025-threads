#include "tbb/korotin_e_crs_multiplication/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/info.h"

bool korotin_e_crs_multiplication_tbb::CrsMultiplicationTBB::PreProcessingImpl() {
  A_N_ = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[0]);
  A_rI_ = std::vector<unsigned int>(in_ptr, in_ptr + A_N_);

  A_Nz_ = task_data->inputs_count[1];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[1]);
  A_col_ = std::vector<unsigned int>(in_ptr, in_ptr + A_Nz_);

  auto *val_ptr = reinterpret_cast<double *>(task_data->inputs[2]);
  A_val_ = std::vector<double>(val_ptr, val_ptr + A_Nz_);

  B_N_ = task_data->inputs_count[3];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[3]);
  B_rI_ = std::vector<unsigned int>(in_ptr, in_ptr + B_N_);

  B_Nz_ = task_data->inputs_count[4];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[4]);
  B_col_ = std::vector<unsigned int>(in_ptr, in_ptr + B_Nz_);

  val_ptr = reinterpret_cast<double *>(task_data->inputs[5]);
  B_val_ = std::vector<double>(val_ptr, val_ptr + B_Nz_);

  unsigned int output_size = task_data->outputs_count[0];
  output_rI_ = std::vector<unsigned int>(output_size);

  return true;
}

bool korotin_e_crs_multiplication_tbb::CrsMultiplicationTBB::ValidationImpl() {
  return task_data->inputs_count[1] == task_data->inputs_count[2] &&
         task_data->inputs_count[4] == task_data->inputs_count[5] &&
         task_data->inputs_count[0] == task_data->outputs_count[0] &&
         *std::max_element(reinterpret_cast<unsigned int *>(task_data->inputs[1]),
                           reinterpret_cast<unsigned int *>(task_data->inputs[1]) + task_data->inputs_count[1]) <=
             task_data->inputs_count[3] - 2;
}

bool korotin_e_crs_multiplication_tbb::CrsMultiplicationTBB::RunImpl() {
  std::vector<unsigned int> tr_i(*std::max_element(B_col_.begin(), B_col_.end()) + 2, 0);
  unsigned int i = 0;
  unsigned int j = 0;
  for (i = 0; i < B_Nz_; i++) {
    tr_i[B_col_[i] + 1]++;
  }
  for (i = 1; i < tr_i.size(); i++) {
    tr_i[i] += tr_i[i - 1];
  }

  std::vector<unsigned int> tcol(B_Nz_, 0);
  std::vector<double> tval(B_Nz_, 0);
  for (i = 0; i < B_N_ - 1; i++) {
    for (j = B_rI_[i]; j < B_rI_[i + 1]; j++) {
      tval[tr_i[B_col_[j]]] = B_val_[j];
      tcol[tr_i[B_col_[j]]] = i;
      tr_i[B_col_[j]]++;
    }
  }
  for (i = tr_i.size() - 1; i > 0; i--) {
    tr_i[i] = tr_i[i - 1];
  }
  tr_i[0] = 0;

  std::fill(output_rI_.begin(), output_rI_.end(), 0);
  output_col_.clear();
  output_val_.clear();
  std::vector<std::vector<double>> local_val(A_N_);
  std::vector<std::vector<int>> local_col(A_N_);
  std::vector<unsigned int> temp_rI(A_N_, 0);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, A_N_ - 1),
                    [&](const tbb::blocked_range<size_t>& r) {
    for (size_t k = r.begin(); k != r.end(); ++k) {
      for (size_t s = 0; s < tr_i.size() - 1; ++s) {
        double sum = 0;
        unsigned int ai = A_rI_[k];
        unsigned int bt = tr_i[s];
        while (ai < A_rI_[k + 1] && bt < tr_i[s + 1]) {
          if (A_col_[ai] == tcol[bt]) {
            sum += A_val_[ai] * tval[bt];
            ai++;
            bt++;
          } else if (A_col_[ai] < tcol[bt]) {
            ai++;
          } else {
            bt++;
          }
        }
        if (sum != 0) {
          local_val[k].push_back(sum);
          local_col[k].push_back(s);
          temp_rI[k + 1]++;
        }
      }
    }
    }
  );

  for (unsigned int t = 0; t < A_N_; ++t) {
    output_val_.insert(output_val_.end(), local_val[t].begin(), local_val[t].end());
    output_col_.insert(output_col_.end(), local_col[t].begin(), local_col[t].end());
  }

  for (i = 1; i < A_N_; ++i) {
    output_rI_[i] += output_rI_[i - 1] + temp_rI[i];
  }
  return true;
}

bool korotin_e_crs_multiplication_tbb::CrsMultiplicationTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_rI_.size(); i++) {
    reinterpret_cast<unsigned int *>(task_data->outputs[0])[i] = output_rI_[i];
  }
  for (size_t i = 0; i < output_col_.size(); i++) {
    reinterpret_cast<unsigned int *>(task_data->outputs[1])[i] = output_col_[i];
    reinterpret_cast<double *>(task_data->outputs[2])[i] = output_val_[i];
  }
  task_data->outputs_count.emplace_back(output_col_.size());
  task_data->outputs_count.emplace_back(output_val_.size());
  return true;
}
