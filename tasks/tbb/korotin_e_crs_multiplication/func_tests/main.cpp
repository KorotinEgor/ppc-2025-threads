#include <gtest/gtest.h>

#include <iostream>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/korotin_e_crs_multiplication/include/ops_tbb.hpp"

namespace korotin_e_crs_multiplication_tbb {

std::vector<double> GetRandomMatrix(unsigned int m, unsigned int n) {
  if (m * n == 0) {
    throw std::invalid_argument("Can't creaate matrix with 0 rows or columns");
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(-100.0, 100.0);
  std::vector<double> res(m * n);
  for (unsigned int i = 0; i < m; i++) {
    for (unsigned int j = 0; j < n; j++) {
      res[(i * n) + j] = distrib(gen);
    }
  }
  return res;
}

void MakeCRS(std::vector<unsigned int> &r_i, std::vector<unsigned int> &col, std::vector<double> &val,
             const std::vector<double> &src, unsigned int m, unsigned int n) {
  if (m * n != src.size()) {
    throw std::invalid_argument("Size of the source matrix does not match the specified");
  }
  r_i = std::vector<unsigned int>(m + 1, 0);
  col.clear();
  val.clear();
  for (unsigned int i = 0; i < m; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (src[(i * n) + j] != 0) {
        val.push_back(src[(i * n) + j]);
        col.push_back(j);
        r_i[i + 1]++;
      }
    }
  }
  for (unsigned int i = 1; i <= m; i++) {
    r_i[i] += r_i[i - 1];
  }
}

void MatrixMultiplication(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                          unsigned int m, unsigned int n, unsigned int p) {
  for (unsigned int i = 0; i < m; i++) {
    for (unsigned int k = 0; k < n; k++) {
      for (unsigned int j = 0; j < p; j++) {
        c[(i * p) + j] += a[(i * n) + k] * b[(k * p) + j];
      }
    }
  }
}

}  // namespace korotin_e_crs_multiplication_tbb

TEST(korotin_e_crs_multiplication_tbb, test_rnd_50_50_50) {
  const unsigned int m = 50;
  const unsigned int n = 50;
  const unsigned int p = 50;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> a_val;
  std::vector<double> b_val;
  std::vector<unsigned int> a_ri;
  std::vector<unsigned int> a_col;
  std::vector<unsigned int> b_ri;
  std::vector<unsigned int> b_col;
  a = korotin_e_crs_multiplication_tbb::GetRandomMatrix(m, n);
  b = korotin_e_crs_multiplication_tbb::GetRandomMatrix(n, p);
  korotin_e_crs_multiplication_tbb::MakeCRS(a_ri, a_col, a_val, a, m, n);
  korotin_e_crs_multiplication_tbb::MakeCRS(b_ri, b_col, b_val, b, n, p);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_ri.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_val.data()));
  task_data_tbb->inputs_count.emplace_back(a_ri.size());
  task_data_tbb->inputs_count.emplace_back(a_col.size());
  task_data_tbb->inputs_count.emplace_back(a_val.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_ri.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_val.data()));
  task_data_tbb->inputs_count.emplace_back(b_ri.size());
  task_data_tbb->inputs_count.emplace_back(b_col.size());
  task_data_tbb->inputs_count.emplace_back(b_val.size());

  std::vector<unsigned int> out_ri(a_ri.size(), 0);
  std::vector<unsigned int> out_col(m * p);
  std::vector<double> out_val(m * p);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ri.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
  task_data_tbb->outputs_count.emplace_back(out_ri.size());

  korotin_e_crs_multiplication_tbb::CrsMultiplicationTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  std::vector<double> c(m * p, 0);
  std::vector<double> c_val;
  std::vector<unsigned int> c_ri;
  std::vector<unsigned int> c_col;
  korotin_e_crs_multiplication_tbb::MatrixMultiplication(a, b, c, m, n, p);

  korotin_e_crs_multiplication_tbb::MakeCRS(c_ri, c_col, c_val, c, m, p);
  /*ASSERT_EQ(c_val.size(), out_val.size());
  bool flag = true;
  for (size_t k = 0; k < c_val.size(); k++) {
    if (c_val[k] != out_val[k]) {
      std::cout << "C:   " << c_val[k] << std::endl;
      std::cout << "OUT: " << out_val[k] << std::endl;
      flag = false;
    }
  }
  if (flag)
    std::cout << "\nKAKOGO HERA" << std::endl;
  else std::cout << "\nEI TY, MUDILA" << std::endl;*/
  ASSERT_EQ(c_ri, out_ri);
  ASSERT_EQ(c_col, out_col);
  ASSERT_EQ(c_val, out_val);
}

TEST(korotin_e_crs_multiplication_tbb, test_rndcrs_stat_zeroes) {
  const unsigned int m = 50;
  const unsigned int n = 50;
  const unsigned int p = 50;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> a_val;
  std::vector<double> b_val;
  std::vector<unsigned int> a_ri;
  std::vector<unsigned int> a_col;
  std::vector<unsigned int> b_ri;
  std::vector<unsigned int> b_col;
  a = korotin_e_crs_multiplication_tbb::GetRandomMatrix(m, n);
  b = korotin_e_crs_multiplication_tbb::GetRandomMatrix(n, p);
  for (unsigned int i = 0; i < (m * n) / 2; i++) {
    a[i * 2] = 0;
  }
  for (unsigned int i = 0; i < (n * p) / 2; i++) {
    b[(i * 2) + 1] = 0;
  }

  korotin_e_crs_multiplication_tbb::MakeCRS(a_ri, a_col, a_val, a, m, n);
  korotin_e_crs_multiplication_tbb::MakeCRS(b_ri, b_col, b_val, b, n, p);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_ri.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_val.data()));
  task_data_tbb->inputs_count.emplace_back(a_ri.size());
  task_data_tbb->inputs_count.emplace_back(a_col.size());
  task_data_tbb->inputs_count.emplace_back(a_val.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_ri.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_val.data()));
  task_data_tbb->inputs_count.emplace_back(b_ri.size());
  task_data_tbb->inputs_count.emplace_back(b_col.size());
  task_data_tbb->inputs_count.emplace_back(b_val.size());

  std::vector<double> c(m * p, 0);
  std::vector<double> c_val;
  std::vector<unsigned int> c_ri;
  std::vector<unsigned int> c_col;
  korotin_e_crs_multiplication_tbb::MatrixMultiplication(a, b, c, m, n, p);
  korotin_e_crs_multiplication_tbb::MakeCRS(c_ri, c_col, c_val, c, m, p);

  std::vector<unsigned int> out_ri(a_ri.size(), 0);
  std::vector<unsigned int> out_col(c_col.size());
  std::vector<double> out_val(c_val.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ri.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
  task_data_tbb->outputs_count.emplace_back(out_ri.size());

  korotin_e_crs_multiplication_tbb::CrsMultiplicationTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  ASSERT_EQ(c_ri, out_ri);
  ASSERT_EQ(c_col, out_col);
  ASSERT_EQ(c_val, out_val);
}

TEST(korotin_e_crs_multiplication_tbb, test_rndcrs) {
  const unsigned int m = 50;
  const unsigned int n = 50;
  const unsigned int p = 50;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> a_val;
  std::vector<double> b_val;
  std::vector<unsigned int> a_ri;
  std::vector<unsigned int> a_col;
  std::vector<unsigned int> b_ri;
  std::vector<unsigned int> b_col;
  a = korotin_e_crs_multiplication_tbb::GetRandomMatrix(m, n);
  b = korotin_e_crs_multiplication_tbb::GetRandomMatrix(n, p);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> a_distrib(0, (m * n) - 1);
  std::uniform_int_distribution<unsigned int> b_distrib(0, (n * p) - 1);
  for (unsigned int i = 0; i < (m * n) - m; i++) {
    a[a_distrib(gen)] = 0;
  }
  for (unsigned int i = 0; i < (n * p) - p; i++) {
    b[b_distrib(gen)] = 0;
  }

  korotin_e_crs_multiplication_tbb::MakeCRS(a_ri, a_col, a_val, a, m, n);
  korotin_e_crs_multiplication_tbb::MakeCRS(b_ri, b_col, b_val, b, n, p);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_ri.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_val.data()));
  task_data_tbb->inputs_count.emplace_back(a_ri.size());
  task_data_tbb->inputs_count.emplace_back(a_col.size());
  task_data_tbb->inputs_count.emplace_back(a_val.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_ri.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_val.data()));
  task_data_tbb->inputs_count.emplace_back(b_ri.size());
  task_data_tbb->inputs_count.emplace_back(b_col.size());
  task_data_tbb->inputs_count.emplace_back(b_val.size());

  std::vector<double> c(m * p, 0);
  std::vector<double> c_val;
  std::vector<unsigned int> c_ri;
  std::vector<unsigned int> c_col;
  korotin_e_crs_multiplication_tbb::MatrixMultiplication(a, b, c, m, n, p);
  korotin_e_crs_multiplication_tbb::MakeCRS(c_ri, c_col, c_val, c, m, p);

  std::vector<unsigned int> out_ri(a_ri.size(), 0);
  std::vector<unsigned int> out_col(c_col.size());
  std::vector<double> out_val(c_val.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ri.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
  task_data_tbb->outputs_count.emplace_back(out_ri.size());

  korotin_e_crs_multiplication_tbb::CrsMultiplicationTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  ASSERT_EQ(c_ri, out_ri);
  ASSERT_EQ(c_col, out_col);
  ASSERT_EQ(c_val, out_val);
}

TEST(korotin_e_crs_multiplication_tbb, test_rnd_20_40_60) {
  const unsigned int m = 20;
  const unsigned int n = 40;
  const unsigned int p = 60;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> a_val;
  std::vector<double> b_val;
  std::vector<unsigned int> a_ri;
  std::vector<unsigned int> a_col;
  std::vector<unsigned int> b_ri;
  std::vector<unsigned int> b_col;
  a = korotin_e_crs_multiplication_tbb::GetRandomMatrix(m, n);
  b = korotin_e_crs_multiplication_tbb::GetRandomMatrix(n, p);
  korotin_e_crs_multiplication_tbb::MakeCRS(a_ri, a_col, a_val, a, m, n);
  korotin_e_crs_multiplication_tbb::MakeCRS(b_ri, b_col, b_val, b, n, p);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_ri.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_val.data()));
  task_data_tbb->inputs_count.emplace_back(a_ri.size());
  task_data_tbb->inputs_count.emplace_back(a_col.size());
  task_data_tbb->inputs_count.emplace_back(a_val.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_ri.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_val.data()));
  task_data_tbb->inputs_count.emplace_back(b_ri.size());
  task_data_tbb->inputs_count.emplace_back(b_col.size());
  task_data_tbb->inputs_count.emplace_back(b_val.size());

  std::vector<unsigned int> out_ri(a_ri.size(), 0);
  std::vector<unsigned int> out_col(m * p);
  std::vector<double> out_val(m * p);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ri.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
  task_data_tbb->outputs_count.emplace_back(out_ri.size());

  korotin_e_crs_multiplication_tbb::CrsMultiplicationTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  std::vector<double> c(m * p, 0);
  std::vector<double> c_val;
  std::vector<unsigned int> c_ri;
  std::vector<unsigned int> c_col;
  korotin_e_crs_multiplication_tbb::MatrixMultiplication(a, b, c, m, n, p);

  korotin_e_crs_multiplication_tbb::MakeCRS(c_ri, c_col, c_val, c, m, p);
  ASSERT_EQ(c_ri, out_ri);
  ASSERT_EQ(c_col, out_col);
  ASSERT_EQ(c_val, out_val);
}

TEST(korotin_e_crs_multiplication_tbb, test_rnd_rnd_bords) {
  const unsigned int m = (rand() % 50) + 1;
  const unsigned int n = (rand() % 50) + 1;
  const unsigned int p = (rand() % 50) + 1;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> a_val;
  std::vector<double> b_val;
  std::vector<unsigned int> a_ri;
  std::vector<unsigned int> a_col;
  std::vector<unsigned int> b_ri;
  std::vector<unsigned int> b_col;
  a = korotin_e_crs_multiplication_tbb::GetRandomMatrix(m, n);
  b = korotin_e_crs_multiplication_tbb::GetRandomMatrix(n, p);
  korotin_e_crs_multiplication_tbb::MakeCRS(a_ri, a_col, a_val, a, m, n);
  korotin_e_crs_multiplication_tbb::MakeCRS(b_ri, b_col, b_val, b, n, p);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_ri.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_val.data()));
  task_data_tbb->inputs_count.emplace_back(a_ri.size());
  task_data_tbb->inputs_count.emplace_back(a_col.size());
  task_data_tbb->inputs_count.emplace_back(a_val.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_ri.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_val.data()));
  task_data_tbb->inputs_count.emplace_back(b_ri.size());
  task_data_tbb->inputs_count.emplace_back(b_col.size());
  task_data_tbb->inputs_count.emplace_back(b_val.size());

  std::vector<unsigned int> out_ri(a_ri.size(), 0);
  std::vector<unsigned int> out_col(m * p);
  std::vector<double> out_val(m * p);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ri.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
  task_data_tbb->outputs_count.emplace_back(out_ri.size());

  korotin_e_crs_multiplication_tbb::CrsMultiplicationTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  std::vector<double> c(m * p, 0);
  std::vector<double> c_val;
  std::vector<unsigned int> c_ri;
  std::vector<unsigned int> c_col;
  korotin_e_crs_multiplication_tbb::MatrixMultiplication(a, b, c, m, n, p);

  korotin_e_crs_multiplication_tbb::MakeCRS(c_ri, c_col, c_val, c, m, p);
  ASSERT_EQ(c_ri, out_ri);
  ASSERT_EQ(c_col, out_col);
  ASSERT_EQ(c_val, out_val);
}