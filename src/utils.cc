// Copyright 2012 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <assert.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_sf.h>

#include <iostream>

#include "utils.h"

namespace hlda {

// =======================================================================
// Utils
// =======================================================================

gsl_rng* Utils::RANDNUMGEN = NULL;

double Utils::Sum(const vector<double>& v) {
  double sum = 0;
  int size = v.size();
  for (int i = 0; i < size; i++) {
    sum += v[i];
  }
  return sum;
}

double Utils::LogSum(double log_a, double log_b) {
  if (log_a < log_b) {
      return(log_b + log(1 + exp(log_a - log_b)));
  } else {
      return(log_a + log(1 + exp(log_b - log_a)));
  }
}

int Utils::SampleFromLogPr(const vector<double>& log_pr) {
  assert(log_pr.size() > 0);
  // Initialize the log_sum to the log probability at level 0.
  double log_sum = log_pr[0];
  int levels = log_pr.size();

  // Update the log sum by taking into account all other levels.
  for (int i = 1; i < levels; i++) {
    log_sum = LogSum(log_sum, log_pr[i]);
  }

  // Obtain a random number.
  double rand_no = RandNo();

  double log_exp = exp(log_pr[0] - log_sum);
  int result = 0;
  while (rand_no >= log_exp) {
    result++;
    log_exp += exp(log_pr[result] - log_sum);
  }

  return result;
}

void Utils::InitRandomNumberGen(long rng_seed) {
  if (RANDNUMGEN != NULL) return;

  RANDNUMGEN = gsl_rng_alloc(gsl_rng_taus);
  cout << "Random seed = " << rng_seed << endl;
  gsl_rng_set(RANDNUMGEN, rng_seed);
}

void Utils::Shuffle(gsl_permutation* permutation, int size) {
  assert(RANDNUMGEN != NULL);
  gsl_ran_shuffle(RANDNUMGEN, permutation->data, size, sizeof(size_t));
}

double Utils::RandGauss(double mean, double stdev) {
  assert(RANDNUMGEN != NULL);
  double gauss = gsl_ran_gaussian_ratio_method(RANDNUMGEN, stdev) + mean;
  return(gauss);
}

double Utils::RandNo() {
  assert(RANDNUMGEN != NULL);
  return gsl_rng_uniform(RANDNUMGEN);
}

}  // namespace hlda


