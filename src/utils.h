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


#ifndef UTILS_H_
#define UTILS_H_

#include <gsl/gsl_permutation.h>
#include <gsl/gsl_randist.h>

#include <vector>

using namespace std;

namespace hlda {

// This class provides functionality for summing values,
// for reading data from files and also provides
// an interface to gsl specific methods.
// This class is not thread-safe (see private static gsl_rng* RANDNUMGEN).
class Utils {
 public:
  // Sum up the values in a vector.
  static double Sum(const vector<double>& v);

  // An approximation of the logarithm of a sum.
  // Given log(a) denoted with log_a and log(b) denoted with log_b
  // it approximates log(a + b) as:
  // if log_a < log_b then log(a + b) = log_b + log(1 + exp(log_a - log_b))
  // otherwise log(a + b) = log_a + log(1 + exp(log_b - log_a)).
  static double LogSum(double log_a, double log_b);

  // Sample log probabilities.
  // The log_pr vector keeps for each level
  // the current log probability of the word + the current
  // log probability of the level in the tree.
  // These values are used to sample the new level.
  // The vector should contain at least one element.
  static int SampleFromLogPr(const vector<double>& log_pr);

  // Shuffle the values in a gsl_permutation.
  static void Shuffle(gsl_permutation* permutation, int size);

  // Initialize the random number generator.
  // rng_seed is the random number generator seed.
  static void InitRandomNumberGen(long rng_seed);

  // Return a gsl Gaussian random variate with mean and stdev as parameters
  static double RandGauss(double mean, double stdev);

  // Return a random number using the gsl random number generator.
  static double RandNo();

 private:
  static gsl_rng* RANDNUMGEN;
};

}  // namespace hlda

#endif  // UTILS_H_
