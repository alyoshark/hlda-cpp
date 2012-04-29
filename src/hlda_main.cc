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


#include <iostream>

#include "gibbs.h"

using hlda::GibbsSampler;
using hlda::GibbsState;

#define MAX_ITERATIONS 10000

int main(int argc, char** argv) {
  if (argc == 3) {
    // The random number generator seed.
    // For testing an example seed is: t = 1147530551;
    long rng_seed;
    (void) time(&rng_seed);

    std::string filename_corpus = argv[1];
    std::string filename_settings = argv[2];
    hlda::GibbsState* gibbs_state = hlda::GibbsSampler::InitGibbsStateRep(
        filename_corpus, filename_settings, rng_seed);

    for (int i = 0; i < MAX_ITERATIONS; i++) {
      hlda::GibbsSampler::IterateGibbsState(gibbs_state);
    }

    delete gibbs_state;
  } else {
    cout << "Arguments: "
        "(1) corpus filename "
        "(2) settings filename" << endl;
  }
  return 0;
}

