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


#ifndef GIBBS_H_
#define GIBBS_H_

#include <string>

#include "corpus.h"
#include "topic.h"
#include "tree.h"
#include "utils.h"

namespace hlda {

// The Gibbs state of the HLDA implementation.
// Each Gibbs state has a corpus and a tree, and
// keeps current scores, the current iteration and
// the sampling parameters.
class GibbsState {
 public:
  GibbsState();
  ~GibbsState();

  GibbsState(const GibbsState& from);
  GibbsState& operator=(const GibbsState& from);

  // Computes the Gibbs score, which is a perplexity score for
  // the model.
  // It is formed by summing the GEM score obtained for the
  // GEM distribution with the Eta score (the topic score) and
  // the Gamma score (the Chinese Restaurant Process CRP score).
  double computeGibbsScore();

  void setScore(double score) { score_ = score; }
  double getScore() const { return score_; }

  void setGemScore(double gem_score) { gem_score_ = gem_score; }
  double getGemScore() const { return gem_score_; }

  void setEtaScore(double eta_score) { eta_score_ = eta_score; }
  double getEtaScore() const { return eta_score_; }

  void setGammaScore(double gamma_score) { gamma_score_ = gamma_score; }
  double getGammaScore() const { return gamma_score_; }

  double getMaxScore() const { return max_score_; }
  void setMaxScore(double max_score) { max_score_ = max_score; }

  void setSampleEta(int sample_eta) { sample_eta_ = sample_eta; }
  void setSampleGem(int sample_gem) { sample_gem_ = sample_gem; }

  void setCorpus(const Corpus& corpus) { corpus_ = corpus; }
  Corpus* getMutableCorpus() { return &corpus_; }

  void setTree(const Tree& tree) { tree_ = tree; }
  Tree* getMutableTree() { return &tree_; }

  int getIteration() const { return iteration_; }
  void setIteration(int iteration) { iteration_ = iteration; }
  void incIteration(int val) { iteration_ += val; }

  int getLevelLag() const { return level_lag_; }
  int getShuffleLag() const { return shuffle_lag_; }
  int getHyperLag() const { return hyper_lag_; }
  int getSampleEta() const { return sample_eta_; }
  int getSampleGem() const { return sample_gem_; }
  int getSampleGam() const { return sample_gam_; }

 private:
  Corpus corpus_;
  Tree tree_;

  // The current score obtained by summing the Eta, Gamma and
  // GEM scores.
  double score_;

  // Corresponds to the GEM distribution with 2 parameters:
  // gem_mean and gem_scale.
  double gem_score_;

  // Corresponds to the topic prior Eta parameter.
  double eta_score_;

  // Corresponds to the CRP parameter Gamma.
  double gamma_score_;

  // The current maximum score over several iterations.
  double max_score_;

  // Current iteration.
  int iteration_;

  // Sampling parameters.
  int shuffle_lag_;
  int hyper_lag_;
  int level_lag_;
  int sample_eta_;
  int sample_gem_;
  int sample_gam_;
};

// This class provides functionality for reading input for the
// Gibbs state, initializing the Gibbs state,
// and performing iterations of the Gibbs state.
class GibbsSampler {
 public:
  // Read input corpus and state parameters from file.
  static void ReadGibbsInput(
      GibbsState* gibbs_state,
      const std::string& filename_corpus,
      const std::string& filename_settings);

  // Initialize Gibbs state.
  static void InitGibbsState(
      GibbsState* gibbs_state);

  // Initialize Gibbs state - repeat the initialization REP_NO,
  // by calling InitGibbsState.
  // Keep the Gibbs state with the best score.
  // rng_seed is the random number generator seed.
  static GibbsState* InitGibbsStateRep(
      const std::string& filename_corpus,
      const std::string& filename_settings,
      long rng_seed);

  // Iterations of the Gibbs state.
  // Sample the document path and the word levels in the tree.
  // Sample hyperparameters: Eta, GEM mean and scale.
  static void IterateGibbsState(GibbsState* gibbs_state);
};

}  // namespace hlda

#endif  // GIBBS_H_
