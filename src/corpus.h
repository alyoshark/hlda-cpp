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


#ifndef CORPUS_H_
#define CORPUS_H_

#include <string>

#include "document.h"
#include "utils.h"

namespace hlda {

// A corpus containing a number of documents.
// The parameters of the GEM distribution: gem_mean_ and
// gem_scale_ are also defined at the corpus level.
class Corpus {
 public:
  Corpus();
  Corpus(double gem_mean, double gem_scale);
  ~Corpus();

  Corpus(const Corpus& from);
  Corpus& operator=(const Corpus& from);

  void setWordNo(int word_no) { word_no_ = word_no; }
  int getWordNo() const { return word_no_; }

  void addDocument(const Document& document) {
    documents_.push_back(document);
  }
  int getDocuments() const { return documents_.size(); }
  Document* getMutableDocument(int i) { return &documents_.at(i); }
  void setDocuments(const vector<Document>& documents) {
    documents_ = documents;
  }

  double getGemMean() const { return gem_mean_; }
  void setGemMean(double gem_mean) { gem_mean_ = gem_mean; }

  double getGemScale() const { return gem_scale_; }
  void setGemScale(double gem_scale) { gem_scale_ = gem_scale; }

 private:
  // Parameters of the GEM distribution.
  // gem_mean shows the proportion of general words relative to specific words.
  double gem_mean_;

  // gem_scale shows how strictly documents should follow the general
  // versus specific word proportions.
  double gem_scale_;

  // The number of distinct words in the corpus.
  int word_no_;

  // The documents in this corpus.
  vector<Document> documents_;
};

// This class provides functionality for reading a corpus from a file,
// computing and updating the GEM distribution parameters
// and permuting the documents in the corpus.
class CorpusUtils {
 public:
  // Read corpus from file.
  static void ReadCorpus(
      const std::string& filename,
      Corpus* corpus,
      int depth);

  // Corpus level GEM score.
  static double GemScore(
      Corpus* corpus);

  // Update the GEM scale parameter.
  // The new GEM scale parameter is based on Gaussian random variates.
  // Repeat REP_NO_GEM number of times.
  static void UpdateGemScale(Corpus* corpus);

  // Update the GEM scale parameter.
  // The new GEM scale parameter is based on Gaussian random variates.
  // Repeat REP_NO_GEM number of times.
  static void UpdateGemMean(Corpus* corpus);

  // Permute the documents in the corpus.
  static void PermuteDocuments(Corpus* corpus);
};

}  // namespace hlda

#endif  // CORPUS_H_
