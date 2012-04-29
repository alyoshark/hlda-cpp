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
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_sf.h>
#include <math.h>

#include "document.h"
#include "utils.h"

namespace hlda {

// =======================================================================
// Word
// =======================================================================

Word::Word(int id, int count, int level)
    : id_(id),
      count_(count),
      level_(level) {
}

Word::Word(const Word& from)
    : id_(from.id_),
      count_(from.count_),
      level_(from.level_) {
}

Word& Word::operator =(const Word& from) {
  if (this == &from) return *this;
  id_ = from.id_;
  count_ = from.count_;
  level_ = from.level_;
  return *this;
}

Word::~Word() {
}


// =======================================================================
// Document
// =======================================================================

Document::Document() {
}

Document::Document(int id, int depth)
    : id_(id),
      depth_(depth),
      score_(0.0) {
  level_counts_ = new int[depth];
  log_pr_level_ = new double[depth];

  for (int i = 0; i < depth; i++) {
    level_counts_[i] = 0;
    log_pr_level_[i] = 0.0;
    path_.push_back(NULL);
  }
}

Document::Document(const Document& from)
    : id_(from.id_),
      words_(from.words_),
      depth_(from.depth_),
      score_(from.score_) {
  level_counts_ = new int[depth_];
  log_pr_level_ = new double[depth_];

  for (int i = 0; i < depth_; i++) {
    level_counts_[i] = from.level_counts_[i];
    log_pr_level_[i] = from.log_pr_level_[i];
    path_.push_back(from.path_.at(i));
  }
}

Document& Document::operator =(const Document& from) {
  if (this == &from) return *this;

  id_ = from.id_;
  words_ = from.words_;
  depth_ = from.depth_;
  score_ = from.score_;

  int* new_level_counts = new int[depth_];
  double* new_log_pr_level = new double[depth_];

  for (int i = 0; i < depth_; i++) {
    new_level_counts[i] = from.level_counts_[i];
    new_log_pr_level[i] = from.log_pr_level_[i];
    path_[i] = from.path_.at(i);
  }

  delete[] level_counts_;
  delete[] log_pr_level_;

  level_counts_ = new_level_counts;
  log_pr_level_ = new_log_pr_level;

  return *this;
}

Document::~Document() {
  delete[] level_counts_;
  delete[] log_pr_level_;
}

void Document::addWord(
    int word_id, int word_count, int level) {
  Word word(word_id, word_count, level);
  for (int i = 0; i < word_count; i++) {
    words_.push_back(word);
  }
}

void Document::initLevelCounts(int depth) {
  for (int i = 0; i < depth; i++) {
    level_counts_[i] = 0;
    log_pr_level_[i] = 0.0;
  }
}

void Document::computeLogPrLevel(double gem_mean, double gem_scale, int depth) {
  int sum_level_counts = getSumLevelCounts(depth);
  double sum_log_pr = 0.0;
  double last_section = 0.0;

  for (int i = 0; i < depth - 1; i++) {
    sum_level_counts -= level_counts_[i];

    double expected_stick_len =
        ((1 - gem_mean) * gem_scale + level_counts_[i]) /
        (gem_scale + level_counts_[i] + sum_level_counts);

    log_pr_level_[i] = log(expected_stick_len) + sum_log_pr;

    if (i == 0) {
      last_section = log_pr_level_[i];
    } else {
      last_section = Utils::LogSum(log_pr_level_[i], last_section);
    }
    sum_log_pr += log(1 - expected_stick_len);
  }

  last_section = log(1.0 - exp(last_section));
  log_pr_level_[depth - 1] = last_section;
}

int Document::getSumLevelCounts(int depth) const {
  int sum = 0;
  for (int i = 0; i < depth; i++) {
    sum += level_counts_[i];
  }
  return sum;
}

// =======================================================================
// DocumentUtils
// =======================================================================

void DocumentUtils::SampleLevels(
    Document* document,
    int permute_words,
    bool remove,
    double gem_mean,
    double gem_scale) {
  int depth = document->getMutablePathTopic(0)->getMutableTree()->getDepth();
  vector<double> log_pr(depth);

  // Permute the words in the document.
  if (permute_words == 1) {
    PermuteWords(document);
  }

  for (int i = 0; i < document->getWords(); i++) {
    Word* word = document->getMutableWord(i);
    if (remove) {
      int level = word->getLevel();
      // Update the word level.
      document->updateLevelCounts(level, -1);
      // Decrease the word count.
      document->getMutablePathTopic(level)->updateWordCount(word->getId(), -1);
    }

    // Compute probabilities.
    // Compute log probabilities for all levels.
    // Use the corpus GEM mean and scale.
    document->computeLogPrLevel(gem_mean, gem_scale, depth);

    for (int j = 0; j < depth; j++) {
      double log_pr_level = document->getLogPrLevel(j);
      double log_pr_word =
          document->getMutablePathTopic(j)->getLogPrWord(word->getId());
      double log_value = log_pr_level + log_pr_word;

      // Keep for each level the log probability of the word +
      // log probability of the level.
      // Use these values to sample the new level.
      log_pr.at(j) = log_value;
    }

    // Sample the new level and update.
    int new_level = Utils::SampleFromLogPr(log_pr);
    document->getMutablePathTopic(new_level)->updateWordCount(word->getId(), 1);
    word->setLevel(new_level);
    document->updateLevelCounts(new_level, 1);
  }
}

void DocumentUtils::PermuteWords(Document* document) {
  int size = document->getWords();
  vector<Word> permuted_words;

  // Permute the values in perm.
  // These values correspond to the indices of the words in the
  // word vector of the document.
  gsl_permutation* perm = gsl_permutation_calloc(size);
  Utils::Shuffle(perm, size);
  int perm_size = perm->size;
  assert(size == perm_size);

  for (int i = 0; i < perm_size; i++) {
    permuted_words.push_back(*document->getMutableWord(perm->data[i]));
  }

  document->setWords(permuted_words);

  gsl_permutation_free(perm);
}

// =======================================================================
// DocumentTreeUtils
// =======================================================================

void DocumentTreeUtils::SampleDocumentPath(
    Tree* tree,
    Document* document,
    bool remove,
    int start_level) {

  // Remove the document from the path at the specified level (start_level).
  if (remove) {
    RemoveDocumentFromPath(tree, document, start_level);
  }

  double log_sum = 0.0;

  // Path probabilities.
  vector<double> path_pr(tree->getDepth(), 0.0);
  Topic* start_topic = document->getMutablePathTopic(start_level);

  // Compute path probabilities starting at the topic and
  // visiting all its children depth-first.
  DocumentTopicUtils::ProbabilitiesDfs(
      start_topic, document, &log_sum,
      &path_pr, start_level);

  // Sample node and fill tree.
  Topic* topic = TopicUtils::SampleTopic(
      document->getMutablePathTopic(start_level), log_sum);
  topic = TopicUtils::AddTopic(topic);

  // Add path to the document, start at the specified start level.
  DocumentTopicUtils::AddPathToDocument(topic, document, start_level);
}

void DocumentTreeUtils::RemoveDocumentFromPath(
      Tree* tree,
      Document* document,
      int start_level) {
  UpdateTreeFromDocument(document, -1, start_level);
  Topic* topic = document->getMutablePathTopic(tree->getDepth() - 1);
  TopicUtils::Prune(topic);
}

void DocumentTreeUtils::UpdateTreeFromDocument(
      Document* document,
      int update,
      int start_level) {
  // The depth of the tree.
  int depth = document->getMutablePathTopic(0)->getMutableTree()->getDepth();

  // Update the word count for all the words in the document.
  for (int i = 0; i < document->getWords(); i++) {
    int level = document->getMutableWord(i)->getLevel();
    if (level > start_level) {
      document->getMutablePathTopic(level)->updateWordCount(
          document->getMutableWord(i)->getId(), update);
    }
  }

  // Update the document count for the topics in the path.
  for (int i = start_level + 1; i < depth; i++) {
    document->getMutablePathTopic(i)->incDocumentNo(update);
  }
}

// =======================================================================
// DocumentTopicUtils
// =======================================================================

void DocumentTopicUtils::AddPathToDocument(
    Topic* topic,
    Document* document,
    int start_level) {
  int depth = topic->getMutableTree()->getDepth();
  int level = depth - 1;

  // Set the path for this document.
  do {
    document->setPathTopic(level, topic);
    topic = topic->getMutableParent();
    level--;
  } while (level >= start_level);

  // Update the topics from the document.
  DocumentTreeUtils::UpdateTreeFromDocument(document, 1, start_level);
}

void DocumentTopicUtils::ProbabilitiesDfs(
    Topic* topic,
    Document* document,
    double* log_sum,
    vector<double>* path_pr,
    int start_level) {
  int level = topic->getLevel();
  int depth = topic->getMutableTree()->getDepth();

  double eta = topic->getMutableTree()->getEta(topic->getLevel());
  int term_no = topic->getCorpusWordNo();

  // Set path probability for current topic node in the tree.
  path_pr->at(level)  = LogGammaRatio(document, topic, level, eta, term_no);

  double parent_log_val = 0.0;
  if (level > start_level) {
    parent_log_val = log(topic->getMutableParent()->getDocumentNo() +
                topic->getMutableParent()->getScaling());
    path_pr->at(level) += log(topic->getDocumentNo()) - parent_log_val;
  }

  // Set path probabilities for levels below this topic.
  if (level < depth - 1) {
    for (int i = level + 1; i < depth; i++) {
      eta = topic->getMutableTree()->getEta(i);
      path_pr->at(i) = LogGammaRatio(document, NULL, i, eta, term_no);
    }

    path_pr->at(level+1) += log(topic->getScaling());
    path_pr->at(level+1) -= log(topic->getDocumentNo() + topic->getScaling());
  }

  // Set probability for the topic.
  double probability = 0.0;
  for (int i = start_level; i < depth; i++) {
    probability += path_pr->at(i);
  }
  topic->setProbability(probability);

  // Update the normalizing constant.
  if (level == start_level) {
    *log_sum = probability;
  } else {
    *log_sum = Utils::LogSum(*log_sum, probability);
  }

  // Recursive call for the children.
  for (int i = 0; i < topic->getChildren(); i++) {
    ProbabilitiesDfs(
        topic->getMutableChild(i), document, log_sum, path_pr, start_level);
  }
}

double DocumentTopicUtils::LogGammaRatio(
      Document* document,
      Topic* topic,
      int level,
      double eta,
      int term_no) {
  vector<int> count(term_no, 0);
  double result = 0.0;

  // Initialize the count for each word.
  for (int i = 0; i < document->getWords(); i++) {
    count[document->getMutableWord(i)->getId()] = 0;
  }

  for (int i = 0; i < document->getWords(); i++) {
    if (document->getMutableWord(i)->getLevel() == level) {
      count[document->getMutableWord(i)->getId()]++;
    }
  }

  int word_no = 0;

  // Topic can be NULL, in which case the result doesn't include the
  // Word count for the topic.
  if (topic != NULL) {
    word_no = topic->getTopicWordNo();
  }

  double value = word_no + term_no * eta;
  result = gsl_sf_lngamma(value);
  value = word_no + document->getLevelCounts(level) + term_no * eta;
  result -= gsl_sf_lngamma(value);

  for (int i = 0; i < document->getWords(); i++) {
    int word_id = document->getMutableWord(i)->getId();
    if (count[word_id] > 0) {
      int word_count = 0;
      if (topic != NULL) {
        word_count = topic->getWordCount(word_id);
      }
      result -= gsl_sf_lngamma(word_count + eta);
      result += gsl_sf_lngamma(word_count + count[word_id] + eta);
      count[word_id] = 0;
    }
  }

  return result;
}

}  // namespace hlda



