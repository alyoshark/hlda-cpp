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


#include <math.h>
#include <gsl/gsl_sf.h>

#include "topic.h"

namespace hlda {

// =======================================================================
// Topic
// =======================================================================

Topic::Topic(int level, Topic* parent, Tree* tree, int corpus_word_no)
    : topic_word_no_(0),
      corpus_word_no_(corpus_word_no),
      document_no_(0),
      level_(level),
      parent_(parent),
      tree_(tree),
      probability_(0.0) {
  id_ = tree->getNextId();
  tree->incNextId(1);

  // Scaling parameter sampled from prior.
  scaling_ = tree->getScalingShape() * tree->getScalingScale();
  // Log probabilities.
  double eta = tree->getEta(level);
  double word_log_pr = log(eta) - log(eta * corpus_word_no);

  log_pr_word_ = new double[corpus_word_no];
  word_counts_ = new int[corpus_word_no];
  lgam_word_count_eta_ = new double[corpus_word_no];
  for (int i = 0; i < corpus_word_no; i++) {
    log_pr_word_[i] = word_log_pr;
    word_counts_[i] = 0;
    lgam_word_count_eta_[i] = 0.0;
  }
}

Topic::Topic(const Topic& from, Topic* parent, Tree* tree)
    : topic_word_no_(from.topic_word_no_),
      corpus_word_no_(from.corpus_word_no_),
      document_no_(from.document_no_),
      id_(from.id_),
      level_(from.level_),
      scaling_(from.scaling_),
      probability_(from.probability_) {
  parent_ = parent;
  for (int i = 0; i < from.getChildren(); i++) {
    Topic* child = new Topic(*from.children_[i], this, tree);
    children_.push_back(child);
  }
  tree_ = tree;

  log_pr_word_ = new double[corpus_word_no_];
  word_counts_ = new int[corpus_word_no_];
  lgam_word_count_eta_ = new double[corpus_word_no_];
  for (int i = 0; i < corpus_word_no_; i++) {
    word_counts_[i] = from.word_counts_[i];
    log_pr_word_[i] = from.log_pr_word_[i];
    lgam_word_count_eta_[i] = from.lgam_word_count_eta_[i];
  }
}

Topic::Topic(const Topic& from) {
}

void Topic::operator =(const Topic& from) {
}

void Topic::updateWordCount(int word_id, int update) {
  // Find the word counts for the word with word_id, and update the counts.
  word_counts_[word_id] += update;
  topic_word_no_ += update;

  // Update log probability and log gamma.
  double eta = tree_->getEta(level_);

  // Update the log probability for the word.
  double word_log_pr = log(word_counts_[word_id] + eta) -
          log(topic_word_no_ + corpus_word_no_ * eta);
  log_pr_word_[word_id] = word_log_pr;

  // Update the pre-computed Gamma function (word counts + eta) for the word.
  double lgam_word_count_eta =
      gsl_sf_lngamma(word_counts_[word_id] + eta);
  lgam_word_count_eta_[word_id] = lgam_word_count_eta;
}

Topic::~Topic() {
  int size = children_.size();
  for (int i = 0; i < size; i++) {
    delete children_[i];
  }
  delete[] word_counts_;
  delete[] log_pr_word_;
  delete[] lgam_word_count_eta_;
}

// =======================================================================
// TopicUtils
// =======================================================================

double TopicUtils::EtaScore(Topic* topic) {
  double score = 0;
  int word_count_size = topic->getCorpusWordNo();
  double eta = topic->getMutableTree()->getEta(topic->getLevel());

  // The current eta score.
  score = gsl_sf_lngamma(word_count_size * eta) -
      word_count_size * gsl_sf_lngamma(eta);

  // Update the score based on the pre-computed Gamma (word count + eta)
  for (int i = 0; i < word_count_size; i++) {
    score += topic->getLgamWordCountEta(i);
  }

  score -= gsl_sf_lngamma(topic->getTopicWordNo() + word_count_size * eta);

  // Recursive call for the children.
  for (int i = 0; i < topic->getChildren(); i++) {
    if (topic->getMutableChild(i)->getTopicWordNo() > 0) {
      score += EtaScore(topic->getMutableChild(i));
    }
  }

  return score;
}

double TopicUtils::GammaScore(Topic* topic) {
  double score = 0;

  if (topic->getChildren() > 0) {
    score -= gsl_sf_lngamma(topic->getScaling() + topic->getDocumentNo());

    for (int i = 0; i < topic->getChildren(); i++) {
      score += gsl_sf_lngamma(
          topic->getScaling() + topic->getMutableChild(i)->getDocumentNo());
      score += GammaScore(topic->getMutableChild(i));
    }
  }

  return score;
}

Topic* TopicUtils::AddTopic(Topic* parent_topic) {
  int root_level = parent_topic->getLevel();
  if (root_level < parent_topic->getMutableTree()->getDepth() - 1) {
    Topic* child_topic = AddChildTopic(parent_topic);
    return AddTopic(child_topic);
  } else {
    return parent_topic;
  }
}

Topic* TopicUtils::AddChildTopic(Topic* parent_topic) {
  // new child
  int parent_level = parent_topic->getLevel();
  int parent_word_counts = parent_topic->getCorpusWordNo();
  Topic* child = new Topic(parent_level + 1, parent_topic,
              parent_topic->getMutableTree(), parent_word_counts);
  parent_topic->addChild(child);
  return child;
}

void TopicUtils::Prune(Topic* topic) {
  if (topic->getDocumentNo() == 0) {
    Topic* parent = topic->getMutableParent();

    // Delete topic node.
    Remove(topic);
    if (parent != NULL) {
      Prune(parent);
    }
  }
}

void TopicUtils::Remove(Topic* topic) {
  for (int i = 0; i < topic->getChildren(); i++) {
    Remove(topic->getMutableChild(i));
  }

  // Update the parent tree.
  Topic* parent = topic->getMutableParent();
  for (int i = 0; i < parent->getChildren(); i++) {
    if (parent->getMutableChild(i) == topic) {
      parent->setChild(i, parent->getMutableChild(parent->getChildren() - 1));
      parent->removeLastChild();
    }
  }

  delete(topic);
}

Topic* TopicUtils::SampleTopic(Topic* root, double log_sum) {
  double sum = 0.0;
  double rand_no = Utils::RandNo();
  return SampleDfs(root, &sum, rand_no, log_sum);
}

Topic* TopicUtils::SampleDfs(
    Topic* topic, double* sum, double rand_no, double log_sum) {
  *sum += exp(topic->getProbability() - log_sum);
  if (*sum >= rand_no) {
    return topic;
  } else {
    for (int i = 0; i < topic->getChildren(); i++) {
      Topic* child = SampleDfs(
          topic->getMutableChild(i), sum, rand_no, log_sum);
      if (child != NULL) {
        return child;
      }
    }
  }
  return NULL;
}

}  // namespace hlda


