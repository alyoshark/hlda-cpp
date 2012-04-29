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

#include "tree.h"

#define REP_NO_ETA 100
#define ETA_STDEV 0.005
#define GAM_STDEV 0.005

namespace hlda {

// =======================================================================
// Tree
// =======================================================================

Tree::Tree()
    : depth_(0),
      scaling_shape_(0.0),
      scaling_scale_(0.0),
      root_topic_(NULL),
      next_id_(0) {
}

Tree::Tree(int depth,
           int word_no,
           const vector<double>& eta,
           double scaling_shape,
           double scaling_scale)
    : depth_(depth),
      eta_(eta),
      scaling_shape_(scaling_shape),
      scaling_scale_(scaling_scale),
      next_id_(0) {
  root_topic_ = new Topic(0, NULL, this, word_no);
}

Tree::Tree(const Tree& from)
    : depth_(from.depth_),
      eta_(from.eta_),
      scaling_shape_(from.scaling_shape_),
      scaling_scale_(from.scaling_scale_),
      next_id_(from.next_id_) {
  // Create a new topic.
  root_topic_ = new Topic(*from.root_topic_, NULL, this);
}

Tree& Tree::operator =(const Tree& from) {
  if (this == &from) return *this;
  depth_ = from.depth_;
  eta_ = from.eta_;
  scaling_shape_ = from.scaling_shape_;
  scaling_scale_ = from.scaling_scale_;
  next_id_ = from.next_id_;
  root_topic_ = new Topic(*from.root_topic_, NULL, this);

  return *this;
}

Tree::~Tree() {
  delete root_topic_;
}

// =======================================================================
// TreeUtils
// =======================================================================

void TreeUtils::UpdateEta(Tree* tree) {
  // Get number of Eta values, corresponding to the depth of the tree.
  int eta_depth = tree->getDepth();

  // Repeat a number of times.
  for (int i = 0; i < REP_NO_ETA; i++) {
    // The Eta score for the root topic.
    double root_eta_score = TopicUtils::EtaScore(tree->getMutableRootTopic());

    // The Eta scores for the levels in the tree.
    for (int level = 0; level < eta_depth; level++) {
      double old_eta = tree->getEta(level);
      // A new Eta score based on Gaussian random variates.
      double new_eta = Utils::RandGauss(old_eta, ETA_STDEV);

      // Decide if to keep the new Eta value.
      if (new_eta > 0) {
        tree->setEta(level, new_eta);
        double new_eta_score = TopicUtils::EtaScore(
            tree->getMutableRootTopic());
        double rand = Utils::RandNo();
        if (rand > exp(new_eta_score - root_eta_score)) {
          tree->setEta(level, old_eta);
        } else {
          root_eta_score = new_eta_score;
        }
      }
    }
  }
}

}  // namespace hlda


