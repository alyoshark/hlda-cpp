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


#ifndef TREE_H_
#define TREE_H_

#include <vector>

#include "topic.h"
#include "utils.h"

namespace hlda {

class Topic;

// The tree representing the hierarchy of topics.
// A tree has a certain depth, a number of scaling parameters,
// the topic Dirichlet parameter and the next topic id.
// It keeps a pointer to the root topic.
//
class Tree {
 public:
  Tree();
  Tree(int depth, int word_no, const vector<double>& eta,
       double scaling_shape, double scaling_scale);
  ~Tree();

  Tree(const Tree& from);
  Tree& operator=(const Tree& from);

  void setRootTopic(Topic* root_topic) { root_topic_ = root_topic; }
  Topic* getMutableRootTopic() { return root_topic_; }

  int getNextId() const { return next_id_; }
  void incNextId(int val) { next_id_ += val; }

  double getEta(int i) const { return eta_[i]; }
  int getDepth() const { return depth_; }

  void setEta(int i, double value) { eta_[i] = value; }

  double getScalingShape() const { return scaling_shape_; }
  double getScalingScale() const { return scaling_scale_; }

 private:
  // Depth of the tree.
  int depth_;

  // Topic Dirichlet parameter.
  // The Eta parameter represents the expected variance of the
  // underlying topics.
  vector<double> eta_;

  // Scaling shape parameter for the G prior.
  double scaling_shape_;

  // Scaling scale parameter for the G prior.
  double scaling_scale_;

  // Root topic of the tree.
  // Each node of the tree is associated with a topic.
  Topic* root_topic_;

  // The next id for the following topic.
  int next_id_;
};

// This class provides functionality for updating the Eta parameter.
class TreeUtils {
 public:
  // Updates the Eta parameter.
  // The new Eta score is based on Gaussian random variates.
  // Repeat REP_NO_ETA number of times.
  static void UpdateEta(Tree* tree);
};

}  // namespace hlda

#endif  // TREE_H_
