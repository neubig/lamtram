#pragma once

#include <vector>
#include <lamtram/sentence.h>

// A training target, where:
// * first is a dense vector of distributions
// * second is a sparse vector of distributions
typedef std::pair<std::vector<float>, std::vector<std::pair<int, float> > > DistTarget;
typedef std::pair<int, std::vector<std::pair<int, float> > > IndexedDistTarget;

// A training context, where:
// * first is a set of dense features 
// * second is a set of word ids
typedef std::pair<std::vector<float>, std::vector<lamtram::WordId> > AggregateContext;
typedef std::pair<int, std::vector<lamtram::WordId> > IndexedAggregateContext;

// A set of aggregate training instances
typedef std::pair<AggregateContext, std::vector<std::pair<DistTarget, int> > > AggregateInstance;
typedef std::pair<IndexedAggregateContext, std::vector<std::pair<IndexedDistTarget, int> > > IndexedAggregateInstance;

// Sentence-based data types
// * First is a sentence
// * Second is a vector of context/distribution-target pairs
typedef std::pair<lamtram::Sentence, std::vector<std::pair<std::vector<float>, DistTarget> > > SentenceInstance;
typedef std::pair<lamtram::Sentence, std::vector<std::pair<int, IndexedDistTarget> > > IndexedSentenceInstance;

// Contains all the data and various bookkeeping variables for 
template <class T>
struct TrainingData {
  TrainingData(const std::string & _name = "") : name(_name), data(), all_words(0), unk_words(0), batch_ranges(), eval_ranges(), curr_order() { }
  
  // Interfaces to vector
  size_t size() const { return data.size(); }
  void resize(size_t s) { data.resize(s); }
  typename std::vector<T>::iterator begin() { return data.begin(); }
  typename std::vector<T>::iterator end() { return data.end(); }
  typename std::vector<T>::const_iterator begin() const { return data.begin(); }
  typename std::vector<T>::const_iterator end() const { return data.end(); }
  void push_back(const T & val) { data.push_back(val); }
  size_t num_minibatches() { return batch_ranges.size()-1; } 

  std::string name;
  std::vector<T> data;
  int all_words, unk_words;
  std::vector<size_t> batch_ranges;
  std::vector<size_t> eval_ranges;
  std::vector<size_t> curr_order;
};

typedef TrainingData<AggregateInstance> AggregateData;
typedef TrainingData<SentenceInstance> SentenceData;
typedef TrainingData<IndexedAggregateInstance> IndexedAggregateData;
typedef TrainingData<IndexedSentenceInstance> IndexedSentenceData;
