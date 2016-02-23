#include <lamtram/ensemble-decoder.h>
#include <lamtram/macros.h>
#include <cnn/nodes.h>
#include <boost/range/irange.hpp>
#include <cfloat>

using namespace lamtram;
using namespace std;
using namespace cnn::expr;


EnsembleDecoder::EnsembleDecoder(const vector<EncoderDecoderPtr> & encdecs, const vector<EncoderAttentionalPtr> & encatts, const vector<NeuralLMPtr> & lms)
      : encdecs_(encdecs), encatts_(encatts), word_pen_(0.0), size_limit_(2000), beam_size_(1), ensemble_operation_("sum") {
  if(encdecs.size() + encatts.size() + lms.size() == 0)
    THROW_ERROR("Cannot decode with no models!");
  for(auto & ed : encdecs) {
    lms_.push_back(ed->GetDecoderPtr());
    externs_.push_back(NULL);
  }
  for(auto & ed : encatts) {
    lms_.push_back(ed->GetDecoderPtr());
    externs_.push_back(ed->GetExternCalcPtr());
  }
  for(auto & lm : lms) { 
    lms_.push_back(lm);
    externs_.push_back(NULL);
  }
  unk_id_ = lms_[0]->GetUnkId();
}

vector<vector<Expression> > EnsembleDecoder::GetInitialStates(const Sentence & sent_src, cnn::ComputationGraph & cg) {
  vector<vector<Expression> > last_state(encdecs_.size() + encatts_.size() + lms_.size());
  int id = 0;
  for(auto & tm : encdecs_)
    last_state[id++] = tm->GetEncodedState(sent_src, false, cg);
  for(int i : boost::irange(0, (int)encatts_.size()))
    externs_[id + i]->InitializeSentence(sent_src, false, cg);
  return last_state;
}

Expression EnsembleDecoder::EnsembleProbs(const std::vector<Expression> & in, cnn::ComputationGraph & cg) {
  if(in.size() == 1) return in[0];
  return average(in);
}

Expression EnsembleDecoder::EnsembleLogProbs(const std::vector<Expression> & in, cnn::ComputationGraph & cg) {
  if(in.size() == 1) return in[0];
  Expression i_average = average(in);
  return log_softmax({i_average});
}

namespace lamtram {

template<>
Expression EnsembleDecoder::EnsembleSingleProb(const std::vector<Expression> & in, const Sentence & sent, int t, cnn::ComputationGraph & cg) {
  // cout << "word: " << sent[t] << endl;
  if(in.size() == 1)
    return pick({in[0]}, sent[t]);
  std::vector<Expression> i_probs(in);
  for(int i : boost::irange(0, (int)in.size()))
    i_probs[i] = pick({in[i]}, sent[t]);
  return average(i_probs);
}

template<>
Expression EnsembleDecoder::EnsembleSingleLogProb(const std::vector<Expression> & in, const Sentence & sent, int t, cnn::ComputationGraph & cg) {
  if(in.size() == 1)
    return pick({in[0]}, sent[t]);
  Expression i_average = average(in);
  Expression i_softmax = log_softmax({i_average});
  return pick({i_softmax}, sent[t]);
}

}


inline void CreateWordsAndMask(const vector<Sentence> & sents, int t, bool inverse_mask, vector<unsigned> & words, vector<float> & mask) {
  words.resize(sents.size()); mask.resize(0);
  for(size_t i = 0; i < sents.size(); i++) {
    if((int)sents[i].size() <= t) {
      if(!mask.size()) mask.resize(sents.size(), inverse_mask ? 0 : 1);
      mask[i] = inverse_mask ? 1 : 0;
      words[i] = 0;
    } else {
      words[i] = sents[i][t];
    }
  }
}


namespace lamtram {
template<>
Expression EnsembleDecoder::EnsembleSingleProb(const std::vector<Expression> & in, const vector<Sentence> & sents, int t, cnn::ComputationGraph & cg) {
  vector<unsigned> words; vector<float> mask;
  CreateWordsAndMask(sents, t, false, words, mask);
  // cout << "words:"; for(auto w : words) cout << " " << w; cout << endl;
  Expression ret;
  if(in.size() == 1) {
    ret = pick({in[0]}, words);
  } else {
    std::vector<Expression> i_probs(in);
    for(int i : boost::irange(0, (int)in.size()))
      i_probs[i] = pick({in[i]}, words);
    ret = average(i_probs);
  }
  if(mask.size()) {
    // cout << "mask:"; for(auto w : mask) cout << " " << w; cout << endl;
    ret = pow(ret, input(cg, cnn::Dim({1}, sents.size()), mask));
  }
  return ret;
}

template<>
Expression EnsembleDecoder::EnsembleSingleLogProb(const std::vector<Expression> & in, const vector<Sentence> & sents, int t, cnn::ComputationGraph & cg) {
  vector<unsigned> words; vector<float> mask;
  CreateWordsAndMask(sents, t, true, words, mask);
  Expression ret;
  if(in.size() == 1) {
    ret = pick({in[0]}, words);
  } else {
    Expression i_average = average(in);
    Expression i_softmax = log_softmax({i_average});
    ret = pick({i_softmax}, words);
  }
  if(mask.size())
    ret = ret * input(cg, cnn::Dim({1}, sents.size()), mask);
  return ret;
}

template <>
void EnsembleDecoder::AddLik<Sentence,LLStats>(const Sentence & sent, const cnn::expr::Expression & exp, LLStats & ll) {
  ll.lik_ += as_scalar(exp.value());
  ll.words_ += sent.size();
  for(unsigned t = 0; t < sent.size(); t++)
    if(sent[t] == unk_id_)
      ++ll.unk_;
}
template <>
void EnsembleDecoder::AddLik<vector<Sentence>,vector<LLStats> >(const vector<Sentence> & sent, const cnn::expr::Expression & exp, std::vector<LLStats> & ll) {
  vector<float> ret = as_vector(exp.value());
  assert(ret.size() == ll.size());
  for(size_t i = 0; i < ret.size(); i++) {
    ll[i].lik_ += ret[i];
    ll[i].words_ += sent[i].size();
    for(unsigned t = 0; t < sent[i].size(); t++)
      if(sent[i][t] == unk_id_)
        ++ll[i].unk_;
  }
}
}

inline int MaxLen(const Sentence & sent) { return sent.size(); }
inline int MaxLen(const vector<Sentence> & sent) {
  size_t val = 0;
  for(const auto & s : sent) { val = max(val, s.size()); }
  return val;
}

inline int GetWord(const vector<Sentence> & vec, int t) { return vec[0][t]; }
inline int GetWord(const Sentence & vec, int t) { return vec[t]; }

template <class Sent, class Stat>
void EnsembleDecoder::CalcSentLL(const Sentence & sent_src, const Sent & sent_trg, Stat & ll) {
  // First initialize states and do encoding as necessary
  cnn::ComputationGraph cg;
  for(auto & tm : encdecs_) tm->NewGraph(cg);
  for(auto & tm : encatts_) tm->NewGraph(cg);
  for(auto & lm : lms_) lm->NewGraph(cg);
  vector<vector<Expression> > last_state = GetInitialStates(sent_src, cg), next_state(lms_.size());
  // Go through and collect the values
  vector<Expression> errs, aligns;
  int max_len = MaxLen(sent_trg);
  for(int t : boost::irange(0, max_len)) {
    GlobalVars::curr_word = GetWord(sent_trg, t);
    // Perform the forward step on all models
    vector<Expression> i_sms;
    for(int j : boost::irange(0, (int)lms_.size()))
      i_sms.push_back(lms_[j]->Forward<Sent>(sent_trg, t, externs_[j].get(), ensemble_operation_ == "logsum", last_state[j], next_state[j], cg, aligns));
    // Ensemble the probabilities and calculate the likelihood
    Expression i_logprob;
    if(ensemble_operation_ == "sum") {
      i_logprob = EnsembleSingleProb(i_sms, sent_trg, t, cg);
      i_logprob = log({i_logprob});
    } else if(ensemble_operation_ == "logsum") {
      i_logprob = EnsembleSingleLogProb(i_sms, sent_trg, t, cg);
    } else {
      THROW_ERROR("Bad ensembling operation: " << ensemble_operation_ << endl);
    }
    // Pick the error and add to the vector
    // cerr << "i_logprob @ " << t << " == " << sent_trg[t] << ": " << as_scalar(i_logprob.value()) << endl;
    errs.push_back(i_logprob);
    last_state = next_state;

  }
  Expression err = sum(errs);
  cg.forward();
  AddLik(sent_trg, err, ll);
}                    

template
void EnsembleDecoder::CalcSentLL<Sentence,LLStats>(const Sentence & sent_src, const Sentence & sent_trg, LLStats & ll);
template
void EnsembleDecoder::CalcSentLL<vector<Sentence>,vector<LLStats> >(const Sentence & sent_src, const vector<Sentence> & sent_trg, vector<LLStats> & ll);

Sentence EnsembleDecoder::Generate(const Sentence & sent_src, Sentence & align) {

  align.clear();

  // First initialize states
  cnn::ComputationGraph cg;
  for(auto & tm : encdecs_) tm->NewGraph(cg);
  for(auto & tm : encatts_) tm->NewGraph(cg);
  for(auto & lm : lms_) lm->NewGraph(cg);

  // Create the initial hypothesis
  vector<vector<vector<Expression> > > last_states(beam_size_, vector<vector<Expression> >(lms_.size()));
  vector<EnsembleDecoderHypPtr> curr_beam(1, 
      EnsembleDecoderHypPtr(new EnsembleDecoderHyp(
          0.0, GetInitialStates(sent_src, cg), Sentence(), Sentence())));
  int bid;
  Expression empty_idx;

  // Perform decoding
  for(int sent_len = 0; sent_len <= size_limit_; sent_len++) {
    // This vector will hold the best IDs
    vector<tuple<cnn::real,int,int,int> > next_beam_id(beam_size_+1, tuple<cnn::real,int,int,int>(-DBL_MAX,-1,-1,-1));
    // Go through all the hypothesis IDs
    for(int hypid = 0; hypid < (int)curr_beam.size(); hypid++) {
      EnsembleDecoderHypPtr curr_hyp = curr_beam[hypid];
      const Sentence & sent = curr_beam[hypid]->GetSentence();
      if(sent_len != 0 && *sent.rbegin() == 0) continue;
      // Perform the forward step on all models
      vector<Expression> i_softmaxes, i_aligns;
      for(int j : boost::irange(0, (int)lms_.size()))
        i_softmaxes.push_back( lms_[j]->Forward(sent, sent_len, externs_[j].get(), ensemble_operation_ == "logsum", curr_hyp->GetStates()[j], last_states[hypid][j], cg, i_aligns) );
      // Ensemble and calculate the likelihood
      Expression i_softmax, i_logprob;
      if(ensemble_operation_ == "sum") {
        i_softmax = EnsembleProbs(i_softmaxes, cg);
        i_logprob = log({i_softmax});
      } else if(ensemble_operation_ == "logsum") {
        i_logprob = EnsembleLogProbs(i_softmaxes, cg);
      } else {
        THROW_ERROR("Bad ensembling operation: " << ensemble_operation_ << endl);
      }
      // Add the word penalty
      vector<cnn::real> softmax = as_vector(cg.incremental_forward());
      softmax[0] -= word_pen_;
      // Find the best aligned source, if any alignments exists
      WordId best_align = -1;
      if(i_aligns.size() != 0) {
        Expression i_align_sum = sum(i_aligns);
        vector<cnn::real> align = as_vector(cg.incremental_forward());
        best_align = 0;
        for(size_t aid = 0; aid < align.size(); aid++)
          if(align[aid] > align[best_align])
            best_align = aid;
      }
      // Find the best IDs
      for(int wid = 0; wid < (int)softmax.size(); wid++) {
        cnn::real my_score = curr_hyp->GetScore() + softmax[wid];
        for(bid = beam_size_; bid > 0 && my_score > std::get<0>(next_beam_id[bid-1]); bid--)
          next_beam_id[bid] = next_beam_id[bid-1];
        next_beam_id[bid] = tuple<cnn::real,int,int,int>(my_score,hypid,wid,best_align);
      }
    }
    // Create the new hypotheses
    vector<EnsembleDecoderHypPtr> next_beam;
    for(int i = 0; i < beam_size_; i++) {
      int hypid = std::get<1>(next_beam_id[i]);
      int wid = std::get<2>(next_beam_id[i]);
      int aid = std::get<3>(next_beam_id[i]);
      if(hypid == -1) break;
      Sentence next_sent = curr_beam[hypid]->GetSentence();
      next_sent.push_back(wid);
      Sentence next_align = curr_beam[hypid]->GetAlignment();
      next_align.push_back(aid);
      if(i == 0 && wid == 0) {
        for(auto a : next_align)
          align.push_back(a);
        if(align.size() != next_sent.size()) THROW_ERROR("align.size() == " << align.size() << ", next_sent.size() == " << next_sent.size());
        return next_sent;
      }
      next_beam.push_back(EnsembleDecoderHypPtr(new EnsembleDecoderHyp(
          std::get<0>(next_beam_id[i]), last_states[hypid], next_sent, next_align)));
    }
    curr_beam = next_beam;
  }
  cerr << "WARNING: Generated sentence size exceeded " << size_limit_ << ". Returning empty sentence." << endl;
  return Sentence();
}
