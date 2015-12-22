#include <lamtram/ensemble-decoder.h>
#include <lamtram/macros.h>
#include <cnn/nodes.h>
#include <boost/range/irange.hpp>
#include <cfloat>

using namespace lamtram;
using namespace std;
using namespace cnn::expr;


EnsembleDecoder::EnsembleDecoder(const vector<EncoderDecoderPtr> & encdecs, const vector<EncoderAttentionalPtr> & encatts, const vector<NeuralLMPtr> & lms, int pad)
      : encdecs_(encdecs), encatts_(encatts), word_pen_(0.0), pad_(pad), size_limit_(2000), beam_size_(1), ensemble_operation_("sum") {
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
    last_state[id++] = tm->GetEncodedState(sent_src, cg);
  for(int i : boost::irange(0, (int)encatts_.size()))
    externs_[id + i]->InitializeSentence(sent_src, cg);
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

Expression EnsembleDecoder::EnsembleSingleProb(const std::vector<Expression> & in, int word, cnn::ComputationGraph & cg) {
  if(in.size() == 1)
    return pick({in[0]}, word);
  std::vector<Expression> i_probs(in);
  for(int i : boost::irange(0, (int)in.size()))
    i_probs[i] = pick({in[i]}, word);
  return average(i_probs);
}

Expression EnsembleDecoder::EnsembleSingleLogProb(const std::vector<Expression> & in, int word, cnn::ComputationGraph & cg) {
  if(in.size() == 1)
    return pick({in[0]}, word);
  Expression i_average = average(in);
  Expression i_softmax = log_softmax({i_average});
  return pick({i_softmax}, word);
}

void EnsembleDecoder::CalcSentLL(const Sentence & sent_src, const Sentence & sent_trg, LLStats & ll) {
  // First initialize states and do encoding as necessary
  cnn::ComputationGraph cg;
  for(auto & tm : encdecs_) tm->NewGraph(cg);
  for(auto & tm : encatts_) tm->NewGraph(cg);
  for(auto & lm : lms_) lm->NewGraph(cg);
  vector<vector<Expression> > last_state = GetInitialStates(sent_src, cg), next_state(lms_.size());
  // Go through and collect the values
  vector<Expression> errs, aligns;
  for(int t : boost::irange(pad_, (int)sent_trg.size())) {
    // Perform the forward step on all models
    vector<Expression> i_sms;
    for(int j : boost::irange(0, (int)lms_.size())) {
      if(ensemble_operation_ == "sum")
        i_sms.push_back(lms_[j]->Forward<cnn::Softmax>(sent_trg, t, externs_[j].get(), last_state[j], next_state[j], cg, aligns));
      else
        i_sms.push_back(lms_[j]->Forward<cnn::LogSoftmax>(sent_trg, t, externs_[j].get(), last_state[j], next_state[j], cg, aligns));
    }
    // Ensemble the probabilities and calculate the likelihood
    Expression i_logprob;
    if(ensemble_operation_ == "sum") {
      i_logprob = EnsembleSingleProb(i_sms, sent_trg[t], cg);
      i_logprob = log({i_logprob});
    } else if(ensemble_operation_ == "logsum") {
      i_logprob = EnsembleSingleLogProb(i_sms, sent_trg[t], cg);
    } else {
      THROW_ERROR("Bad ensembling operation: " << ensemble_operation_ << endl);
    }
    // Pick the error and add to the vector
    errs.push_back(i_logprob);
    last_state = next_state;
    // Add to the words
    if(sent_trg[t] == unk_id_) ll.unk_++;
    ll.words_++;
  }
  sum(errs);
  ll.lik_ += as_scalar(cg.forward());
}                    

Sentence EnsembleDecoder::Generate(const Sentence & sent_src, Sentence & align) {

  // First initialize states
  cnn::ComputationGraph cg;
  for(auto & tm : encdecs_) tm->NewGraph(cg);
  for(auto & tm : encatts_) tm->NewGraph(cg);
  for(auto & lm : lms_) lm->NewGraph(cg);

  // Create the initial hypothesis
  vector<vector<vector<Expression> > > last_states(beam_size_, vector<vector<Expression> >(lms_.size()));
  vector<EnsembleDecoderHypPtr> curr_beam(1, 
      EnsembleDecoderHypPtr(new EnsembleDecoderHyp(
          0.0, GetInitialStates(sent_src, cg), Sentence(pad_, 0))));
  int bid;
  Expression empty_idx;

  // Perform decoding
  for(int sent_len = pad_; sent_len <= size_limit_; sent_len++) {
    // This vector will hold the best IDs
    vector<tuple<cnn::real,int,int,int> > next_beam_id(beam_size_+1, tuple<cnn::real,int,int,int>(-DBL_MAX,-1,-1,-1));
    // Go through all the hypothesis IDs
    for(int hypid = 0; hypid < (int)curr_beam.size(); hypid++) {
      EnsembleDecoderHypPtr curr_hyp = curr_beam[hypid];
      const Sentence & sent = curr_beam[hypid]->GetSentence();
      if(sent_len != pad_ && *sent.rbegin() == 0) continue;
      // Perform the forward step on all models
      vector<Expression> i_softmaxes, i_aligns;
      for(int j : boost::irange(0, (int)lms_.size())) {
        if(ensemble_operation_ == "sum")
          i_softmaxes.push_back( lms_[j]->Forward<cnn::Softmax>(sent, sent_len, externs_[j].get(), curr_hyp->GetStates()[j], last_states[hypid][j], cg, i_aligns) );
        else
          i_softmaxes.push_back( lms_[j]->Forward<cnn::LogSoftmax>(sent, sent_len, externs_[j].get(), curr_hyp->GetStates()[j], last_states[hypid][j], cg, i_aligns) );
      }
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
        align = next_align;
        return next_sent;
      }
      next_beam.push_back(EnsembleDecoderHypPtr(new EnsembleDecoderHyp(
          std::get<0>(next_beam_id[i]), last_states[hypid], next_sent)));
    }
    curr_beam = next_beam;
  }
  cerr << "WARNING: Generated sentence size exceeded " << size_limit_ << ". Returning empty sentence." << endl;
  return Sentence(pad_+1, 0);
}
