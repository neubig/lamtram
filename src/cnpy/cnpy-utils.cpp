#include <cnpy/cnpy-utils.h>
#include <cnn/model.h>

using namespace std;
using namespace lamtram;

void CnpyUtils::copyWeight(const string & name,cnpy::npz_t & model,cnn::LookupParameter & target) {
    auto it = model.find(name);
    if(it != model.end()) {
        int vocSize = it->second.shape[0];
        int hSize = it->second.shape[1];
        float * data = ((float*)it->second.data);
        for (int i = 0; i < vocSize && i < target.get()->all_dim.cols(); i++) {
            vector<float> s(&data[i * hSize],&data[(i+1)*hSize]);
            target.initialize(i,s);
        }
    }else {
        std::cerr << "Missing " << name << " in npz model" << std::endl;
    }

}

void CnpyUtils::copyGRUWeight(const string & prefix,cnpy::npz_t & model,BuilderPtr target) {

    float * data = NULL;
    vector<float> f1;
    vector<float> f2;

    pair<int,int> size = getData(prefix+"W",model,data);
    splitData(data, f1, f2, size);
    target->init_parameters(0,0,f2);
    target->init_parameters(0,3,f1);

    size = getData(prefix+"U",model,data);
    splitData(data, f1, f2, size);
    target->init_parameters(0,1,f2);
    target->init_parameters(0,4,f1);

    size = getData(prefix+"b",model,data);
    assert(size.first % 2 == 0);
    target->init_parameters(0,2,vector<float>(&data[size.first/2],&data[size.first]));
    target->init_parameters(0,5,vector<float>(&data[0],&data[size.first/2]));

    size = getData(prefix+"Wx",model,data);
    target->init_parameters(0,6,vector<float>(&data[0],&data[size.first*size.second]));
    
    size = getData(prefix+"Ux",model,data);
    target->init_parameters(0,7,vector<float>(&data[0],&data[size.first*size.second]));
    
    size = getData(prefix+"bx",model,data);
    target->init_parameters(0,8,vector<float>(&data[0],&data[size.first]));

    
    
}
void CnpyUtils::copyGRUCondWeight(const string & prefix,cnpy::npz_t & model,BuilderPtr target) {

    copyGRUWeight(prefix,model,target);
    float * data = NULL;
    vector<float> f1;
    vector<float> f2;

    pair<int,int> size = getData(prefix+"Wc",model,data);
    splitData(data, f1, f2, size);
    target->init_parameters(0,9+0,f2);
    target->init_parameters(0,9+3,f1);

    size = getData(prefix+"U_nl",model,data);
    splitData(data, f1, f2, size);
    target->init_parameters(0,9+1,f2);
    target->init_parameters(0,9+4,f1);

    size = getData(prefix+"b_nl",model,data);
    assert(size.first % 2 == 0);
    target->init_parameters(0,9+2,vector<float>(&data[size.first/2],&data[size.first]));
    target->init_parameters(0,9+5,vector<float>(&data[0],&data[size.first/2]));

    size = getData(prefix+"Wcx",model,data);
    target->init_parameters(0,9+6,vector<float>(&data[0],&data[size.first*size.second]));
    
    size = getData(prefix+"Ux_nl",model,data);
    target->init_parameters(0,9+7,vector<float>(&data[0],&data[size.first*size.second]));
    
    size = getData(prefix+"bx_nl",model,data);
    target->init_parameters(0,9+8,vector<float>(&data[0],&data[size.first]));

    
    
}

void CnpyUtils::copyAttentionWeight(const string & prefix,cnpy::npz_t & model,ExternAttentionalPtr target) {


    float * data = NULL;
    pair<int,int> size = getData(prefix+"Wc_att",model,data);
    TensorTools::SetElements(target->p_ehid_h_W_.get()->values,vector<float>(&data[0],&data[size.first*size.second]));

    size = getData(prefix+"W_comb_att",model,data);
    TensorTools::SetElements(target->p_ehid_state_W_.get()->values,vector<float>(&data[0],&data[size.first*size.second]));

    size = getData(prefix+"b_att",model,data);
    TensorTools::SetElements(target->p_ehid_h_b_.get()->values,vector<float>(&data[0],&data[size.first]));

    size = getData(prefix+"U_att",model,data);
    TensorTools::SetElements(target->p_e_ehid_W_.get()->values,vector<float>(&data[0],&data[size.first*size.second]));

    size = getData(prefix+"c_tt",model,data);
    TensorTools::SetElements(target->p_e_ehid_b_.get()->values,vector<float>(&data[0],&data[1]));


}

void CnpyUtils::copySoftmaxWeight(const string & prefix,cnpy::npz_t & model,SoftmaxPtr target,int vocSize) {


    SoftmaxMultiLayer * hidden = (SoftmaxMultiLayer *) target.get();
    SoftmaxFull * last = (SoftmaxFull *) hidden->softmax_.get();

    float * data = NULL;
    pair<int,int> size = getData(prefix+"logit_W",model,data);
    vector<float> w(size.first * vocSize);
    for(int i = 0; i < size.first; i++) {
        for(int j = 0; j < vocSize && i < size.second; i++) {
            data[i*size.second + j] = w[i*vocSize + j];
        } 
    }
    TensorTools::SetElements(last->p_sm_W_.get()->values,w);

    size = getData(prefix+"logit_b",model,data);
    vector<float> b(vocSize);
    for(int i = 0; i < vocSize && i < size.first; i++) {
            data[i] = b[i];
    }
    TensorTools::SetElements(last->p_sm_b_.get()->values,b);

    float * ctxData = NULL;
    float * lstmData = NULL;
    float * prevData = NULL;
    
    pair<int,int> ctxSize = getData(prefix+"logit_ctx_W",model,ctxData);
    pair<int,int> lstmSize = getData(prefix+"logit_lstm_W",model,lstmData);
    pair<int,int> prevSize = getData(prefix+"logit_prev_W",model,prevData);
    
    assert(ctxSize.second == lstmSize.second);
    assert(ctxSize.second == prevSize.second);
    vector<float> d((ctxSize.first + lstmSize.first + prevSize.first)*ctxSize.second);
    int s = ctxSize.second + lstmSize.second + prevSize.second;
    int k= 0;
    for(int i = 0; i < prevSize.first;i++) {
        for(int j = 0; j < prevSize.second; j++) {
           d[k*prevSize.second+j] = prevData[i*prevSize.second+j];
        }
        k++;
    }
    for(int i = 0; i < lstmSize.first;i++) {
        for(int j = 0; j < lstmSize.second; j++) {
           d[k*lstmSize.second+j] = lstmData[i*lstmSize.second+j];
        }
        k++;
    }
    for(int i = 0; i < ctxSize.first;i++) {
        for(int j = 0; j < ctxSize.second; j++) {
           d[k*ctxSize.second+j] = ctxData[i*ctxSize.first+j];
        }
        k++;
    }

    TensorTools::SetElements(hidden->p_sm_W_.get()->values,d);

    ctxSize = getData(prefix+"logit_ctx_b",model,ctxData);
    lstmSize = getData(prefix+"logit_lstm_b",model,lstmData);
    prevSize = getData(prefix+"logit_prev_b",model,prevData);

    assert(ctxSize.first == lstmSize.first);
    assert(ctxSize.first == prevSize.first);
    d.resize(ctxSize.first);
    for(int i = 0; i < prevSize.first; i++) {
        d[k] = prevData[i] + lstmData[i] + ctxData[i];
        k++;
    }

    TensorTools::SetElements(hidden->p_sm_b_.get()->values,d);

}



void CnpyUtils::splitData(const float * data, vector<float> & f1, vector<float> & f2, const pair<int,int> & size) {
    cout << "Size data:" << size.first << " " << size.second << endl;
    assert(size.second %2 == 0);
    int half = size.second/2;
    f1.resize(size.first*half);
    f2.resize(size.first*half);
    for (int i = 0; i < size.first; i++) {
        for(int j = 0; j < size.second; j++) {
            if(j < half) {
                f1[i*half+j] = data[i*size.second+j];
            }else {
                f2[i*half+(j-half)] = data[i*size.second+j];
            }
        }
    }
    
    
}
pair<int,int> CnpyUtils::getData(const string & name, cnpy::npz_t & model, float * & data) {
    cout << "Name:" << name << endl;
    auto it = model.find(name);
    if(it != model.end()) {
        data = ((float*)it->second.data);
        cout << "Nematus size:" << it->second.shape[0] << " " << it->second.shape[1] << endl;
        return make_pair(it->second.shape[0],it->second.shape[1]);
    }else {
        std::cerr << "Missing " << name << " in npz model" << std::endl;
        exit(-1);
    }

}