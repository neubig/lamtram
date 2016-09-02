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
    target->init_parameters(0,0,f2);
    target->init_parameters(0,3,f1);

    size = getData(prefix+"U_nl",model,data);
    splitData(data, f1, f2, size);
    target->init_parameters(0,1,f2);
    target->init_parameters(0,4,f1);

    size = getData(prefix+"b_nl",model,data);
    assert(size.first % 2 == 0);
    target->init_parameters(0,2,vector<float>(&data[size.first/2],&data[size.first]));
    target->init_parameters(0,5,vector<float>(&data[0],&data[size.first/2]));

    size = getData(prefix+"Wcx",model,data);
    target->init_parameters(0,6,vector<float>(&data[0],&data[size.first*size.second]));
    
    size = getData(prefix+"Ux_nl",model,data);
    target->init_parameters(0,7,vector<float>(&data[0],&data[size.first*size.second]));
    
    size = getData(prefix+"bx_nl",model,data);
    target->init_parameters(0,8,vector<float>(&data[0],&data[size.first]));

    
    
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
        return make_pair(it->second.shape[0],it->second.shape[1]);
    }else {
        std::cerr << "Missing " << name << " in npz model" << std::endl;
        exit(-1);
    }

}