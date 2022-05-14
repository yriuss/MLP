#include <iostream>
#include <stdio.h>
#include <Eigen/Dense>
#include <random>
#include "matplotlibcpp.h"
#define RELU 0
#define LINEAR 1




//def generate_points():
//
//    x1p = np.random.uniform(size=200)
//    x2p = np.random.uniform(size=200)
//
//    y = []
//    for x1, x2 in zip(x1p, x2p):
//        y.append(
//            sigmoid(x1 + 2*x2) + 0.5 * (x1 - x2)**2 + 0.5 * np.random.standard_normal()
//        )
//    return x1p, x2p, y

double sigmoid(double x){
    return 1.0 / (1 + exp(-x));
}

Eigen::MatrixXd generate_points(uint n_samples){
    //generate random Matrix
    std::random_device rseed;
	std::mt19937 rng(rseed());
	std::uniform_real_distribution<> dist{0,1};
    std::normal_distribution<> n_dist{0,1};
    
    Eigen::MatrixXd x(n_samples, 3);

    for(int i = 0; i < n_samples; i++){
        for(int j = 0; j < 2; j++){
            x(i,j) = dist(rng);
        }
	}
    

    //y = sigmoid(x1 + 2*x2) + 0.5 * (x1 - x2)**2 + 0.5 * np.random.standard_normal()
    for(int i = 0; i < n_samples; i++){
        x(i,2) = sigmoid(x(i, 1) + 2*x(i, 2)) + 0.5 * (x(i, 1) - x(i, 2))*(x(i, 1) - x(i, 2)) + 0.5 * n_dist(rng);
	}
    //return x1p,x2p,y
    return x;
}




/*LAYER*/
class Linear{
public:
    Linear(uint d_in, uint d_out);
    void get_parameters();
    Eigen::MatrixXd forward(Eigen::MatrixXd x);
    Eigen::MatrixXd W();
    Eigen::MatrixXd b();
    void set_W(Eigen::MatrixXd x);
    void set_b(Eigen::MatrixXd x);
private:
    Eigen::MatrixXd _W;
	Eigen::MatrixXd _b;
};

void Linear::set_W(Eigen::MatrixXd x){
    _W = x;
}

void Linear::set_b(Eigen::MatrixXd x){
    _b = x;
}

Eigen::MatrixXd Linear::W(){
    return _W;
}

Eigen::MatrixXd Linear::b(){
    return _b;
}

Linear::Linear(uint d_in, uint d_out){
    
    std::random_device rseed;
	std::mt19937 rng(rseed());
	std::normal_distribution<> dist{0,std::sqrt(2./d_in)};
    //std::cout << d_in << " " << d_out << std::endl;
    _W.resize(d_in, d_out);
    _b.resize(1, d_out);

    for(int j = 0; j < d_out; j++){
		for(int i = 0; i < d_in ; i++){
			_W(i,j) = dist(rng);
		}
        _b(j) = dist(rng);
	}
    //std::cout << "W is :" <<std::endl;
    //std::cout << W << std::endl;
    //std::cout << "b is :" <<std::endl;
    //std::cout << b << std::endl;
    
}

void Linear::get_parameters(){
    std::cout << "The weights are: " << std::endl;
    std::cout << _W << std::endl;
    std::cout << "The biases are: " << std::endl;
    std::cout << _b << std::endl;
}

Eigen::MatrixXd Linear::forward(Eigen::MatrixXd x){
    return x*_W + _b;
}




/*NEURAL NETWORK*/
class NN{
public:
    NN(uint input_size, uint output_size, std::vector<uint> n_nodes, uint n_hidden_layers, std::vector<int> activations);
    Eigen::MatrixXd forward(Eigen::MatrixXd x);
    uint nlayers();
    Eigen::MatrixXd n(uint i);
    Eigen::MatrixXd W(uint i);
    Eigen::MatrixXd b(uint i);
    void set_W(Eigen::MatrixXd x, uint layer);
    void set_b(Eigen::MatrixXd x, uint layer);
private:
    std::vector<Linear> li;
    std::vector<Eigen::MatrixXd> _n;
    std::vector<int> activations;
};

void NN::set_W(Eigen::MatrixXd x, uint layer){
    li[layer].set_W(x);
}

void NN::set_b(Eigen::MatrixXd x, uint layer){
    li[layer].set_b(x);
}

uint NN::nlayers(){
    return li.size() - 1;
}

NN::NN(uint input_size, uint output_size, std::vector<uint> n_nodes, uint n_hidden_layers, std::vector<int> activations){
    li.reserve(n_hidden_layers + 1);
    this->activations = activations;
    for(int i = 0; i < n_hidden_layers + 1 ; i++){
        if(i > 0 && i < n_hidden_layers){
            li.emplace_back(Linear(n_nodes[i - 1], n_nodes[i]));
        }
        else{
            if(i == 0)
                li.emplace_back(Linear(input_size, n_nodes[i]));
            else
                li.emplace_back(Linear(n_nodes[i - 1], output_size));
        }
    }
    
}

Eigen::MatrixXd relu(Eigen::MatrixXd x){
    for(int i =0; i < x.size(); i++){
        if(x(i) > 0)
            continue;
        else
            x(i)  = 0;
    }
    return x;
}

Eigen::MatrixXd drelu(Eigen::MatrixXd x){
    for(int i =0; i < x.size(); i++){
        if(x(i) > 0)
            x(i) = 1;
        else
            x(i)  = 0;
    }
    return x;
}

Eigen::MatrixXd NN::forward(Eigen::MatrixXd x){
    _n.reserve(li.size());
    for(int i = 0; i < li.size(); i++){
        x = li[i].forward(x);
        _n[i] = x;
        if(activations[i] == RELU){
            x = relu(x);
        }else{
            if(activations[i] == LINEAR){
                continue;
            }
        }
    }
    return x;
}

Eigen::MatrixXd NN::n(uint i){
    return _n[i];
}

Eigen::MatrixXd NN::W(uint i){
    return li[i].W();
}

Eigen::MatrixXd NN::b(uint i){
    return li[i].b();
}

class Loss{
public:
    double compute(Eigen::MatrixXd y, Eigen::MatrixXd y_pred);
    void backward(NN& nn);
    Eigen::MatrixXd dLoss_db(uint i);
    Eigen::MatrixXd dLoss_dW(uint i);
private:
    std::vector<Eigen::VectorXd> s;
    Eigen::MatrixXd dloss;
    std::vector<Eigen::MatrixXd> _dLoss_dW;
    std::vector<Eigen::MatrixXd> _dLoss_db;
};

Eigen::MatrixXd Loss::dLoss_dW(uint i){
    return _dLoss_dW[i];
}

Eigen::MatrixXd Loss::dLoss_db(uint i){
    return _dLoss_db[i];
}

void Loss::backward(NN& nn){
    uint M = nn.nlayers();
    Eigen::MatrixXd dummy;
    if(s.size() < M + 1){
        s.reserve(M + 1);
        _dLoss_dW.reserve(M+1);
        _dLoss_db.reserve(M+1);
        for(int i = 0; i < M+1; i++){
            s.emplace_back(dummy);
            _dLoss_db.emplace_back(dummy);
            _dLoss_dW.emplace_back(dummy);
        }
    }

    Eigen::MatrixXd F = drelu(nn.n(M)).array().matrix().asDiagonal();
    
    s[M] = dloss;
    
    _dLoss_dW[M] = s[M];
    _dLoss_db[M] = s[M];
    
    //std::cout << nn.n(M) << std::endl;
    
    for(int m = M; m  > 1; m--){
        //backprop sensibilities
        F = drelu(nn.n(m - 1)).array().matrix().asDiagonal();
        
        s[m - 1] = F*nn.W(m)*s[m];
        
        //update weights and biases
        _dLoss_dW[m - 1] = s[m - 1]*drelu(nn.n(m - 2)).transpose();
        _dLoss_db[m - 1] = s[m - 1];
    }
    F = drelu(nn.n(0)).array().matrix().asDiagonal();
    s[0] = F*nn.W(1)*s[1];
    _dLoss_dW[0] = s[0];
    _dLoss_db[0] = s[0];

}

double Loss::compute(Eigen::MatrixXd y, Eigen::MatrixXd y_pred){
    Eigen::MatrixXd e = (y-y_pred);
    dloss = -2*e;
    return (e*e.transpose()).mean();
}

class Optimizer{
public:
    Optimizer(Loss& loss, double lr);
    void step(NN& nn);
private:
    double _lr;
    Loss* _loss;
};

Optimizer::Optimizer(Loss& loss, double lr){
    _lr = lr;
    _loss = &loss;
}

void Optimizer::step(NN& nn){
    Eigen::MatrixXd new_W, new_b;
    for(int i = 0; i < nn.nlayers() + 1; i++){
        //applying step
        new_W = nn.W(i)- _lr*_loss->dLoss_dW(i);
        new_b = nn.b(i)- _lr*_loss->dLoss_db(i);
        
        //update weights and biases
        nn.set_W(new_W, i);
        nn.set_W(new_b, i);
    }
}

class C1{
public:
    C1();
    void print();
    void add();
private:
    int var = 1;
};

C1::C1(){}

void C1::add(){
    var += 1;
}

void C1::print(){
    std::cout << var << std::endl;
}

class C2{
public:
    C2(C1& teste);
    void print();
    void add();
private:
    C1* m;
};

C2::C2(C1& teste){
    m = &teste;
    m->add();
}

void C2::add(){
    m->add();
}


namespace plt = matplotlibcpp;



void plot_data(std::vector<double> y){
	plt::plot(y);
	plt::title("Loss - MSE");
	plt::xlabel("Épocas");
	plt::ylabel("Loss");
	plt::show();
}


int main(){
    //C1 teste;
    //teste.print();
    //C2 t(teste);
    //teste.print();
    //t.add();
    //teste.print();
    //Linear teste(4,4);
    //teste.get_parameters();
    //
    //Eigen::MatrixXd x(1,4);
    //x(0,1) = 2;
    //std::cout << "x is : " << x << std::endl;
    //std::cout << "sdasdf" << teste.forward(x) << std::endl;
    //Linear teste(2,3);
    

    //std::vector<int> activations = {RELU, RELU, RELU};
    //Eigen::MatrixXd x(1,3);
    //x(0) = 0;
    //x(1) = 1;
    //x(2) = 2;
    //
    //Eigen::MatrixXd x1(1,3);
    //x1(0) = 1;
    //x1(1) = 2;
    //x1(2) = 4;
    //NN nn(3, 3, std::vector<uint> {5, 2, 6}, 3, activations);
    //std::cout << x << std::endl;
    //x = nn.forward(x);
    //std::cout<<x<<std::endl;
    //Loss loss;
    //std::cout << loss.compute(x,x1);
    //Eigen::VectorXd v(3);
    //v << 1, -2, 3;
    //Eigen::MatrixXd F = drelu(v).array().matrix().asDiagonal();
    //std::cout << F << std::endl;
    //Eigen::MatrixXd m(3,3);
    //m =  v.array().matrix().asDiagonal();
    //std::cout << v.transpose()*m << std::endl;
#define N_SAMPLES 100
    Eigen::MatrixXd data;

    data = generate_points(N_SAMPLES);

    Eigen::MatrixXd y_train = data.col(2);
    Eigen::MatrixXd X_train(N_SAMPLES, 2);
    for(int i = 0; i < 2; i++){
        X_train.col(i) = data.col(i);
    }
    //std::cout << X_train << std::endl;
#define N_EPOCHS 100
    uint input_size = 2;
    uint output_size = 1;
    std::vector<uint> n_nodes;
    n_nodes.push_back(4);
    uint n_hidden_layers = 1;
    std::vector<int> activations;
    activations.push_back(RELU);
    activations.push_back(LINEAR);
    NN nn(input_size, output_size, n_nodes, n_hidden_layers, activations);
    
    Eigen::MatrixXd y_pred;
    Loss loss_func;
    double lr = 0.001;
    Optimizer optimizer(loss_func ,lr);
    double loss = 0;

    std::vector<double> vloss;

    vloss.reserve(N_EPOCHS);
    for(int epoch = 0; epoch < N_EPOCHS; epoch++){
        for(int i = 0; i < N_SAMPLES; i++){
            y_pred = nn.forward(X_train.row(i));
            
            loss += loss_func.compute(y_train.row(i), y_pred);
            //std::cout << loss << std::endl; exit(-1);
            loss_func.backward(nn);
            optimizer.step(nn);
        }
        std::cout << "loss in epoch " << epoch << " is: " << loss/N_SAMPLES << std::endl;
        vloss.emplace_back(loss/N_SAMPLES);
        loss = 0;
    }
    plot_data(vloss);

    return 0;
}