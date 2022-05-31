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

Eigen::MatrixXd generate_points(uint TRAIN_SAMPLES){
    //generate random Matrix
    std::random_device rseed;
	std::mt19937 rng(rseed());
	std::uniform_real_distribution<> dist{0,1};
    std::normal_distribution<> n_dist{0,1};
    
    Eigen::MatrixXd x(TRAIN_SAMPLES, 3);

    for(int i = 0; i < TRAIN_SAMPLES; i++){
        for(int j = 0; j < 2; j++){
            x(i,j) = dist(rng);
        }
	}
    

    //y = sigmoid(x1 + 2*x2) + 0.5 * (x1 - x2)**2 + 0.5 * np.random.standard_normal()
    for(int i = 0; i < TRAIN_SAMPLES; i++){
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
    _W.resize(d_out, d_in);
    _b.resize(d_out, 1);
    for(int i = 0; i < d_out; i++){
		for(int j = 0; j < d_in ; j++){
			_W(i,j) = dist(rng);
		}
        _b(i) = dist(rng);
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
    return x*_W.transpose() + _b.transpose();
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
    Eigen::MatrixXd input();
private:
    std::vector<Linear> li;
    std::vector<Eigen::MatrixXd> _n;
    std::vector<int> activations;
    Eigen::MatrixXd _input;
};


Eigen::MatrixXd NN::input(){
    return _input;
}

void NN::set_W(Eigen::MatrixXd x, uint layer){
    li[layer].set_W(x);
    //if(layer == 1){
    //    std::cout << x.cols() << " " << x.rows() << std::endl;
    //}
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
            //std::cout << "passei aqui" << std::endl;
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

Eigen::MatrixXd dlinear(Eigen::MatrixXd x){
    for(int i =0; i < x.size(); i++){
        x(i) = 1;
    }
    return x;
}

Eigen::MatrixXd NN::forward(Eigen::MatrixXd x){
    _n.reserve(li.size());
    //std::cout << x.cols() << " " << x.rows() << std::endl;
    _input = x;
    //std::cout << li.size() << std::endl;exit(-1);
    for(int i = 0; i < li.size(); i++){
        x = li[i].forward(x);
        _n.emplace_back(x);
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
    Eigen::MatrixXd dummy(1,1);
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
    //std::cout << M << std::endl; exit(-1);
    //std::cout << dlinear(nn.n(M-1)).array().matrix().asDiagonal().rows() << " " << dlinear(nn.n(M-1)).array().matrix().asDiagonal().cols() << std::endl;
    Eigen::MatrixXd F(nn.n(M-1).size(), nn.n(M-1).size());
    F.setZero();
    Eigen::MatrixXd aux;
    if(LINEAR == LINEAR){
        //std::cout << dlinear(nn.n(M-1)).asDiagonal() <<std::endl;
        aux = dlinear(nn.n(M-1));
        for(int i = 0; i < aux.size(); i++)
            F(i,i) = aux(i);
    }else{
        F = drelu(nn.n(M-1)).array().matrix().asDiagonal();
    }
    s[M] = dloss;
    
    _dLoss_dW[M] = s[M]*drelu(nn.n(M-1));
    
    // /std::cout << _dLoss_dW[M] << std::endl;exit(-1);
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
    aux = dlinear(nn.n(0));
    F.resize(nn.n(0).size(), nn.n(0).size());
    F.setZero();
    for(int i = 0; i < aux.size(); i++)
        F(i,i) = aux(i);
    s[0] = F*nn.W(1).transpose()*s[1];
    //std::cout << nn.W(1).cols() << " " << nn.W(1).rows() <<std::endl;
    //std::cout << s[1].cols() << " " << s[1].rows() <<std::endl;
    //std::cout << F.cols() << " " << F.rows() << std::endl;
    //std::cout << s[0].cols() << " " << s[0].rows() << std::endl;
    //std::cout << nn.W(0).cols() << " " << nn.W(0).rows() <<std::endl;exit(-1);
    _dLoss_dW[0] = s[0]*nn.input();
    _dLoss_db[0] = s[0];
    //std::cout << nn.input().cols() << " " << nn.input().rows() << std::endl; 
    //std::cout << s[0].cols() << " " << s[0].rows() <<std::endl;
    //std::cout << dlinear(nn.n(0)).transpose().cols() << " " << dlinear(nn.n(0)).transpose().rows() <<std::endl;
    //std::cout << _dLoss_dW[0].cols() << " " << _dLoss_dW[0].rows() << std::endl;
    //std::cout << _dLoss_db[0].cols() << " " << _dLoss_db[0].rows() << std::endl;exit(-1);

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
        //std::cout << _loss->dLoss_dW(i) << std::endl;exit(-1);
        new_b = nn.b(i)- _lr*_loss->dLoss_db(i);
        //if(i==1){
        //    std::cout << nn.W(i).cols() << " " << nn.W(i).rows() <<std::endl;
        //    std::cout << _loss->dLoss_dW(i).cols() << " " << _loss->dLoss_dW(i).rows() <<std::endl;
        //    std::cout << new_W.cols() << " " << new_W.rows() << std::endl;
        //}
        //std::cout << nn.W(i).cols() << " " << nn.W(i).rows() <<std::endl;
        //std::cout << _loss->dLoss_dW(i).cols() << " " << _loss->dLoss_dW(i).rows() <<std::endl;
        //std::cout << new_W.cols() << " " << new_W.rows() << std::endl;
        //update weights and biases
        
        nn.set_W(new_W, i);
        nn.set_b(new_b, i);
    }//exit(-1);
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


void plot_comparative(std::vector<double> y1, std::vector<double> y2){
	plt::plot(y1);
    plt::plot(y2);
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





#define TRAIN_SAMPLES 200
#define TEST_SAMPLES 1000
    Eigen::MatrixXd data_train, data_test;

    data_train = generate_points(TRAIN_SAMPLES);

    Eigen::MatrixXd y_train = data_train.col(2);
    Eigen::MatrixXd X_train(TRAIN_SAMPLES, 2);

    for(int i = 0; i < 2; i++){
        X_train.col(i) = data_train.col(i);
    }

    data_test = generate_points(TEST_SAMPLES);

    Eigen::MatrixXd y_test = data_test.col(2);
    Eigen::MatrixXd X_test(TEST_SAMPLES, 2);
    
    for(int i = 0; i < 2; i++){
        X_test.col(i) = data_test.col(i);
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
    
    Eigen::MatrixXd y_pred, y_pred_test;
    Loss loss_func;
    double lr = 0.001;
    Optimizer optimizer(loss_func ,lr);
    double loss = 0;

    std::vector<double> vloss_train, vloss_test;

    vloss_train.reserve(N_EPOCHS+1);
    vloss_test.reserve(N_EPOCHS+1);
    for(int epoch = 0; epoch < N_EPOCHS; epoch++){
        for(int i = 0; i < TRAIN_SAMPLES; i++){
            y_pred = nn.forward(X_train.row(i));
            loss += loss_func.compute(y_train.row(i), y_pred);
            loss_func.backward(nn);
            optimizer.step(nn);
        }

        
        std::cout << "loss for train samples in epoch " << epoch + 1 << " is: " << loss/TRAIN_SAMPLES << std::endl;
        vloss_train.emplace_back((double)loss/TRAIN_SAMPLES);
        loss = 0;

        for(int i = 0; i < TEST_SAMPLES; i++){
            y_pred_test = nn.forward(X_test.row(i));
            loss += loss_func.compute(y_test.row(i), y_pred_test);
        }
        std::cout << "loss for test samples in epoch " << epoch + 1 << " is: " << loss/TEST_SAMPLES << std::endl;
        vloss_test.emplace_back((double)loss/TEST_SAMPLES);
        loss = 0;
    }
    
    
    //plot_data(vloss_train);
    //plot_data(vloss_test);
    plot_comparative(vloss_train, vloss_test);
    
    return 0;
}