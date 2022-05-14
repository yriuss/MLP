#include <iostream>
#include <stdio.h>
#include <Eigen>

class Layer{
public:
    Layer(uint d_in, uint d_out);

private:
    Eigen::MatrixXd weights;

};

Layer::Layer(uint d_in, uint d_out){

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

int main(){
    C1 teste;
    teste.print();
    C2 t(teste);
    teste.print();
    t.add();
    teste.print();
    return 0;
}