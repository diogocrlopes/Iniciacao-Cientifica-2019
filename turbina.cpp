#include <iostream>
#include <armadillo>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;
using namespace arma;

void criaMatrizMassRB(mat& massRB, vec& upsilon, double m, double Ixx, double Q, double J, double Iyy, double Izz,
    double nIzz, double Jt, double tempoControle, double P)
{
    massRB(0, 0) = 1;
    massRB(1, 1) = 1;
    massRB(2, 2) = 1;
    massRB(3, 3) = 1;
    massRB(4, 4) = 1;
    massRB(5, 5) = 1;
    massRB(6, 6) = m;    //(0,0)
    massRB(7, 7) = m;    //(1,1)
    massRB(8, 8) = m;    //(2,2)
    massRB(9, 9) = Ixx + Q + J;     //(3,3)
    massRB(10, 10) = Iyy + Q + J / 2;   //(4,4)
    massRB(11, 11) = Izz + nIzz - J / 2;    //(5,5)  
    massRB(14, 14) = 1;
    massRB(12, 12) = 1;   //(6,6)
    massRB(15, 15) = Jt;  //(7,7)
    massRB(13, 13) = tempoControle;  //(8,8)

    massRB(6, 10) = P;    //(0,4)
    massRB(7, 9) = -P;   //(1,3)
    massRB(9, 7) = -P;   //(3,1)
    massRB(10, 6) = P;    //(4,0)

    massRB(11, 15) = -J * sin(upsilon(4));  //(5,6)
}

void criaMatrizkRB(mat& kRB, vec& upsilon, vec& upsilonPonto, double m, double r, double Q, double P, double q, double p,
    double Izz, double nIzz, double Ixx, double J, double Iyy)
{
    kRB(6, 6) = 5.0e+05;   //amortecimento em X
    kRB(7, 7) = 5.0e+05;   //amortecimento em Y
    kRB(8, 8) = 5.0e+05;   //amortecimento em Z
    kRB(9, 9) = 3.7e+08;     //amortecimento em Momento em X
    kRB(10, 10) = 3.7e+08;   //amortecimento em Momento em Y
    kRB(11, 11) = 3.7e+08;   //amortecimento em Momento em Z


    kRB(6, 7) = m * r;
    kRB(7, 6) = -m * r;

    kRB(6, 8) = -m * q;
    kRB(8, 6) = m * q;

    kRB(7, 8) = m * p;
    kRB(8, 7) = -m * p;

    kRB(6, 9) = -P * r;
    kRB(7, 10) = -P * r;
    kRB(9, 6) = P * r;
    kRB(10, 7) = P * r;


    kRB(9, 8) = -P * p;
    kRB(8, 9) = P * p;

    kRB(10, 8) = -P * q;
    kRB(8, 10) = P * q;

    kRB(10, 9) = (r * (Izz + nIzz - Ixx - Q - J / 2)) / 2;
    kRB(11, 9) = (q * (Ixx - Iyy + J / 2)) / 2;

    kRB(9, 10) = (-r * (Izz + nIzz - Iyy - Q)) / 2;
    kRB(9, 11) = (-q * (Izz + nIzz - Iyy - Q)) / 2;

    kRB(10, 11) = (p * (Izz + nIzz - Ixx - Q - J / 2)) / 2;
    kRB(11, 10) = (-p * (Iyy - Ixx - J / 2)) / 2;

    kRB(9, 15) = -upsilonPonto(4) * J * sin(upsilon(4));
    kRB(10, 15) = J * (upsilonPonto(5) + upsilonPonto(3) * sin(upsilon(4)));

    kRB(14, 15) = -1;
    kRB(13, 13) = 1;
    kRB(12, 15) = -1;
    kRB(0, 6) = -(cos(upsilon(5)) * cos(upsilon(4)));
    kRB(1, 6) = -(sin(upsilon(5)) * cos(upsilon(4)));
    kRB(2, 6) = -(-sin(upsilon(4)));
    kRB(0, 7) = -((-sin(upsilon(5)) * cos(upsilon(3))) + (cos(upsilon(5)) * sin(upsilon(4)) * sin(upsilon(3))));
    kRB(1, 7) = -((cos(upsilon(5)) * cos(upsilon(3))) + (sin(upsilon(3)) * sin(upsilon(4)) * sin(upsilon(5))));
    kRB(2, 7) = -((cos(upsilon(4)) * sin(upsilon(3))));
    kRB(0, 8) = -((sin(upsilon(5)) * sin(upsilon(3)) + cos(upsilon(5)) * cos(upsilon(3)) * sin(upsilon(4))));
    kRB(1, 8) = -((-cos(upsilon(5)) * sin(upsilon(3)) + sin(upsilon(4)) * sin(upsilon(5)) * cos(upsilon(3))));
    kRB(2, 8) = -((cos(upsilon(3)) * cos(upsilon(4))));
    kRB(3, 9) = -1;
    kRB(3, 10) = -(sin(upsilon(3)) * tan(upsilon(4)));
    kRB(3, 11) = -(cos(upsilon(3)) * tan(upsilon(4)));
    kRB(4, 10) = -(cos(upsilon(3)));
    kRB(4, 11) = -(-sin(upsilon(3)));
    kRB(5, 10) = -(sin(upsilon(3)) / cos(upsilon(4)));
    kRB(5, 11) = -(cos(upsilon(3)) / cos(upsilon(4)));
}

/********************************
Funções para poder realizar a interpolação
********************************/
void indexMatrix(mat& Matrix, float lamb, vec& index)
{
    int x1, x2;
    int a = 0, b;
    while (lamb >= Matrix(0, a))
    {
        x1 = a;
        a++;
        x2 = a;
    }
    b = index(0);
    index(b + 1) = x1;
    index(b + 2) = x2;
    index(0) += 2;
}


void calculoDiferenca(mat& Matrix, vec& index, float controle, float variavel)
{
    double diferenca, var1{ 0 }, b;
    diferenca = abs(Matrix(0, index(controle + 1)) - Matrix(0, index(controle)));

    if (Matrix(0, index(controle)) != variavel)
    {
        var1 = variavel - Matrix(0, index(controle));
    }
    b = index(0);
    index(b + 1) = var1;
    index(b + 2) = diferenca;
    index(0) += 2;
}

void interpCq(mat& Cq, vec& index)
{
    int x1, x2, x3, x4;
    double yCq, yCq2, interCq;
    x1 = index(1);
    x2 = index(2);
    x3 = index(3);
    x4 = index(4);
    yCq = Cq(x1, x3) + ((Cq(x2, x3) - Cq(x1, x3)) * ((index(5) / index(6))));
    yCq2 = Cq(x1, x4) + ((Cq(x2, x4) - Cq(x1, x4)) * ((index(5) / index(6))));

    interCq = yCq + ((yCq2 - yCq) * ((index(7) / index(8))));
    index(9) = interCq;
}
void interpCt(mat& Ct, vec& index)
{
    int x1, x2, x3, x4;
    double yCt, yCt2, interCt;
    x1 = index(1);
    x2 = index(2);
    x3 = index(3);
    x4 = index(4);
    yCt = Ct(x1, x3) + ((Ct(x2, x3) - Ct(x1, x3)) * ((index(5) / index(6))));
    yCt2 = Ct(x1, x4) + ((Ct(x2, x4) - Ct(x1, x4)) * ((index(5) / index(6))));

    interCt = yCt + ((yCt2 - yCt) * ((index(7) / index(8))));
    index(10) = interCt;
}


void interpCp(mat& Cp, vec& index)
{
    int x1, x2, x3, x4;
    double yCp, yCp2, interCp;
    x1 = index(1);
    x2 = index(2);
    x3 = index(3);
    x4 = index(4);
    yCp = Cp(x1, x3) + ((Cp(x2, x3) - Cp(x1, x3)) * ((index(5) / index(6))));
    yCp2 = Cp(x1, x4) + ((Cp(x2, x4) - Cp(x1, x4)) * ((index(5) / index(6))));

    interCp = yCp + ((yCp2 - yCp) * ((index(7) / index(8))));
    index(11) = interCp;
}

void calculoPotEmpTor(vec& armazenaVariavel, vec& upsilon, double raio, double vento, double torqueGer, vec& index) // função para calcular Potência, Empuxo e Torque
{
    double potencia, empuxo, torque;
    double rho{ 1.225 }, velVento; // raio{63}
    velVento = vento;

    torque = 0.5 * rho * 3.14 * pow(raio, 3) * pow(vento, 2) * index(9);

    armazenaVariavel(4) = (torque);

    empuxo = 0.5 * rho * 3.14 * pow(raio, 2) * pow(vento, 2) * index(10);

    armazenaVariavel(5) = empuxo;

    potencia = 0.5 * rho * 3.14 * pow(raio, 2) * pow(vento, 3) * index(11);

    armazenaVariavel(7) = potencia;

}

void eqsim(vec& upsilon, vec& upsilonPonto, vec& tau, mat& massRB, mat& kRB, vec& armazenaVariavel, double vento,
    mat& Beta, mat& Lambda, mat& Cq, mat& Ct, mat& Cp, vec& index, double omegaRef, double Kp, double Tj) //calcula as equações diferenciais
{

    float conTintegrativo, conTproporcional;
    double m{ 13820460 }, Ixx{ 6827000000 }, Iyy{ 6827000000 }, Izz{ 12260000000 };
    double Q{ 3608630060 }, J{ 23552094 }, Jt{ 5141423.44 }, nIzz{ 2607890 };
    double tempoControle{ 2 }, P{ 35539000 }, torqueGer{ 3709155.452 }, raio{ 63 };
    double lamb, bet, interCq, interCt, interCp, erro;
    double potencia, empuxo, torque;

    if (upsilon(13) > 0.37) //condicional para manter o beta dentro do intervalo possível
    {
        upsilon(13) = 0.37;
    }
    else if (upsilon(13) < -0.17)
    {
        upsilon(13) = -0.17;
    }
    if (vento > 22)
    {
        upsilon(15) = 0;
    }

    massRB(11, 12) = -J * upsilon(4);

    empuxo = armazenaVariavel(5);
    /********************************
     Declaração das forças e momentos nos três eixos
    ********************************/
    tau(6) = -7.08e+04 * upsilon(0) + 1.08e+05 * upsilon(4) + armazenaVariavel(5); // força em x
    tau(7) = -7.08e+04 * upsilon(1) - 1.08e+05 * upsilon(3); // força em y
    tau(8) = -3.836e+06 * upsilon(2) - 1839000 - 1.91e+04 * upsilon(2); // força em z
    tau(9) = -3.776e+08 * upsilon(3) - 8.73e+07 * upsilon(3) ; // momento em X
    tau(10) = -3.776e+08 * upsilon(4) - 8.73e+07 * upsilon(4) + empuxo * 100.6; //momento em y
    tau(11) = -1.17e+08 * upsilon(5); // momento em z

    /**********************Início da atuação do controle****************************/

    tau(12) = -omegaRef; // omega de referência

    /*if (conTintegrativo < -0.17)
    {
        upsilon(12) = -0.17 * Tj / Kp; // o valor mínimo que pode ocorrer é -0.072779230
    }
    else if (conTintegrativo > 0.37)
    {
        upsilon(12) = 0.37 * Tj / Kp; // o valor mínimo que pode ocorrer é 0.158401853
    }*/

    erro = (upsilon(15) - omegaRef);

    upsilonPonto(12) = erro; //upsilon(15) - omegaRef;

    conTintegrativo = (Kp * (1 / Tj) * upsilon(12));

    conTproporcional = (Kp * erro); //armazenaVariavel(6)

    tau(13) = conTproporcional + conTintegrativo; // theta controle

    if (conTintegrativo < -0.17)
    {
        upsilon(12) = -0.17 * Tj / Kp; //-0.072779230
    }
    else if (conTintegrativo > 0.37)
    {
        upsilon(12) = 0.37 * Tj / Kp;//0.158401853
    }

    if (tau(13) < -0.17)
    {
        tau(13) = -0.17;
    }
    else if (tau(13) > 0.37)
    {
        tau(13) = 0.37;
    }
    /******************************************************/

    armazenaVariavel(6) = upsilonPonto(12); //guardando o omegaPonto dentro do armazenaVariavel(6) para uso posterior


    kRB(6, 6) = 5.0e+05;   //amortecimento em X
    kRB(7, 7) = 5.0e+05;   //amortecimento em Y
    kRB(8, 8) = 5.0e+05;   //amortecimento em Z
    kRB(9, 9) = 3.7e+8;     //amortecimento em Momento em X
    kRB(10, 10) = 3.7e+8;   //amortecimento em Momento em Y
    kRB(11, 11) = 3.7e+8;   //amortecimento em Momento em Z

    kRB(6, 7) = m * upsilonPonto(5);
    kRB(7, 6) = -m * upsilonPonto(5);

    kRB(6, 8) = -m * upsilonPonto(4);
    kRB(8, 6) = m * upsilonPonto(4);

    kRB(7, 8) = m * upsilonPonto(3);
    kRB(8, 7) = -m * upsilonPonto(3);

    kRB(6, 9) = -P * upsilonPonto(5);
    kRB(7, 10) = -P * upsilonPonto(5);
    kRB(9, 6) = P * upsilonPonto(5);
    kRB(10, 7) = P * upsilonPonto(5);


    kRB(9, 8) = -P * upsilonPonto(3);
    kRB(8, 9) = P * upsilonPonto(3);

    kRB(10, 8) = -P * upsilonPonto(4);
    kRB(8, 10) = P * upsilonPonto(4);

    kRB(10, 9) = (upsilonPonto(5) * (Izz + nIzz - Ixx - Q - J / 2)) / 2;
    kRB(11, 9) = (upsilonPonto(5) * (Ixx - Iyy + J / 2)) / 2;

    kRB(9, 10) = (-upsilonPonto(5) * (Izz + nIzz - Iyy - Q)) / 2;
    kRB(9, 11) = (-upsilonPonto(4) * (Izz + nIzz - Iyy - Q)) / 2;

    kRB(10, 11) = (upsilonPonto(3) * (Izz + nIzz - Ixx - Q - J / 2)) / 2;
    kRB(11, 10) = (-upsilonPonto(3) * (Iyy - Ixx - J / 2)) / 2;

    kRB(9, 15) = -upsilonPonto(4) * J * sin(upsilon(4));
    kRB(10, 15) = J * (upsilonPonto(5) + upsilonPonto(3) * sin(upsilon(4)));

    kRB(0, 6) = -(cos(upsilon(5)) * cos(upsilon(4)));
    kRB(1, 6) = -(sin(upsilon(5)) * cos(upsilon(4)));
    kRB(2, 6) = -(-sin(upsilon(4)));
    kRB(14, 15) = -1;
    kRB(13, 13) = 1;
    kRB(12, 15) = -1;
    kRB(0, 7) = -((-sin(upsilon(5)) * cos(upsilon(3))) + (cos(upsilon(5)) * sin(upsilon(4)) * sin(upsilon(3))));
    kRB(1, 7) = -((cos(upsilon(5)) * cos(upsilon(3))) + (sin(upsilon(3)) * sin(upsilon(4)) * sin(upsilon(5))));
    kRB(2, 7) = -((cos(upsilon(4)) * sin(upsilon(3))));
    kRB(0, 8) = -((sin(upsilon(5)) * sin(upsilon(3)) + cos(upsilon(5)) * cos(upsilon(3)) * sin(upsilon(4))));
    kRB(1, 8) = -((-cos(upsilon(5)) * sin(upsilon(3)) + sin(upsilon(4)) * sin(upsilon(5)) * cos(upsilon(3))));
    kRB(2, 8) = -((cos(upsilon(3)) * cos(upsilon(4))));
    kRB(3, 10) = -(sin(upsilon(3)) * tan(upsilon(4)));
    kRB(3, 11) = -(cos(upsilon(3)) * tan(upsilon(4)));
    kRB(4, 10) = -(cos(upsilon(3)));
    kRB(4, 11) = -(-sin(upsilon(3)));
    kRB(5, 10) = -(sin(upsilon(3)) / cos(upsilon(4)));
    kRB(5, 11) = -(cos(upsilon(3)) / cos(upsilon(4)));

    //Cálculo torque turbina
    lamb = (upsilon(15) * raio) / vento;

    bet = upsilon(13) * 57.2957795;

    /********************************
      Interpolação dos coeficientes de potência, empuxo e torque da turbina
     ********************************/
    int x1, x2;
    int a{ 0 }, b;
    while (lamb >= Lambda(0, a))
    {
        x1 = a;
        a++;
        x2 = a;
    }

    int y1, y2;
    int c{ 0 }, d;

    while (bet >= Beta(0, c))
    {
        y1 = c;
        c++;
        y2 = c;
    }

    double diferenca1, var1{ 0 };

    diferenca1 = abs(Lambda(0, x2) - Lambda(0, x1));

    if (Lambda(0, x1) != lamb)
    {
        var1 = lamb - Lambda(0, x1);
    }

    double diferenca2, var2{ 0 };

    diferenca2 = abs(Beta(0, y2) - Beta(0, y1));

    if (Beta(0, y1) != bet)
    {
        var2 = bet - Beta(0, y1);
    }

    double yCq, yCq2;
    yCq = Cq(x1, y1) + ((Cq(x2, y1) - Cq(x1, y1)) * ((var1 / diferenca1)));
    yCq2 = Cq(x1, y2) + ((Cq(x2, y2) - Cq(x1, y2)) * ((var1 / diferenca1)));

    interCq = yCq + ((yCq2 - yCq) * ((var2 / diferenca2)));
    index(9) = interCq;

    double yCt, yCt2;
    yCt = Ct(x1, y1) + ((Ct(x2, y1) - Ct(x1, y1)) * ((var1 / diferenca1)));
    yCt2 = Ct(x1, y2) + ((Ct(x2, y2) - Ct(x1, y2)) * ((var1 / diferenca1)));

    interCt = yCt + ((yCt2 - yCt) * ((var2 / diferenca2)));
    index(10) = interCt;

    double yCp, yCp2;
    yCp = Cp(x1, y1) + ((Cp(x2, y1) - Cp(x1, y1)) * ((var1 / diferenca1)));
    yCp2 = Cp(x1, y2) + ((Cp(x2, y2) - Cp(x1, y2)) * ((var1 / diferenca1)));

    interCp = yCp + ((yCp2 - yCp) * ((var2 / diferenca2)));
    index(11) = interCp;

    double rho{ 1.225 };

    /********************************
    Cálculo do torque,empuxo e potência da turbina
   ********************************/

    torque = 0.5 * rho * 3.14 * pow(raio, 3) * pow(vento, 2) * interCq;
    armazenaVariavel(4) = (torque);

    empuxo = 0.5 * rho * 3.14 * pow(raio, 2) * pow(vento, 2) * interCt;

    //tau(9) = -3.776e+08 * upsilon(3) - 8.73e+07*upsilon(3) + empuxo*100.6; Alterei lá para cima, verificar se não deu erro

    armazenaVariavel(5) = empuxo;

    potencia = 0.5 * rho * 3.14 * pow(raio, 2) * pow(vento, 3) * interCp;
    armazenaVariavel(7) = potencia;

    tau(15) = torque - (torqueGer)-163230 * upsilon(15);

    upsilonPonto = inv(massRB) * ((tau)-(kRB * upsilon));

    /********************************
    Local onde se pode zerar as acelerações em cada eixo
    ********************************/
    /*upsilonPonto(0)= 0;
    upsilonPonto(1)= 0;
    upsilonPonto(2)= 0;
    upsilonPonto(3) = 0;
    upsilonPonto(4) = 0;
    upsilonPonto(5) = 0;
    upsilonPonto(6) = 0;
    upsilonPonto(7) = 0;
    upsilonPonto(8) = 0;
    upsilonPonto(9) = 0;
    upsilonPonto(10) = 0;
    upsilonPonto(11) = 0;
    upsilonPonto(12) = 0;
    upsilonPonto(13) = 0;
    upsilonPonto(14) = 0;*/

}

void RKt4ordem(vec& upsilon, vec& upsilonPonto, vec& tau, mat& massRB, mat& kRB, double dt, vec& t, double nsis, vec& armazenaVariavel, double vento,
    mat& Beta, mat& Lambda, mat& Cq, mat& Ct, mat& Cp, vec& index, float psiReferencia, float Kp, float Tj) // Função Runge Kutta de 4 ordem
{
    /********************************
        Declaração de Variáveis
    ********************************/

    vec sy = zeros<vec>(nsis); //preenche os vetores com zero
    vec y0 = zeros<vec>(nsis); //preenche os vetores com zero
    vec y1 = zeros<vec>(nsis); //preenche os vetores com zero
    vec y2 = zeros<vec>(nsis); //preenche os vetores com zero

    float prt1, prt2, h, tempo, newdt;
    newdt = 1;
    tempo = t(0);

    eqsim(upsilon, upsilonPonto, tau, massRB, kRB, armazenaVariavel, vento, Beta, Lambda, Cq, Ct, Cp, index, psiReferencia, Kp, Tj);

    h = dt / 2;

    for (int i = 0; i < nsis; i++)
    {
        sy(i) = upsilon(i);
        y0(i) = upsilonPonto(i);
        upsilon(i) = h * upsilonPonto(i) + upsilon(i);
    }

    t(0) = t(0) + h;
    tempo = t(0);

    newdt = 0;
    eqsim(upsilon, upsilonPonto, tau, massRB, kRB, armazenaVariavel, vento, Beta, Lambda, Cq, Ct, Cp, index, psiReferencia, Kp, Tj);

    for (int i = 0; i < nsis; i++)
    {
        y1(i) = upsilonPonto(i);
        upsilon(i) = sy(i) + h * upsilonPonto(i);
    }

    eqsim(upsilon, upsilonPonto, tau, massRB, kRB, armazenaVariavel, vento, Beta, Lambda, Cq, Ct, Cp, index, psiReferencia, Kp, Tj);

    for (int i = 0; i < nsis; i++)
    {
        y2(i) = upsilonPonto(i);
        upsilonPonto(i) = sy(i) + dt * upsilonPonto(i);
    }

    t(0) = t(0) + h;
    tempo = t(0);
    eqsim(upsilon, upsilonPonto, tau, massRB, kRB, armazenaVariavel, vento, Beta, Lambda, Cq, Ct, Cp, index, psiReferencia, Kp, Tj);

    h = h / 3;

    for (int i = 0; i < nsis; i++)
    {
        prt1 = 2 * (y1(i) + y2(i));
        prt2 = y0(i) + upsilonPonto(i);
        upsilon(i) = sy(i) + h * prt1 + h * prt2;
    }

}


int main()
{
    /********************************
          Declaração de Variáveis
    ********************************/

    double dt{ .01 }, tempoFinal{ 1000 }, nf; // dt = intervalo de integração | nf = tempoFinal / dt

    double m{ 13820460 }, Ixx{ 6827000000 }, Iyy{ 6827000000 }, Izz{ 12260000000 }; // variáveis das tabelas de mRB e kRB
    double Q{ 3608630060 }, P{ 35539000 }, J{ 23552094 }, Jt{ 5141423.44 }, nIzz{ 2607890 };   // variáveis das tabelas de mRB e kRB

    double r{ 0 }, q{ 0 }, u{ 0 }, v{ 0 }, w{ 0 }, p{ 0 };
    double uPonto{ 0 }, vPonto{ 0 }, wPonto{ 0 }, pPonto{ 0 }, qPonto{ 0 }, rPonto{ 0 };

    double pos1{ 8.211 }, pos2{ 0 }, pos3{ -0.4777 }, pos4{ 0 }, pos5{ 0.1229 }, pos6{ 0 };      //posição com relação a coorenada fixa
    double pos1Ponto{ 0 }, pos2Ponto{ 0 }, pos3Ponto{ 0 }, pos4Ponto{ 0 }, pos5Ponto{ 0 }, pos6Ponto{ 0 };   // velocidade com relação a coordenada fixa

    double psi{ 0 }, psiPonto{ 1.348 }, psiDoisPontos{ 0 }, theta{ 0.081838489 }, thetaPonto{ 0 }, thetaPontos{ 0 };

    double ForcaX{ 0 }, ForcaY{ 0 }, ForcaZ{ 0 }, MomentoX{ 0 }, MomentoY{ 0 }, MomentoZ{ 0 }, torqueGer{ 3709155.452 }, raio{ 63 }, xAux{ 0 }, xAuxPonto, torqueTurb;

    double temp, lamb, bet;
    double Kp{ 1.88268 }, Tj{ 8.06 }, tempoControle{ 10 }, omegaRef{ 1.348 }; ; // variáveis do controle

    double vento{ 12 };

    float nsis;
    double armazena[10]; // vetor para armazenar variáveis temporárias
      /********************************
          Declaração de Matrizes
    ********************************/
    mat Lambda; // declarando a matriz
    Lambda.load("Lambda.txt"); // carregando os dados dos arquivos txt
    mat Beta; // declarando a matriz
    Beta.load("Beta.txt"); // carregando os dados dos arquivos txt
    mat Cp; // declarando a matriz
    Cp.load("C_P.txt"); // carregando os dados dos arquivos txt
    mat Cq; // declarando a matriz
    Cq.load("C_Q.txt"); // carregando os dados dos arquivos txt
    mat Ct; // declarando a matriz
    Ct.load("C_T.txt");  // carregando os dados dos arquivos txt
    vec index(15); // vetor de 15 linhas para guardar variáveis temporárias
    index(0) = 0;

    mat massRB(16, 16); // declaração do vetor massRB
    massRB.fill(0.0); // preenchimento do vetor massRB com todos os campos com zero

    vec tau; // declaração do vetor tau, que vai conter as forças e as informações do controle
    vec armazenaVariavel(8); // declaração de vetor para armazenar variáveis
    armazenaVariavel.fill(0.0); // preenchimento de vetor com 0

    mat kRB = zeros<mat>(16, 16); // declaração da matriz kRB e preenchimento dela com 0

    vec upsilon(16); // declaração do vetor upsilon com 16 campos iniciando seu indice no (0)
    upsilon.fill(0.0); // preenchimento com 0 do vetor upsilon
    xAux = theta * Tj / Kp; // cálculo

    upsilon = { pos1, pos2, pos3, pos4, pos5, pos6, u , v , w , p , q , r , xAux , theta ,psi , psiPonto }; // vetor com a posição em coordenada fixa e a velocidade em coordenada movel

    vec upsilonPonto(16);
    upsilonPonto.fill(0.0);
    upsilon(15) = omegaRef;

    xAuxPonto = upsilon(15) - omegaRef; //verificar se no estado inicial fica assim - originalmente é 1.348 -
    upsilonPonto = { pos1Ponto, pos2Ponto, pos3Ponto, pos4Ponto, pos5Ponto, pos6Ponto, uPonto,vPonto, wPonto , pPonto , qPonto , rPonto, xAuxPonto, thetaPonto, psiPonto , psiDoisPontos }; // vetor com a velocidade em coordenada fixa e a aceleração em coordenada movel

    upsilonPonto(14) = omegaRef; // colocando o dentro do vetor upsilonPonto o valor do omegaRef informado no campo das variáveis.
    criaMatrizMassRB(massRB, upsilon, m, Ixx, Q, J, Iyy, Izz, nIzz, Jt, tempoControle, P); // função para inicializar a matriz mRB
    criaMatrizkRB(kRB, upsilon, upsilonPonto, m, r, q, P, Q, p, Izz, nIzz, Ixx, J, Iyy); // função para inicializar a matriz kRB

    tau = { 0,0,0,0,0,0, -ForcaX , -ForcaY , -ForcaZ , -MomentoX , -MomentoY , -MomentoZ , -omegaRef , (Kp * (upsilonPonto(12) + ((1 / Tj) * upsilon(12)))) , 0 , 0 };

    //as forças em x e o momento em x serão inicializados mais abaixo pois precisam antes do cálculos dos coeficientes de empuxo, torque e potencia
    tau(7) = -7.08e+04 * upsilon(1) - 1.08e+05 * upsilon(3); // definição da força em Y
    tau(8) = -3.836e+06 * upsilon(2) - 1839000 - 1.91e+04 * upsilon(2); //definição da força em Z 1º Hidrostática, 2º Amarração, 3º Amarração também, 4º Peso.
    tau(10) = -3.776e+08 * upsilon(4) - 8.73e+07 * upsilon(4); // Momento em Y
    tau(11) = -1.17e+08 * upsilon(5); // Momeno em Z

    upsilonPonto(13) = (tau(13) - upsilon(13)) / tempoControle; // upsilonPonto(13) é o thetaControle

    /********************************
        Declaração de tabelas e Escrita dos termos no resposta.txt
    ********************************/

    ofstream resposta("resposta.txt"); // comando para criar o arquivo resposta como resposta.txt
    if (resposta.is_open()) { cout << "ok com arquivo resposta" << endl; } // verificação se o resposta.txt está aberto

    resposta << "tempo, pos1, pos2, pos3, pos4, pos5, pos6, u , v , w , p , q , r , xAux , theta ,psi , psiPonto , pos1Ponto, pos2Ponto, pos3Ponto, pos4Ponto, pos5Ponto, pos6Ponto, uPonto,vPonto, wPonto , pPonto , qPonto , rPonto, xAuxPonto, thetaPonto, psiPonto , psiDoisPontos,";
    resposta << "0,0,0,0,0,0, -forçaEmX , -forçaEmY , -forçaEmZ , -MomentoEmX , -MomentoEmY , -MomentoEmZ , -omegaRef , thetaControle , 0 , 0,";
    resposta << "torque, empuxo, potencia, Coef. Cq, Coef. Ct, Coef. Cp, Controle Proporcional, Controle Integrativo ,vento" << endl;

    /*****************************fim***********************************/


    lamb = (upsilon(15) * raio) / vento; // cálculo do lambda
    bet = upsilon(13) * 57.2957795; // cálculo do beta

     /********************************
        Funções para gerar os coeficientes Cq, Ct, Cp
    ********************************/
    indexMatrix(Lambda, lamb, index);
    indexMatrix(Beta, bet, index);
    calculoDiferenca(Lambda, index, 1, lamb);
    calculoDiferenca(Beta, index, 3, bet);
    interpCq(Cq, index);
    interpCt(Ct, index);
    interpCp(Cp, index);

    calculoPotEmpTor(armazenaVariavel, upsilon, raio, vento, torqueGer, index); // função para cálculo de Potência, Empuxo e Torque

    /******************************fim**********************************/

    double vaiDarCerto, jaDeuCerto;
    long double torque;

    torque = 0.5 * 1.225 * 3.14 * pow(raio, 3) * pow(vento, 2) * index(9); // definição do torque

    double empuxo;
    empuxo = armazenaVariavel(5);


    tau(6) = -7.08e+04 * upsilon(0) + 1.08e+05 * upsilon(4) + empuxo; // Força em X

    tau(9) = -3.776e+08 * upsilon(3) - 8.73e+07 * upsilon(3) + empuxo * 100.6; // Momento em X


    tau(15) = torque - (torqueGer)-163230 * upsilon(15); // Equação diferencial da turbina

    /********************************
        Início dos cálculos
  ********************************/

    vec t(1); // declaração do vetor t para guardar as informações do tempo
    t(0) = 0;
    nf = tempoFinal / dt;

    index(12) = Kp * upsilonPonto(12);  //cálculo do controle proporcional
    index(13) = Kp / Tj * upsilon(12); //cálculo do controle integrativo

    armazenaVariavel(6) = 0; //upsilonPonto(12);
    armazenaVariavel(7) = upsilon(13); // upsilon(13) é tetha ( ângulo da pá)
    upsilonPonto(15) = tau(15) / Jt; //cálculo da aceleração de rotação da pá
    upsilon(15) = upsilonPonto(14); //igualdade entre as duas velocidades angulares de rotação omega

    /********************************/ // Parte que escreve os valores iniciais no arquivo resposta.txt
    resposta << t(0) << ",";
    for (int i = 0; i < 16; i++)
    {
        resposta << upsilon(i) << ",";
    }
    for (int i = 0; i < 16; i++)
    {
        resposta << upsilonPonto(i) << ",";
    }
    for (int i = 0; i < 16; i++)
    {
        resposta << tau(i) << ",";
    }

    resposta << armazenaVariavel(4) << "," << armazenaVariavel(5) << "," << armazenaVariavel(7) << "," << index(9) << "," << index(10) << "," << index(11) << "," << index(12) << "," << index(13) << "," << vento << endl;
    /********************************/

    float conTproporcional, conTintegrativo; // declaração das variáveis para o controle proporcional e integrativo


    /********************************
        Início das iterações
    ********************************/


    for (int n = 1; n <= nf; n++)
    {
        /**********************************/ // Verificação de qual é a ordem do sistema
        if (vento <= 11)

            nsis = 12;
        else
            nsis = 16;
        /***********************************/
        index(0) = 0;

        /**********************************/  // Cálculo do vento em tres partes
        if (n < nf / 3)
        {
            vento = 12;

        }
        else if (n < 2 * nf / 3)
        {
            vento = 12 + ((9 * n / nf) - 3);
        }
        else if (n > 2 * nf / 3)
        {
            vento = 15;
        }

        /**********************************/

        RKt4ordem(upsilon, upsilonPonto, tau, massRB, kRB, dt, t, nsis, armazenaVariavel, vento, Beta, Lambda, Cq, Ct, Cp, index, omegaRef, Kp, Tj);

        cout << "psiponto é " << upsilon(15) << endl; // impressão do resultado do omega da turbina

        /********************************/ // Parte que escreve os valores iniciais no arquivo resposta.txt
        resposta << t(0) << ",";

        for (int i = 0; i < 16; i++)
        {
            resposta << upsilon(i) << ",";
        }
        for (int i = 0; i < 16; i++)
        {
            resposta << upsilonPonto(i) << ",";
        }
        for (int i = 0; i < 16; i++)
        {
            resposta << tau(i) << ",";
        }

        index(12) = Kp * upsilonPonto(12);
        index(13) = Kp / Tj * upsilon(12);

        resposta << armazenaVariavel(4) << "," << armazenaVariavel(5) << "," << armazenaVariavel(7) << "," << index(9) << "," << index(10) << "," << index(11) << "," << index(12) << "," << index(13) << "," << vento << endl;
    }

    resposta.close(); // fechamento do arquivo de texto

    return 0;
}
