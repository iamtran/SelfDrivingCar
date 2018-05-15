#include <iostream>
#include "tools.h"
#include <math.h>       /* atan2 */
using namespace std;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
    VectorXd rmse(4);
    rmse << 0,0,0,0;
        // check the validity of the following inputs:
        //  * the estimation vector size should not be zero
        //  * the estimation vector size should equal ground truth vector size
    if(estimations.size() != ground_truth.size() || estimations.size() == 0){
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }
        //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];

                //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }
        //calculate the mean
    rmse = rmse/estimations.size();
        //calculate the squared root
    rmse = rmse.array().sqrt();
        //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
    MatrixXd Hj(3,4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //pre-compute a set of terms to avoid repeated calculation
    float c1 = px*px+py*py;
    float c2 = sqrt(c1);
    float c3 = (c1*c2);

    //check division by zero
    if(fabs(c1) < 0.0001){
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0, 
          -(py/c1), (px/c1), 0, 0, 
          py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

    return Hj;
}
MatrixXd Tools::Calculate_Q(float noise_ax , float noise_ay , float dt) {
    
    // Precompute some usefull values to speed up calculations of Q
    float dt_2 = dt * dt; //dt^2
    float dt_3 = dt_2 * dt; //dt^3
    float dt_4 = dt_3 * dt; //dt^4
    float dt_4_4 = dt_4 / 4; //dt^4/4
    float dt_3_2 = dt_3 / 2; //dt^3/2
    MatrixXd Q  = MatrixXd(4, 4);
    Q  << dt_4_4 * noise_ax, 0, dt_3_2 * noise_ax, 0,
	  0, dt_4_4 * noise_ay, 0, dt_3_2 * noise_ay,
	  dt_3_2 * noise_ax, 0, dt_2 * noise_ax, 0,
 	  0, dt_3_2 * noise_ay, 0, dt_2 * noise_ay;
    return Q;
}

MatrixXd Tools::Calculate_F (float dt) {
    MatrixXd F = MatrixXd(4, 4);
    F << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
        0, 0, 0, 1;
    return F;
}
MatrixXd Tools::Calculate_P (float p_value=1000) {
    MatrixXd P  = MatrixXd (4,4);
    P << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1000, 0,
         0, 0, 0, 1000;
    return P;
}

VectorXd Tools::Convert_Polar2Cartesian (float rho, float rho_dot, float phi)
{
   float x = rho * cos(phi); 
   float y = rho * sin(phi);
   float vx = rho_dot * cos(phi);
   float vy = rho_dot * sin(phi);
   VectorXd vector_x (4) ;
   vector_x << x, y, vx , vy;
   return vector_x;
}
VectorXd Tools::Convert_Cartesian2Polar (const VectorXd& x_vector)
{
    double rho     = sqrt(x_vector(0)*x_vector(0) + x_vector(1)*x_vector(1));
    //double theta   = atan(x_vector(1) / x_vector(0));
    // FIX #2: use of atan2
    double theta   = atan2(x_vector(1) , x_vector(0));
    theta          = NormalizeAngle (theta);
    double rho_dot = (x_vector(0)*x_vector(2) + x_vector(1)*x_vector(3)) / rho;
    VectorXd h = VectorXd(3); // h(x_)
    h << rho, theta, rho_dot;
    return h;
}
float Tools::NormalizeAngle (float x)
{
    if ((x > M_PI) || (x < -M_PI)) {
        cout << "NormalizeAngle()" << endl;
    }
    while (x > M_PI) x -= 2.0 * M_PI;
    while (x < -M_PI) x += 2.0 * M_PI;
  return x;
}
