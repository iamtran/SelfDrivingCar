//CarND-PID-Control-Project/PId.cpp
#include <iostream>
#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    Kp_ = Kp;
    Ki_ = Ki;
    Kd_ = Kd;
    p_error_ = 0;
    i_error_ = 0;
    d_error_ = 0;
    packet_count = 0;
}

void PID::ChangeKValues(double Kp, double Ki, double Kd) {

    Kp_ = Kp;
    Ki_ = Ki;
    Kd_ = Kd;
}
void PID::DisplayKvalues() {
    cout << "PID  : " << "Kp_= " << Kp_     << " Ki_= " << Ki_     << " Kd_= " <<  Kd_    << endl;
    //cout << "Input: " << "Kp_= " << Kp_init << " Ki_= " << Ki_init << " Kd_= " << Kd_init << endl;
}
void PID::UpdateError(double cte) {
    d_error_ = (cte - p_error_);
    p_error_ = cte;
    i_error_ += cte;
}

double PID::TotalError() {
    return (-Kp_ * p_error_ -Kd_ * d_error_ -Ki_ * i_error_);
}

double PID::Throttle(double goal_throttle)
{
    double throttle = goal_throttle ;
    // the following going the reverse ?
    // throttle = goal_throttle + (-Kp_ * p_error_ -Kd_ * d_error_ -Kp_ * i_error_);
    
    throttle = goal_throttle + (-Kp_ * p_error_ -Kd_ * d_error_ );
    if (throttle < 0.0 ) throttle = - throttle;
    return throttle;
}

double PID::steeringAngle ()
{
    double steeringAngle = (-Kp_ * p_error_) - (Kd_ * d_error_) -  (Ki_ * i_error_) ;
    return steeringAngle ;
}

