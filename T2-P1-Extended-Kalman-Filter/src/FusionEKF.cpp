#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define MICRO_SECOND 1000000.0
#define EPSILON      0.0001

extern Tools tools;
/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_      = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  //ekf_.P_ =  tools.Calculate_P(1000);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  //cout <<"FusionEKF::FusionEKF()" <<endl;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}


void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    //cout << ekf_.P_ << endl;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      cout << "Radar: Init\n";
      float rho     = measurement_pack.raw_measurements_[0]; // range
      float phi     = measurement_pack.raw_measurements_[1]; // bearing
      float rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho
      //phi           = tools.NormalizeAngle (phi);
      // angle normalization
      while ( phi > M_PI) phi -=2.*M_PI;
      while ( phi <-M_PI) phi +=2.*M_PI;
      ekf_.x_       = tools.Convert_Polar2Cartesian (rho, rho_dot, phi); 
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      cout << "Lidar: Init\n";
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    if (fabs(ekf_.x_(0)) < EPSILON ) {
        ekf_.x_(0) =EPSILON;
        cout << "EPSILON 1\n";
    }
    if (fabs(ekf_.x_(1)) < EPSILON){
        ekf_.x_(1) = EPSILON;
    }
    // done initializing, no need to predict or update
    //cout << "EPSILON 2\n";
    ekf_.P_ = tools.Calculate_P(1000);
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    //cout << "EPSILON 3\n";
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  float noise_ax = 9.0;
  float noise_ay = 9.0;

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / MICRO_SECOND;
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.F_ = tools.Calculate_F (dt);
  ekf_.Q_ = tools.Calculate_Q(noise_ax , noise_ay , dt);
  //cout << "Predict(before) \n";
  ekf_.Predict();
  //cout << "Predict(end) \n";
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    //cout << "ekf_.UpdateEKF \n";
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    //cout << "ekf_.Update\n";
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  //cout << "x_ = \n" << ekf_.x_ << endl;
  //cout << "P_ = \n" << ekf_.P_ << endl;
}
