// file : ukf.cpp
#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
//////////////////////////////////////////////////////////////////////////////////
UKF::UKF() {
  // Need Initialize :
  is_initialized_ = false;

  // Set state dimension
  n_x_ = 5;

  // Set augmented dimension
  n_aug_ = 7;

  // set measurement dimension, radate can measre r phi r_dot
  n_z_ = 3;

  // Define spreading parameter
  lambda_ = 0;
  /////////////////////////////////////////////////////////////

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
  //use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  //use_radar_ = false;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1; // 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5; // 30;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  // Need Initialize :
  // Matrix to hold sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  lambda_ = 3 - n_aug_;
  n_sig_ = 2 * n_aug_ + 1;
  // Vector for weights
  weights_ = VectorXd(2*n_aug_+1);
  // Noise matrices
  R_radar_ = MatrixXd(n_z_,n_z_);
  R_lidar_ = MatrixXd(2,2);

  // Start time
  time_us_ = 0;
  // NIS init
  NIS_radar_ = 0;
  NIS_lidar_ = 0;

  packet_radar_ = 0;
  packet_lidar_ = 0;


  x_aug_ = VectorXd (n_aug_);
  P_aug_ = MatrixXd (n_aug_, n_aug_);

  /////////////////////////////////////////////////////////////
  // Create R for update noise later
  R_lidar_ << std_laspx_*std_laspx_, 0,
               0                    , std_laspy_*std_laspy_;

  // Create R for update noise later
  R_radar_ << std_radr_*std_radr_, 0, 0,
                 0, std_radphi_*std_radphi_, 0,
                 0, 0, std_radrd_*std_radrd_;

} // constructor

//////////////////////////////////////////////////////////////////////////////////
UKF::~UKF() {} //destructor
//////////////////////////////////////////////////////////////////////////////////
void UKF::Init_Laser_Parameters (MeasurementPackage meas_package)
{
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    // Initialize state.
    // ***** Last three values below can be tuned *****
    //x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 4, 0.5, 0.0;
    x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0.0;
  }


}
//////////////////////////////////////////////////////////////////////////////////
void UKF::Init_Radar_Parameters (MeasurementPackage meas_package)
{
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    double rho    = meas_package.raw_measurements_(0);
    double phi    = meas_package.raw_measurements_(1);
    double rho_dot = meas_package.raw_measurements_(2);

    // polar to cartesian - r * cos(angle) for x and r * sin(angle) for y
    // ***** Middle value for 'v' can be tuned *****
    double  px = rho * cos(phi); 
    double  py = rho * sin(phi);
    double  vx = rho_dot * cos(phi);
    double  vy = rho_dot * sin(phi);
    //double  v  = sqrt(vx * vx + vy * vy);
    double  v  = rho_dot;
    x_ << px, py, v, 0, 0;

  }
}
//////////////////////////////////////////////////////////////////////////////////
void UKF::Init_weights ()
{
  //define spreading parameter
  double lambda = 3 - n_aug_;
  //set vector for weights
  double weight_0 = lambda/(lambda + n_aug_);
  weights_(0) = weight_0;

  for (int i=1; i < 2 * n_aug_ + 1; i++) {
    double weight = 0.5/(n_aug_+lambda);
    weights_(i) = weight;
  }
  //cout << weights_;
}
/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
//////////////////////////////////////////////////////////////////////////////////
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  // Calculate delta_t, store current time for future
  double delta_t = (measurement_pack.timestamp_ - time_us_) / 1000000.0;
  time_us_ = measurement_pack.timestamp_;

  if (!is_initialized_) {
    //cout << "Initialize \n";
    
    x_ << 0, 0, 0, 0, 0;
    P_ = MatrixXd::Identity(5,5);
    Init_weights ();
    Init_Laser_Parameters (measurement_pack);
    Init_Radar_Parameters (measurement_pack);
    //cout << "Initialize  >>\n";
    is_initialized_ = true;
    time_us_ = measurement_pack.timestamp_;
    return;
  }
 
  if ((measurement_pack.sensor_type_ == MeasurementPackage::LASER) && (use_laser_)) {
    //cout << "LIDAR ";
    packet_lidar_ ++;
    Prediction(delta_t);
    UpdateLidar(measurement_pack);
  } else if ((measurement_pack.sensor_type_ == MeasurementPackage::RADAR) && (use_radar_)) {
    //cout << "RADAR  ";
    packet_radar_ ++;
    Prediction(delta_t);
    UpdateRadar(measurement_pack);
  } else {
    cout << "NONO\n";
  } 
  //cout << "ProcessMeasurement <<\n";
}
#include "s2_ukf"
#include "s3_ukf"
#include "s4_ukf"
#include "s8_radar_hook"
/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
//////////////////////////////////////////////////////////////////////////////////
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  //MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //cout << " Prediction " << "(" << packet_lidar_ << " , " << packet_radar_ << ")" << " << \n";
  AugmentedSigmaPoints(&Xsig_aug);
  SigmaPointPrediction(&Xsig_pred_, Xsig_aug, delta_t);
  PredictMeanAndCovariance(&x_, &P_, Xsig_pred_);

}
//////////////////////////////////////////////////////////////////////////////////
double UKF::NormalizeAngle (double x)
{
  while (x > M_PI) x -= 2.0 * M_PI;
  while (x < -M_PI) x += 2.0 * M_PI;
  return x;
}

void UKF::tune_parameters (int gLidarRadarFlag, double g_std_a, double g_std_ywdd)
{
  if (gLidarRadarFlag == 1) { 
    use_laser_ = true; use_radar_ = false;
  } else if (gLidarRadarFlag ==2) {
    use_laser_ = false;use_radar_ = true ;
  } else if (gLidarRadarFlag == 3){
    use_laser_ = true ;use_radar_ = true ;
  } else {
    cout << "Error in setting Lidar Radar value \n";
  }
  std_a_     = g_std_a;
  std_yawdd_ = g_std_ywdd;
}

