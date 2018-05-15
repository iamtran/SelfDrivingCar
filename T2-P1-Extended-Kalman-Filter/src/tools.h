#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * A helper method to calculate Jacobians.
  */
  MatrixXd CalculateJacobian(const VectorXd& x_state);

  MatrixXd Calculate_Q(float noise_ax , float noise_ay , float dt);
  MatrixXd Calculate_F (float dt);
  MatrixXd Calculate_P (float p_value);
  VectorXd Convert_Polar2Cartesian (float rho, float rho_dot, float phi);
  VectorXd Convert_Cartesian2Polar (const VectorXd& x_vector);
  float    NormalizeAngle (float phi);

};

#endif /* TOOLS_H_ */
