//CarND-MPC-Project/MPC.h
#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;
#define DT 0.05
#define LF 2.67
#define NUMBER_OF_STEPS 12
#define MILE_HR_METER_SEC 0.44704
struct MPC_Output{

		vector<double> X;
		vector<double> Y;
		vector<double> Psi;
		vector<double> V;
		vector<double> Cte;
		vector<double> EPsi;
		vector<double> Delta;
		vector<double> A;
};


class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  MPC_Output     Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
};

#endif /* MPC_H */
