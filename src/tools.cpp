#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
  cout << "Calculating RMSE...";

  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size()==0){
      cout << "ERROR - Vector size is zero";
      return rmse;
  }
  if (estimations.size()!=ground_truth.size()){
      cout << "ERROR - Vector sizes are not equal" << endl;
      return rmse;
  }

  // accumulate squared residuals
  for (int i=0; i < estimations.size(); ++i) {
    VectorXd res = estimations[i] - ground_truth[i];
    res = res.array() * res.array();
    rmse += res;
  }
  // calculate the mean
  rmse = rmse / estimations.size();
  // calculate the squared root
  rmse = rmse.array().sqrt();
  // return the result
  cout << "Complete" << endl;
  return rmse;
}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  // Calculate th Jacobian.

  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // check division by zero
  if (px == 0 && py == 0){
      cout << "CalculateJacobian() - Error - Division by Zero" << endl;
      return Hj;
  }
  // compute the Jacobian matrix
  float px2 = px * px;
  float py2 = py * py;
  float c1 = px2 + py2;
  float c2 = sqrt(c1);
  float c3 = c1*c2;
    
  Hj << px/c2, py/c2, 0, 0,
        -py/c1, px/c1, 0, 0,
        px*(vx*py-vy*px)/c3, py*(vy*px-vx*py)/c3, px/c2, py/c2; 

  return Hj;
}
