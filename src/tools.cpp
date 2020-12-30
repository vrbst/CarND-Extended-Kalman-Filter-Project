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
   * DONE: Calculate the RMSE here. 
   * 
   * Source: Lesson 23: Extended Kalman Filters - Evaluating KF Performance 2
   */
   VectorXd rmse(4);
   rmse << 0,0,0,0;

   if (estimations.size() != ground_truth.size() || estimations.size() == 0)
      return rmse;

   for (unsigned int i = 0; i < estimations.size(); ++i)
   {
      VectorXd residual = estimations[i] - ground_truth[i];
      //cout << "i=" << i << " Residual:" << residual << endl;
      residual = residual.array() * residual.array();
      rmse += residual;
   }
      
   rmse = rmse / estimations.size(); // calculate the mean
   rmse = rmse.array().sqrt(); // calculate the squared root

   //cout << "RMSE:" << rmse << endl;
   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * DONE:
   * Calculate a Jacobian here.
   */
   MatrixXd Hj(3,4);

   float Px = x_state(0);
   float Py = x_state(1);
   float Vx = x_state(2);
   float Vy = x_state(3);

   if (Px == 0 && Py == 0)
      return Hj; // Error, cannot divide by 0 so return with the empty Matrix

   float val0 = pow(Px, 2) + pow(Py, 2);
   float val1 = sqrt(val0);
   float val2 = pow(val0, 3/2);

   Hj << Px / val1 , Py / val1, 0, 0,
        -Py / val0, Px / val0, 0, 0,
        (Py * (Vx * Py - Vy * Px)) / val2, (Px * (Vy * Px - Vx * Py)) / val2, Px / val1, Py / val1;

   return Hj;
}
