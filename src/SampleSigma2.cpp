#include <RcppArmadillo.h>
#include "lmBayes.h"

double SampleSigma2(arma::vec Y, arma::mat X, arma::vec Beta, double Alpha, double Theta, int n) {
  double AlphaNew = Alpha + n / 2;
  arma::vec Residuals = Y - X * Beta;
  arma::mat tResiduals = arma::trans(Residuals);
  arma::mat ThetaNewMat = Theta + tResiduals * Residuals / 2 ;
  double ThetaNew = arma::as_scalar(ThetaNewMat);
  return rInverseGamma(AlphaNew, ThetaNew);
}