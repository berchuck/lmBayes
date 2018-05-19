#include <RcppArmadillo.h>
#include "lmBayes.h"

arma::mat SampleBeta(arma::vec Y, arma::mat X, double Sigma2, arma::mat EyeP, double SigmaBeta2) {
  arma::mat Xt = arma::trans(X);
  arma::mat CovBeta = arma::inv( (Xt * X) / Sigma2 + EyeP / SigmaBeta2 );
  arma::vec MeanBeta = CovBeta * ( (Xt * Y) / Sigma2 );
  arma::mat out = rMVN(1, MeanBeta, CovBeta);
  return arma::trans(out);
}
 