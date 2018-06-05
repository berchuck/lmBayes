#include <RcppArmadillo.h>
#include "lmBayes.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat lm(arma::vec Y, arma::mat X, Rcpp::List Hypers, Rcpp::List Inits, int NSims) {

  //Data Objects
  int n, p;
  p = X.n_cols;
  n = X.n_rows;

  //Matrix Object
  arma::mat EyeP = arma::eye(p, p);

  //Initial Values
  arma::vec Beta = Rcpp::as<arma::vec>(Inits[0]);
  double Sigma2 = Rcpp::as<double>(Inits[1]);

  //Hyperparameters
  double SigmaBeta2 = Rcpp::as<double>(Hypers[0]);
  arma::vec Sigma2Hypers = Rcpp::as<arma::vec>(Hypers[1]);
  double Alpha = arma::as_scalar(Sigma2Hypers[0]);
  double Theta = arma::as_scalar(Sigma2Hypers[1]);

  //MCMC Objects
  arma::mat Out(p + 1, NSims);

  //MCMC Sampler
  for (int s = 0; s < NSims; s++) {

    //Full Conditional For Beta
    Beta = SampleBeta(Y, X, Sigma2, EyeP, SigmaBeta2);

    //Full Conditional For Sigma2
    Sigma2 = SampleSigma2(Y, X, Beta, Alpha, Theta, n);

    //Save Output
    Out(arma::span(0, (p-1)), s) = Beta;
    Out(p, s) = Sigma2;

    //Output Progress
    Rcpp::Rcout << "Iteration " << s + 1 << std::endl;

  }

  //Return MCMC Draws
  return Out;

}
