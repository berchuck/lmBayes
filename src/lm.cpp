#include <RcppArmadillo.h>
#include "lmBayes.h"

//' Bayesian linear regression
//'
//' The function implements Bayesian linear regression using an Markov chain Monte Carlo sampler. It assumes a normal
//' distribution prior on the \code{Beta} coefficients and an inverse-Gamma prior for \code{Sigma2}. The matrix that is
//' returned from this function has \code{P + 1} columns and \code{NSims} rows. The first \code{P} columns are the cofficients
//' and the last column is variance parameter. The value \code{N} is the number of observations in the study, \code{P} is the
//' number of independent variables, and \code{NSims} is the number of iterations the sampler is run for.
//'
//' @param Y An \code{N x 1} response vector.
//' @param X A \code{N x P} matrix of independent variables.
//' @param Hypers A \code{list} object with two components named, \code{Beta} and \code{Sigma2}. The object \code{Beta}
//'        is a scalar indicating the hyperprior variance of the zero centered normal distributed prior for \code{Beta},
//'        and the object \code{Sigma2} is a two dimensional vector indicating the hyperparameters for the inverse-Gamma
//'        prior on \code{Sigma2}.
//' @param Inits A \code{list} object with two components named, \code{Beta} and \code{Sigma2}. The object \code{Beta}
//'        is a \code{P} dimensional vector of initial values for the vector \code{Beta}, and the object \code{Sigma2}
//'        is a scalar indicating the initial value of \code{Sigma2}.
//' @param NSims A scalar indicating the number of iterations the MCMC sampler is run.
//' @export
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
  arma::vec Beta = Inits[0];
  double Sigma2 = Inits[1];

  //Hyperparameters
  double SigmaBeta2 = Hypers[0];
  arma::vec Sigma2Hypers = Hypers[1];
  double Alpha = Sigma2Hypers[0];
  double Theta = Sigma2Hypers[1];

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
