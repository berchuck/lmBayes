#ifndef __lmBayes__
#define __lmBayes__

double rInverseGamma(double Alpha, double Theta);
arma::mat rMVN(int n, arma::vec mu, arma::mat sigma);
arma::mat SampleBeta(arma::vec Y, arma::mat X, double Sigma2, arma::mat EyeP, double SigmaBeta2);
double SampleSigma2(arma::vec Y, arma::mat X, arma::vec Beta, double Alpha, double Theta, int n);

#endif // __lmBayes__
