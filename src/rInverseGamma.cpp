#include <RcppArmadillo.h>
#include "lmBayes.h"

double rInverseGamma(double Alpha, double Theta) {
  return 1 / R::rgamma(Alpha, 1 / Theta);
}
