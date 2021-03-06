#'lm
#'
#' Bayesian linear regression
#'
#' The function implements Bayesian linear regression using a Markov chain Monte Carlo sampler. It assumes a normal
#' distribution prior on the \code{Beta} coefficients and an inverse-Gamma prior for \code{Sigma2}. The matrix that is
#' returned from this function has \code{P + 1} rows and \code{NSims} columns The first \code{P} rows are the cofficients
#' and the last row is the variance parameter. The value \code{N} is the number of observations in the study, \code{P} is the
#' number of independent variables, and \code{NSims} is the number of iterations the sampler is run for.
#'
#' @name lm
#' @author Samuel I. Berchuck \email{sib2@duke.edu}
#' @param Y An \code{N x 1} response vector.
#' @param X A \code{N x P} matrix of independent variables.
#' @param Hypers A \code{list} object with two components named, \code{Beta} and \code{Sigma2}. The object \code{Beta}
#'        is a scalar indicating the hyperprior variance of the zero centered normal distributed prior for \code{Beta},
#'        and the object \code{Sigma2} is a two dimensional vector indicating the hyperparameters for the inverse-Gamma
#'        prior on \code{Sigma2}.
#' @param Inits A \code{list} object with two components named, \code{Beta} and \code{Sigma2}. The object \code{Beta}
#'        is a \code{P} dimensional vector of initial values for the vector \code{Beta}, and the object \code{Sigma2}
#'        is a scalar indicating the initial value of \code{Sigma2}.
#' @param NSims A scalar indicating the number of iterations the MCMC sampler is run.
#' @export
NULL
