library(Rcpp)
sourceCpp("./src/Concert-binary.cpp")
#' Create indices for multiple source datasets
#'
#' This function generates a list of indices to split data into multiple source datasets based on the provided vector of dataset sizes.
#'
#' @param n_vec A vector of integers indicating the number of data points in each dataset.
#' @param num_K An integer indicating the number of datasets (excluding the target dataset).
#'
#' @return A list of length `num_K`, where each element is a vector of indices corresponding to the rows of the input data for that dataset.
#'
#' @export
#'
#' @examples
#' n_vec <- c(100, 150, 200)
#' num_K <- 2
#' sample_ind(n_vec, num_K)
sample_ind <- function(n_vec, num_K){
  # The indices of different source data sets
  ind <- list()
  for (k in 1:num_K){
    if (k == 1){
      ind[[1]] <- 1:n_vec[1]
    } else{
      ind[[k]] <- (sum(n_vec[1:(k-1)])+1): sum(n_vec[1:k])
    }
  }
  return(ind)
}
#' Evaluate prediction error
#'
#' This function calculates the prediction error for binary classification based on logistic regression.
#'
#' @param est A vector representing the estimated coefficients.
#' @param X_test A matrix of test feature data.
#' @param y_test A vector of true labels for the test data.
#'
#' @return A numeric value representing the prediction error.
#'
#' @export
#'
#' @examples
#' est <- c(0.5, -0.2)
#' X_test <- matrix(c(1, 2, 3, 4), ncol=2)
#' y_test <- c(1, 0)
#' eval_pre(est, X_test, y_test)
eval_pre <- function(est, X_test=NULL, y_test=NULL){
  pred_err <- NA
  if(!is.null(X_test)& !is.null(y_test)){
    y_pred <- 1/(1+exp(-X_test%*%est)) >= 0.5
    pred_err <- sum(y_test!=y_pred)/length(y_test)
  }
  return(pred_err=pred_err)
}
#' Performance metrics for regression
#'
#' This function calculates True Positive (TP) and False Positive (FP) rates for the regression model.
#'
#' @param theta A vector of estimated coefficients.
#' @param theta0 A vector of true coefficients.
#'
#' @return A vector of two values: TP and FP rates.
#'
#' @export
#'
#' @examples
#' theta <- c(1, 0, 1, 0, 0)
#' theta0 <- c(1, 0, 0, 1, 0)
#' perf(theta, theta0)
perf <- function(theta, theta0){
  Itheta0 <- as.numeric(theta0 != 0)
  Itheta <- as.numeric(theta != 0)

  TP <- sum(Itheta[which(Itheta0 == 1)])/sum(Itheta0)
  FP <- sum(Itheta[which(Itheta0 == 0)])/max(1,sum(Itheta))

  return(c(TP, FP))
}
#' Lasso regression estimation using glmnet
#'
#' This function performs Lasso regression to estimate the coefficients using the glmnet package.
#'
#' @param X A matrix of feature data.
#' @param y A vector of target variable values.
#' @param lambda The regularization parameter for Lasso.
#'
#' @return A vector of estimated coefficients.
#'
#' @export
#'
#' @examples
#' X <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#'               11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
#'             nrow = 20, ncol = 2)
#' y <- c(rep(0, 10), rep(1, 10))
#' lambda <- 0.1
#' lasso_est_R(X, y, lambda)
lasso_est_R <- function(X, y, lambda){
  beta <- as.numeric(glmnet::glmnet(X, y, family="binomial", lambda=lambda)$beta)
  return(beta)
}

#' Variational Spike-and-Slab Transfer Regression (Concert)
#'
#' This function implements the Variational Spike-and-Slab Transfer Regression model.
#'
#' @param X A matrix of feature data.
#' @param y A vector of target variable values.
#' @param n_vec A vector indicating the size of each source dataset.
#' @param threshold The convergence threshold for the algorithm.
#' @param max_iter The maximum number of iterations.
#' @param tau A vector of hyperparameters for each dataset.
#' @param q_k The probability of a coefficient being non-zero for each dataset.
#' @param eta The learning rate for updating the coefficients.
#' @param q_0 The prior probability of a coefficient being zero.
#'
#' @return A list containing the posterior estimates for the coefficients and other model outputs.
#'
#' @export
#'
#' @examples
#' set.seed(42)
#' n1 <- 100; n2 <- 120; n3 <- 110
#' p <- 20
#' n_vec <- c(n1, n2, n3)
#' X1 <- matrix(rnorm(n1 * p), nrow = n1, ncol = p)
#' X2 <- matrix(rnorm(n2 * p), nrow = n2, ncol = p)
#' X3 <- matrix(rnorm(n3 * p), nrow = n3, ncol = p)
#' beta_true <- c(rep(2, 5), rep(0, 15))
#' y1 <- rbinom(n1, 1, plogis(X1 %*% beta_true + rnorm(n1, sd = 0.5)))
#' y2 <- rbinom(n2, 1, plogis(X2 %*% beta_true + rnorm(n2, sd = 0.5)))
#' y3 <- rbinom(n3, 1, plogis(X3 %*% beta_true + rnorm(n3, sd = 0.5)))
#' X <- rbind(X1, X2, X3)
#' y <- c(y1, y2, y3)
#' Concert_R(X, y, n_vec, threshold = 1e-6, max_iter = 100, tau = c(1, 1, 1),
#'          q_k = 0.2, eta = 1, q_0 = 0.05)
Concert_R <- function(X, y, n_vec, threshold=1e-6, max_iter=1000,
                          tau=rep(1,length(n_vec)-1), q_k=0.2, eta=1, q_0=0.05){
  # threshold=1e-5; max_iter=1000; tau=1; q_k=0.2; eta=1; q_0=0.05
  p <- ncol(X)
  K <- length(n_vec)-1 # target
  ind <- sample_ind(n_vec, K+1)

  trace_m_beta0 <- matrix(0, nrow=p, ncol=1); trace_m_beta0 <- trace_m_beta0[,-1]
  trace_gamma_Z <- matrix(0, nrow=p, ncol=1); trace_gamma_Z <- trace_gamma_Z[,-1]
  trace_m_betak <- matrix(0, nrow=K*p, ncol=1); trace_m_betak <- trace_m_betak[,-1]
  trace_gamma_I <- matrix(0, nrow=K*p, ncol=1); trace_gamma_I <- trace_gamma_I[,-1]

  #################### Initialization ####################
  m_beta0 <- as.numeric(glmnet::glmnet(X[ind[[1]],], y[ind[[1]]], alpha=1,
                                       family="binomial", lambda = 0.01)$beta)
  V_beta0 <- rep(1,p)
  gamma_Z <- as.numeric(m_beta0!=0)
  # gamma_Z <- rep(q_0, p)
  m_betak <- matrix(0,p,K); gamma_I <- matrix(q_k,p,K); V_betak <- matrix(1,p,K)
  for (k in 2:(K+1)){
    m_betak[,k-1] <- as.numeric(glmnet::glmnet(X[ind[[k]],], y[ind[[k]]], alpha=1,
                                               family="binomial", lambda = 0.01)$beta)
    gamma_I[,k-1] <- as.numeric(m_betak[,k-1]!=0)
  }
  E_W <- list()
  for (k in 1:(K+1)){
    E_W[[k]] <-  rep(0.5,n_vec[k])
  }

  #################### CAVI ####################
  diff_m <- 10
  iter_num <- 0
  while (diff_m > threshold){
  # while (iter_num < 10){
    iter_start <- Sys.time()
    # old_m <- cbind(m_beta0, m_betak)
    old_m <- m_beta0

    ### beta0: m_beta0, V_beta0
    ### Z: gamma_Z
    order_beta0 <- order(abs(m_beta0), decreasing=T) # prioritized update scheme
    for (j in order_beta0){
      sum_var <- t(X[ind[[1]],j])%*%diag(E_W[[1]])%*%X[ind[[1]],j]+1/eta^2
      sum_mean <- t(y[ind[[1]]]-1/2)%*%X[ind[[1]],j]-t(X[ind[[1]],j])%*%diag(E_W[[1]])%*%X[ind[[1]],-j]%*%(gamma_Z[-j]*m_beta0[-j])
      for (k in 2:(K+1)){
        sum_var <- sum_var+
          t(X[ind[[k]],j])%*%diag(E_W[[k]])%*%X[ind[[k]],j]*gamma_I[j,k-1]+1/tau[k-1]^2*(1-gamma_I[j,k-1])
        sum_mean <- sum_mean+
          gamma_I[j,k-1]*(t(y[ind[[k]]]-1/2)%*%X[ind[[k]],j]-
                            t(X[ind[[k]],j])%*%diag(E_W[[k]])%*%X[ind[[k]],-j]%*%(gamma_I[-j,k-1]*gamma_Z[-j]*m_beta0[-j]+
                                                                                    (1-gamma_I[-j,k-1])*m_betak[-j,k-1]))+
          (1-gamma_I[j,k-1])*1/tau[k-1]^2*m_betak[j,k-1]
      }
      V_beta0[j] <- 1/sum_var
      m_beta0[j] <- sum_mean/sum_var

      logit <- m_beta0[j]^2/(2*V_beta0[j])+log(q_0*sqrt(V_beta0[j])/((1-q_0)*eta))
      gamma_Z[j] <- 1/(1+exp(-logit))
    }
    trace_m_beta0 <- cbind(trace_m_beta0, m_beta0)
    trace_gamma_Z <- cbind(trace_gamma_Z, gamma_Z)
    # cat("Iter ", iter_num, ": ", m_beta0[1:16], "\n", sep="")

    ### betak: m_betak, V_betak
    ### I_betak: gamma_I_k
    for (k in 2:(K+1)){
      order_betak <- order(abs(m_betak[,k-1]), decreasing=T) # prioritized update scheme
      for (j in order_betak){
        V_betak[j,k-1] <- 1/(t(X[ind[[k]],j])%*%diag(E_W[[k]])%*%X[ind[[k]],j]+1/tau[k-1]^2)
        m_betak[j,k-1] <- (t(y[ind[[k]]]-1/2)%*%X[ind[[k]],j]-
                             t(X[ind[[k]],j])%*%diag(E_W[[k]])%*%X[ind[[k]],-j]%*%(gamma_I[-j,k-1]*gamma_Z[-j]*m_beta0[-j]+
                                                                                     (1-gamma_I[-j,k-1])*m_betak[-j,k-1])+
                                              1/tau[k-1]^2*gamma_Z[j]*m_beta0[j])*V_betak[j,k-1]

        logit <- -m_betak[j,k-1]^2/(2*V_betak[j,k-1])-log((1-q_k)*sqrt(V_betak[j,k-1])/(q_k*tau[k-1]))+
          gamma_Z[j]*(t(y[ind[[k]]]-1/2)%*%X[ind[[k]],j]*m_beta0[j]-
             1/2*(t(X[ind[[k]],j])%*%diag(E_W[[k]])%*%X[ind[[k]],j]-1/tau[k-1]^2)*(m_beta0[j]^2+V_beta0[j])-
             m_beta0[j]*t(X[ind[[k]],j])%*%diag(E_W[[k]])%*%X[ind[[k]],-j]%*%(gamma_I[-j,k-1]*gamma_Z[-j]*m_beta0[-j]+
                                                                 (1-gamma_I[-j,k-1])*m_betak[-j,k-1]))
        gamma_I[j,k-1] <- 1/(1+exp(-logit))
      }
    }
    trace_m_betak <- cbind(trace_m_betak, as.numeric(m_betak))
    trace_gamma_I <- cbind(trace_gamma_I, as.numeric(gamma_I))

    ### omega
    # c_omega_0
    m2_beta0 <- diag(gamma_Z*V_beta0)+
      (gamma_Z%*%t(gamma_Z)+diag(gamma_Z*(1-gamma_Z)))*(m_beta0%*%t(m_beta0))
    c_omega_0 <- sqrt(diag(X[ind[[1]],]%*%m2_beta0%*%t(X[ind[[1]],])))
    E_W[[1]] <- 1/(2*c_omega_0)*tanh(c_omega_0/2)

    # c_omega_k
    for (k in 2:(K+1)){
      m2_betak <- diag((1-gamma_I[,k-1])*V_betak[,k-1])+
        (gamma_I[,k-1]%*%t(gamma_I[,k-1])+diag(gamma_I[,k-1]*(1-gamma_I[,k-1])))*m2_beta0+
        ((1-gamma_I[,k-1])%*%t(1-gamma_I[,k-1])+diag((1-gamma_I[,k-1])*gamma_I[,k-1]))*(m_betak[,k-1]%*%t(m_betak[,k-1]))+
        (gamma_I[,k-1]%*%t(1-gamma_I[,k-1])-diag(gamma_I[,k-1]*(1-gamma_I[,k-1])))*((gamma_Z*m_beta0)%*%t(m_betak[,k-1]))+
        ((1-gamma_I[,k-1])%*%t(gamma_I[,k-1])-diag((1-gamma_I[,k-1])*gamma_I[,k-1]))*(m_betak[,k-1]%*%t(gamma_Z*m_beta0))
      c_omega_k <- sqrt(diag(X[ind[[k]],]%*%m2_betak%*%t(X[ind[[k]],])))
      E_W[[k]] <- 1/(2*c_omega_k)*tanh(c_omega_k/2)
    }

    # diff_m <- sum((cbind(m_beta0, m_betak)-old_m)^2)
    diff_m <- sum((m_beta0-old_m)^2)
    iter_end <- Sys.time()
    iter_time <- difftime(iter_end, iter_start)[[1]]

    iter_num <- iter_num+1
    # cat("Iter ", iter_num, ": ", diff_m, " with time ", iter_time, " secs.", "\n", sep="")
    if (iter_num > max_iter){
      break
    }
  }

  result <- list(m_beta0, gamma_Z, m_betak, gamma_I, V_beta0, V_betak,
                 trace_m_beta0, trace_gamma_Z)
  names(result) <- c("m_beta0", "gamma_Z", "m_betak", "gamma_I", "V_beta0", "V_betak",
                     "trace_m_beta0", "trace_gamma_Z")
  return(result)
}

#' Variational Spike-and-Slab Regression (Solo Model)
#'
#' This function implements a Variational Spike-and-Slab Regression model, which is used to estimate the regression coefficients for a target dataset using logistic regression and a variational inference approach.
#'
#' @param X A matrix of size `n x p` representing the feature data, where `n` is the number of observations and `p` is the number of features.
#' @param y A vector of length `n` representing the binary target variable for the regression model.
#' @param threshold A numeric value specifying the convergence threshold. The algorithm stops if the difference between successive estimates of the coefficients is smaller than this value. Default is `1e-6`.
#' @param max_iter An integer specifying the maximum number of iterations. Default is `1000`.
#' @param eta A numeric value representing the learning rate or scale factor for the variance of the coefficients. Default is `1`.
#' @param q_0 A numeric value representing the prior probability of a coefficient being zero (the "spike" in the spike-and-slab prior). Default is `0.05`.
#'
#' @return A list containing the following elements:
#'   - `m_beta0`: A vector of the estimated regression coefficients for the target model.
#'   - `gamma_Z`: A vector of the posterior probabilities that each coefficient is non-zero (indicator variable for each coefficient).
#'   - `V_beta0`: A vector of the variances of the estimated coefficients.
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(200), ncol=2)
#' y <- sample(0:1, 100, replace=TRUE)
#' result <- Solo_R(X, y, threshold=1e-6, max_iter=1000, eta=1, q_0=0.05)
#' result$m_beta0
#' result$gamma_Z
#' result$V_beta0
#'
### only target
Solo_R <- function(X, y, threshold=1e-6, max_iter=1000, eta=1, q_0=0.05){
  p <- ncol(X)
  n0 <- nrow(X)

  trace_m_beta0 <- matrix(0, nrow=p, ncol=1); trace_m_beta0 <- trace_m_beta0[,-1]
  trace_gamma_Z <- matrix(0, nrow=p, ncol=1); trace_gamma_Z <- trace_gamma_Z[,-1]

  #################### Initialization ####################
  m_beta0 <- as.numeric(glmnet::glmnet(X, y, lambda = 0.1)$beta)
  V_beta0 <- rep(0,p)
  gamma_Z <- as.numeric(m_beta0!=0)
  E_W <-  rep(0.5,n0)

  #################### CAVI ####################
  diff_m <- 10
  iter_num <- 0
  while (diff_m > threshold){
    iter_start <- Sys.time()
    old_m <- m_beta0

    ### beta0: m_beta0, V_beta0
    ### Z: gamma_Z
    order_beta0 <- order(abs(m_beta0), decreasing=T) # prioritized update scheme
    for (j in order_beta0){
      sum_var <- t(X[,j])%*%diag(E_W)%*%X[,j]+1/eta^2
      sum_mean <- t(y-1/2)%*%X[,j]-t(X[,j])%*%diag(E_W)%*%X[,-j]%*%(gamma_Z[-j]*m_beta0[-j])
      V_beta0[j] <- 1/sum_var
      m_beta0[j] <- sum_mean/sum_var

      logit <- m_beta0[j]^2/(2*V_beta0[j])+log(q_0*sqrt(V_beta0[j])/((1-q_0)*eta))
      gamma_Z[j] <- 1/(1+exp(-logit))
    }
    trace_m_beta0 <- cbind(trace_m_beta0, m_beta0)
    trace_gamma_Z <- cbind(trace_gamma_Z, gamma_Z)

    # c_omega_0
    m2_beta0 <- diag(gamma_Z*V_beta0)+
      (gamma_Z%*%t(gamma_Z)+diag(gamma_Z*(1-gamma_Z)))*(m_beta0%*%t(m_beta0))
    c_omega_0 <- sqrt(diag(X%*%m2_beta0%*%t(X)))
    E_W <- 1/(2*c_omega_0)*tanh(c_omega_0/2)

    diff_m <- sum((m_beta0-old_m)^2)
    iter_end <- Sys.time()
    iter_time <- difftime(iter_end, iter_start)[[1]]

    iter_num <- iter_num+1
    # cat("Iter ", iter_num, ": ", diff_m, " with time ", iter_time, " secs.", "\n", sep="")
    if (iter_num > max_iter){
      break
    }
  }

  result <- list(m_beta0, gamma_Z, V_beta0)
  names(result) <- c("m_beta0", "gamma_Z", "V_beta0")
  return(result)
}

