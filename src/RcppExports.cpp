// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// Concert
List Concert(const arma::mat X, const arma::vec y, const arma::vec n_vec, const arma::vec tau, const arma::vec q_k, double eta, double q_0, double threshold, int max_iter);
RcppExport SEXP _CONCERT_Concert(SEXP XSEXP, SEXP ySEXP, SEXP n_vecSEXP, SEXP tauSEXP, SEXP q_kSEXP, SEXP etaSEXP, SEXP q_0SEXP, SEXP thresholdSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type n_vec(n_vecSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type q_k(q_kSEXP);
    Rcpp::traits::input_parameter< double >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< double >::type q_0(q_0SEXP);
    Rcpp::traits::input_parameter< double >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(Concert(X, y, n_vec, tau, q_k, eta, q_0, threshold, max_iter));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_CONCERT_Concert", (DL_FUNC) &_CONCERT_Concert, 9},
    {NULL, NULL, 0}
};

RcppExport void R_init_CONCERT(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}