library(testthat)
test_that("Test for Concert_R", {
  X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  y <- sample(c(0, 1), 100, replace = TRUE)
  n_vec <- c(50, 50)
  result <- Concert_R(X, y, n_vec)
  expect_type(result$m_beta0, "double")
  expect_equal(length(result$m_beta0), ncol(X))
})
