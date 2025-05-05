library(Horsekicks)
data(hkdeaths)

# Filter for G corps
g_data <- subset(hkdeaths, corps == "G")  # Assumes column is named 'corps'

# Extract the drowning death counts
x <- g_data$drown  # Replace 'drown' with correct column name if different

# Null hypothesis value
lambda0 <- 3.75

# Calculate test statistic
n <- length(x)
xbar <- mean(x)
D <- 2 * n * (lambda0 - xbar) + 2 * sum(x) * log(xbar / lambda0)

# Compute p-value from chi-squared distribution with 1 df
p_value <- pchisq(D, df = 1, lower.tail = FALSE)

# Output results
cat("Likelihood Ratio Test Statistic:", D, "\n")
cat("P-value:", p_value, "\n")