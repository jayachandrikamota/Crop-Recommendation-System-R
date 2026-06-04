# ==========================================================
# MODEL TRAINING (EXTENDED VERSION)
# ==========================================================

library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(dplyr)

cat("========== STEP 2: MODEL TRAINING ==========\n")

# ----------------------------------------------------------
# 1. LOAD PROCESSED DATA
# ----------------------------------------------------------

cat("\nLoading processed dataset...\n")

obj <- readRDS("data/processed_data.rds")

df_scaled <- obj$data
preproc <- obj$preproc

cat("Data Loaded Successfully\n")

# ----------------------------------------------------------
# 2. DATA SPLITTING (TRAIN-TEST)
# ----------------------------------------------------------

cat("\n========== DATA SPLITTING ==========\n")

set.seed(42)

split <- createDataPartition(df_scaled$label, p=0.8, list=FALSE)

train_data <- df_scaled[split, ]
test_data  <- df_scaled[-split, ]

cat("Training Rows:", nrow(train_data), "\n")
cat("Testing Rows:", nrow(test_data), "\n")

cat("\nTraining Class Distribution:\n")
print(table(train_data$label))

cat("\nTesting Class Distribution:\n")
print(table(test_data$label))

# ----------------------------------------------------------
# 3. BASELINE MODEL: DECISION TREE (GINI)
# ----------------------------------------------------------

cat("\n========== DECISION TREE (GINI) ==========\n")

dt_model <- rpart(label ~ ., data=train_data, method="class")

cat("Decision Tree trained successfully\n")

# Tree visualization
rpart.plot(dt_model, main="Decision Tree (Gini)")

# Tree complexity
cat("\nTree Complexity:\n")
print(dt_model$cptable)

# ----------------------------------------------------------
# 4. DECISION TREE (INFORMATION GAIN)
# ----------------------------------------------------------

cat("\n========== DECISION TREE (ENTROPY) ==========\n")

dt_entropy <- rpart(
  label ~ ., 
  data=train_data,
  method="class",
  parms=list(split="information")
)

cat("Entropy-based Decision Tree trained\n")

rpart.plot(dt_entropy, main="Decision Tree (Entropy)")

# ----------------------------------------------------------
# 5. RANDOM FOREST MODEL
# ----------------------------------------------------------

cat("\n========== RANDOM FOREST ==========\n")

rf_model <- randomForest(
  label ~ ., 
  data=train_data,
  ntree=150,
  importance=TRUE
)

cat("Random Forest trained successfully\n")

# ----------------------------------------------------------
# 6. FEATURE IMPORTANCE ANALYSIS
# ----------------------------------------------------------

cat("\n========== FEATURE IMPORTANCE ==========\n")

importance_vals <- importance(rf_model)

print(importance_vals)

cat("\nTop Features:\n")
print(sort(importance_vals[,1], decreasing=TRUE))

# Plot importance
varImpPlot(rf_model, main="Feature Importance (Random Forest)")

# ----------------------------------------------------------
# 7. MODEL INTERPRETATION (VERY IMPORTANT)
# ----------------------------------------------------------

cat("\n========== MODEL INSIGHTS ==========\n")

cat("\n- Decision Tree provides interpretable rules")
cat("\n- Random Forest improves accuracy using ensemble learning")
cat("\n- Important features include rainfall, temperature, and pH")
cat("\n- Ensemble models reduce overfitting compared to single trees")

# ----------------------------------------------------------
# 8. QUICK TRAINING PERFORMANCE CHECK
# ----------------------------------------------------------

cat("\n========== TRAINING PERFORMANCE ==========\n")

# Predictions on training data
train_pred_rf <- predict(rf_model, train_data)

train_cm <- confusionMatrix(train_pred_rf, train_data$label)

cat("\nTraining Accuracy (RF):\n")
print(train_cm$overall["Accuracy"])

# ----------------------------------------------------------
# 9. MODEL STORAGE (VERY IMPORTANT)
# ----------------------------------------------------------

cat("\n========== SAVING MODELS ==========\n")

saveRDS(list(
  dt = dt_model,
  entropy = dt_entropy,
  rf = rf_model,
  test_data = test_data,
  preproc = preproc
), "models/models.rds")

cat("Models saved successfully\n")

# ----------------------------------------------------------
# 10. LOGGING COMPLETION
# ----------------------------------------------------------

cat("\n========== MODEL TRAINING COMPLETED ==========\n")