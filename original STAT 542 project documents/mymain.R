#####################################
# Load libraries
# Load your vocabulary and training data
#####################################
library(text2vec)
library(glmnet)

set.seed(6594)

myvocab <- scan(file = "myvocab.txt", what = character())
train <- read.table("train.tsv", stringsAsFactors = FALSE,
                    header = TRUE)

stop_words = c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
               "you", "your", "yours", "their", "they", "his", "her", 
               "she", "he", "a", "an", "and", "is", "was", "are", "were", 
               "him", "himself", "has", "have", "it", "its", "the", "us")

#####################################
# Define helper functions
#####################################

buildDTM = function(data, text, stop_words, ngrams, min_count, min_dprop, max_dprop) {
  itokens = itoken(text, 
                   preprocessor = tolower, 
                   tokenizer = word_tokenizer)
  vocab = create_vocabulary(itokens, 
                            stopwords = stop_words, 
                            ngram = ngrams)
  vocab = prune_vocabulary(vocab, 
                           term_count_min = min_count, 
                           doc_proportion_max = max_dprop, 
                           doc_proportion_min = min_dprop)
  dtm = create_dtm(itokens, vocab_vectorizer(vocab))
  tfidf = TfIdf$new()
  dtm = fit_transform(dtm, tfidf)
  return(dtm)
}

columnConform = function(data_matrix, column_names) {
  coverage = colnames(data_matrix) %in% column_names
  miss = !(column_names %in% colnames(data_matrix))
  
  new_data = data_matrix[, coverage]
  empty_data = matrix(0, nrow(new_data), sum(miss))
  colnames(empty_data) = column_names[miss]
  new_data = as.matrix(cbind(new_data, empty_data))
  new_data = new_data[, column_names]
  
  return(new_data)
}

#####################################
# Preprocess train data
#####################################

train$review = gsub('<.*?>', ' ', train$review)

dtm_train = buildDTM(data=train, text=train$review, stop_words=stop_words, ngrams=c(1L,4L), 
                     min_count=10, min_dprop=0.001, max_dprop=0.5)
dtm_train = columnConform(dtm_train, myvocab)

#####################################
# Train a binary classification model
#####################################

train_classifier = cv.glmnet(x = dtm_train, y = train$sentiment, 
                             family='binomial', alpha = 0,
                             nfolds = 5, thresh = 1e-5, maxit = 1e3)

#####################################
# Load test data, 
# Preprocess train data, and
# Compute prediction
#####################################
test <- read.table("test.tsv", stringsAsFactors = FALSE,
                   header = TRUE)

test$review = gsub('<.*?>', ' ', test$review)
dtm_test = buildDTM(data=test, text=test$review, stop_words=stop_words, ngrams=c(1L,4L), 
                    min_count=10, min_dprop=0.001, max_dprop=0.5)
dtm_test = columnConform(dtm_test, myvocab)

pred = predict(train_classifier, 
               s = train_classifier$lambda.min, 
               newx = as.matrix(dtm_test),
               type='response')
output = cbind(test$id, pred)
colnames(output) = c("id", "prob")

#####################################
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predicted probs
#####################################
write.table(output, file = "mysubmission.txt", 
            row.names = FALSE, sep='\t')