setwd("C://Users/shivam/Desktop/freelancer/p21/")

data <- read.csv("cereals.csv")

head(data)
summary(data)

cereals <- scale(data[,-9])

d <- dist(as.matrix(cereals))
hc <- hclust(d)
plot(hc)


# K-Means Cluster Analysis
fit <- kmeans(cereals, 5) # 5 cluster solution
# get cluster means 
aggregate(cereals,by=list(fit$cluster),FUN=mean)
# append cluster assignment
cereals <- data.frame(cereals, fit$cluster)


