

# install.packages("arules")
# install.packages("arulesViz")


library(arules)
library(arulesViz)
library(datasets)

setwd("C://Users/shivam/Desktop/freelancer/p21")

data <- read.csv("marketbasketdataset.csv", sep=',',header = TRUE)

complete.cases(data)

for (i in 1:20)
{
  data[,i] = as.logical(data[,i])
}

head(data)

data$id <- NULL

head(data)

raw <- as(data, "transactions")



summary(raw)

itemFrequencyPlot(raw, type="absolute", topN=20)
#See the items with the highest frequency

itemFrequencyPlot(raw, type="absolute", topN=10)

#inspect the sparse matrix:
inspect(raw[1:3])

#frequency of the first six items in alphabetical order:
itemFrequency(raw[,1:6])

#items that are present in over 20% transactions:
itemFrequencyPlot(raw, support= 0.20)
itemFrequencyPlot(raw, topN=5, type="relative")

itemFrequencyPlot(raw, topN=15, type="relative")

#DATA MODELING:
rules_2 <- apriori(raw,parameter = list(supp=0.002, conf=0.90, minlen=2))

rules <- apriori(raw) #taking default values for support and confidence

summary(rules)

inspect(sort(rules, by = "lift")[1:5])

#develop a data model graph:
rules_1 <- apriori(raw,parameter = list(supp=0.1, conf=0.70, minlen=2))

rules_1 <- apriori(raw,parameter = list(supp=0.1, conf=0.70, minlen=3), appearance = list(default="lhs", rhs=c("item_35")), control = list(verbose=F))

plot(rules_1, method="graph", shading="confidence")   #plot a graph to visualize the association rules

rules_2 <- apriori(raw,parameter = list(supp=0.002, conf=0.90, minlen=2)) #observe rules for lower selling items in relation to higher selling ones
