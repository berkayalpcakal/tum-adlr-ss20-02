library(data.table)
library(ggplot2)

setwd("/home/tomasruiz/code/thesis/exps-csvs/panda/")
csvs <- dir()[grepl(".csv", dir())]
getTitle <- function(csvName) if (grepl("ppo", csvName)) "PPO" else "HER+SAC"
loadTbl <- function(fname) cbind(Algorithm=getTitle(fname), GoalGAN=grepl("goalgan", fname), fread(fname))
allData <- rbindlist(lapply(csvs, loadTbl))
allData <- allData[, .(MapPctCovered=mean(MapPctCovered), stdDev=sd(MapPctCovered)), by=.(Algorithm, GoalGAN, Step)]

m <- melt(allData, id.vars = c("Algorithm", "GoalGAN", "Step"))
ggplot(allData, aes(Step, MapPctCovered, color=GoalGAN)) +
    geom_ribbon(aes(ymin=MapPctCovered-stdDev, ymax=MapPctCovered+stdDev, fill=GoalGAN), alpha=0.2, color=NA) +
    geom_point() +
    geom_line() +
    facet_grid(.~Algorithm, scales = "free")

