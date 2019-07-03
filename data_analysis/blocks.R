library(data.table)
library(vioplot)
library(sm)

# script to create viol plots from workgroup / workitem data

# directory with data
blockDir <- "../data/blocks" 

# directory to save plots to
saveDir <- "~/workspace/phd/doc/masters_diss/images/use/blocks"

# pull out all output files from data dir
dataFiles <- list.files(path=blockDir, pattern="*R.out") 

# loop over all files and pull out data from each
for (i in 1:length(dataFiles))
{
    
    #parse name of file, split off the end bit and save as name array
    currentFileName <- dataFiles[i]
    name <- strsplit(currentFileName,"\\R.")
    name <- name[[1]][1]
    
    # create a table from data inside file, columns: "X" "Y" "Z" "time"
    clTable <- read.csv(paste(blockDir,currentFileName,sep="/"), header = FALSE, col.names = c('X', 'Y','Z', 'time'),stringsAsFactors = FALSE, sep=":")

    # caste table as data.frame object
    clFrame <- as.data.frame(clTable)
    
    #lose the Z column
    drops <- c("Z")
    clFrame <- clFrame[, !names(clFrame) %in% drops]

    # pull out the separate variations where X=[2-64]
    x1 <- clFrame$time[clFrame$X==2]
    x2 <- clFrame$time[clFrame$X==4]
    x3 <- clFrame$time[clFrame$X==8]
    x4 <- clFrame$time[clFrame$X==16]
    x5 <- clFrame$time[clFrame$X==32]
    x6 <- clFrame$time[clFrame$X==64]


    # pull out the separate variations where Y=[2-64]
    y1 <- clFrame$time[clFrame$Y==2]
    y2 <- clFrame$time[clFrame$Y==4]
    y3 <- clFrame$time[clFrame$Y==8]
    y4 <- clFrame$time[clFrame$Y==16]
    y5 <- clFrame$time[clFrame$Y==32]
    y6 <- clFrame$time[clFrame$Y==64]

    # rename "name" for title
    name <- gsub("_blocks_","",name)
    
    # set title for graph
    title <- paste(name,"BlockSizeXDimensionVariance",sep="")

    # tell R to save the next graph as this pdf
    pdf(paste(saveDir,paste(paste(name,"ViolPlotX",sep=""),".pdf",sep=""),sep="/"))

    #resize some graph parameters
    par(cex.lab=1.3, cex.axis=1.3) 
    
    # make the violin plot for X blocks
    vioplot(x1, x2, x3, x4, x5, x6, names=c("2", "4", "8", "16", "32", "64") )

    # set titles and labels
    title("",xlab="X Block Size",ylab="time(s)")

    #turn off plot save
    dev.off()
    
    title <- paste(name,"BlockSizeYDimensionVariance",sep="")
    pdf(paste(saveDir,paste(paste(name,"ViolPlotY",sep=""),".pdf",sep=""),sep="/"))
    par(cex.lab=1.3, cex.axis=1.3) 

    # make the violin plot for Y blocks
    vioplot(y1, y2, y3, y4, y5, y6, names=c("2", "4", "8", "16", "32", "64"), col="gold" )
    title("",xlab="Y Block Size",ylab="time(s)")
    dev.off()

}

