library(data.table)
library(plyr)
library(reshape2)
library(ggplot2)

#essentially the same script as 'collectTimingsOriginal.R' but re-purposed (a little) for advanced data

# string values to pull out only data from the file that we are interested in 
grepString <- "egrep \"Program Build:|Kernel|Data Copy Total|Total time|Bandwidth|GFLOPS\""

#directory to save pdf graphs to
saveDir <- "~/workspace/phd/doc/masters_diss/images/use"

# get latest data directory and pull out files
mainDataDir <- "../data"

# number of timesteps
samples <- 4410
# constants for bandwidth calculation
bandwidth_const <- (vol*8*1e-9*3)

# select only 256 or 512 of the following parameters:
latestDataDir <- "adv256"  # use original timings
#latestDataDir <- "adv512"  
vol <- 256 * 256 * 202
#vol <- 512 * 512 * 404
volname <- 256
#volname <- 512
ylimit <- 300
#ylimit <- 2050 

#concatenate name of directory to pull files from 
dataFiles <- system(paste("ls ",paste(mainDataDir,"/",latestDataDir,"/*.txt",sep="")),intern=TRUE)
wholeDataDir <- paste(mainDataDir,latestDataDir, sep="/")

# pull out name of files
dataFilesNames <- strsplit(dataFiles,"-")

# initialise a data frame, otherwise is tricky to pull out data from files into one automatically

firstFileName <- dataFiles[1]

#create a table from the first CSV file
initialTable <- read.csv(pipe(paste(grepString, firstFileName)),header=FALSE,col.names =c('type','value'),sep=":",stringsAsFactors = FALSE)

#turn it into a vector 
initialTypenames <- as.vector(initialTable[,1])

#add some columns to the vector
initialTypenames <- c("Version", "Platform", initialTypenames) 

# then create the empty data frame with columns based on what's in first data file
originalframe <- data.frame(matrix(ncol = length(initialTypenames)))
frame <- data.frame(matrix(ncol = length(initialTypenames)))
colnames(frame) <- initialTypenames

# loop over all the data files in the relevant directory  
for (i in 1:length(dataFiles))
{
    # pull out the filename
    currentFileName <- dataFiles[i]
    #strip out the version, platfrom, number of samples from the filename 
    Version <- tail(strsplit(dataFilesNames[[i]][1],"/")[[1]],1) # extract last part of first name of path
    Platform <- dataFilesNames[[i]][2]
    NF <- dataFilesNames[[i]][length(dataFilesNames[i][[1]])]
    samples <- NF
   
    # read the file into a table
    table <- read.csv(pipe(paste(grepString, currentFileName)),header=FALSE,col.names =c('type','value'),sep=":",stringsAsFactors = FALSE)

    # rejig the table as a data frame object and pull in the version and platform values 
    table$type <- as.character(table$type)
    dataRow <- t(as.matrix(table[,2]))
    typenames <- as.vector(table[,1])
    colnames(dataRow) <- typenames
    dataRow <- as.data.frame(dataRow)
    dataRow <- cbind(Version, Platform, dataRow)
    frame <- rbind(frame,dataRow)
}

# drop any dud data
frame <- na.omit(frame)

# rename some of the columns
frame <- rename(frame, c("Version"="code_version","Platform"="hw_platform"))

# caste the frame back to a table (what?)
dt <- data.table(frame)

#grep command for column name -- change column name to be more readable
dt$code_version[grepl("targetDP_C_",dt$code_version)] <- "targetDP_C"  

# where the code_version is equal to targetDP_c and the platform is not xeon phi, change name to CPU
dt$hw_platform[dt$code_version=="targetDP_C" & dt$hw_platform!="xeon_phi"] <- "Intel Xeon E5-2670" 

# replace all the different values with medians 
dtavg <- dt[,list("Program Build"=median(`Program Build`),"Kernel1"=median(`Kernel1`),"Kernel2"=median(`Kernel2`),"Kernels"=median(`Kernels`),"Data Copy Total"=median(`Data Copy Total`),"Total Time"=median(`Total time`), "Bandwidth"=median(`Bandwidth`), "GFLOPS"=median(`GFLOPS`)),by=c("code_version","hw_platform")]

# originally ran some local memory data in this directory -- chuck it
dtavg <- dtavg[!like(dtavg$code_version,"local")]


# building the stacked bar plot...

#subset drops colums you don't want
dtavgcut <- subset(dtavg, select=c("hw_platform", "code_version","Total Time","Data Copy Total","Kernel1", "Kernel2"))
# adds new column TimeLeft (unaccounted time remaining)
dtavgcut <- transform(dtavgcut, TimeLeft = (dtavgcut$`Total Time` - dtavgcut$`Data Copy Total`- dtavgcut$Kernel1 - dtavgcut$Kernel2))

#drops column Total Time (since it's a stacked plot, we only need it to add up to total time!)
dtavgcut[,`Total Time`:=NULL]

# give the hardware some proper names
dtavgcut$hw_platform[like(dtavgcut$hw_platform, "AMD_i7_4470K")] <- "AMD R9 259X2"
dtavgcut$hw_platform[like(dtavgcut$hw_platform, "AMD_R280")] <- "AMD R280"
dtavgcut$hw_platform[like(dtavgcut$hw_platform, "Intel")] <- "Intel Xeon E5"
dtavgcut$hw_platform[like(dtavgcut$hw_platform, "GTX")] <- "NVIDIA GTX780"
dtavgcut$hw_platform[like(dtavgcut$hw_platform, "xeon")] <- "Xeon Phi"
dtavgcut$hw_platform[like(dtavgcut$hw_platform, "nvidia_kepler")] <- "NVIDIA K20"
dtavgcut$code_version[like(dtavgcut$code_version, "targetDP_")] <- "targetDP"

dtavg$hw_platform[like(dtavg$hw_platform, "AMD_i7_4470K")] <- "AMD R9 259X2"
dtavg$hw_platform[like(dtavg$hw_platform, "AMD_R280")] <- "AMD R280"
dtavg$hw_platform[like(dtavg$hw_platform, "Intel")] <- "Intel Xeon E5"
dtavg$hw_platform[like(dtavg$hw_platform, "GTX")] <- "NVIDIA GTX780"
dtavg$hw_platform[like(dtavg$hw_platform, "xeon")] <- "Xeon Phi"
dtavg$hw_platform[like(dtavg$hw_platform, "nvidia_kepler")] <- "NVIDIA K20"
dtavg$code_version[like(dtavg$code_version, "targetDP_")] <- "targetDP"


# using ggplost - likes to have the data "melted down" first
melted <- melt(dtavgcut)

# create a nice name for the file to save
nameSave <- paste(saveDir,paste("advancedTimings",volname,".pdf",sep=""),sep="/")

# create the performance timings ggplot - this call is a bit of a beast. most of it is just tweaking to get the graph to look correct and beautiful 
allTimings <- ggplot(melted, aes(x = code_version, y = value, fill = variable))+theme_grey(base_size=16) +  geom_bar(stat = 'identity', position = 'stack') + facet_grid(~ hw_platform) + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) + coord_cartesian(ylim = c(0, ylimit)) + ylab("Time(s)") + xlab("Code Version [ Platform ]")+scale_fill_discrete("Code Section")+scale_fill_brewer( type = "div" , palette = "Spectral" ) +theme(legend.position=c(0.08,0.88))+labs(fill="")

#save the plot 
ggsave(nameSave, plot = allTimings, width = 297, height = 210, units = "mm" )

# create a similar plot, but for bandwidth values 
bandwidth <- subset(dtavg, select=c("hw_platform", "code_version","Bandwidth","Kernel1"))

# calculate the bandwidth from the data and some constants
bandwidth <- transform(bandwidth, Bandwidth_Calc = bandwidth_const / (bandwidth$`Kernel1`/samples))

#drop this data since we won't be using it anymore
bandwidth[,`Kernel1`:=NULL]
bandwidth[,`Bandwidth`:=NULL]
# tidy up the column name
names(bandwidth)[names(bandwidth)=="Bandwidth_Calc"] <- "Bandwidth" 

#now we're going to add some lines to the ggplot
N_hw <- unique(bandwidth$hw_platform)
peaks <- c(288,208,320,288.4,320) # constant values for peak bandwidth and stream values
stream <- c(250,220,240,150,158.4)
stream <- c(240,150,250,221,52)
#nvprof <- c(-50,-50,189.75,165.77,-50,-50)

#melt the data down...
meltedBW <- melt(bandwidth)
#map some colours to the lines
cols <- c("peak"="black","stream"="red","bandwidth"="blue4")

#match peak and stream values to the correct data
bandwidth <- transform(bandwidth, peakBW=peaks[match(bandwidth$hw_platform,N_hw)])
bandwidth <- transform(bandwidth, streamBW=stream[match(bandwidth$hw_platform,N_hw)])

#create a nice name
nameSave <- paste(saveDir,paste("advancedBandwidth",volname,".pdf",sep=""),sep="/")

#another beastly plot - this time for the bandwidth, including the lines
bwplot <- ggplot(meltedBW, aes(x = code_version, y = value, fill = "bandwidth"))+theme_grey(base_size=16) +  geom_bar(stat = 'identity', position = 'stack') + facet_grid(~ hw_platform) + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) + coord_cartesian(ylim = c(0, 350)) + ylab("Bandwidth (GB/s)") + xlab("Code Version [ Platform ]") + geom_hline(aes(yintercept=peakBW,colour="peak"),bandwidth,show_guide=TRUE)+scale_linetype(name = "Stream") + geom_hline(aes(yintercept=streamBW,colour="stream"),bandwidth,show_guide=TRUE) + scale_colour_manual(name = "Bandwidth Limits: ",values=c("stream"="red","peak"="black")) + scale_fill_manual(name="Calculated Bandwidth: ", values=c("bandwidth"="blue4"))+theme(legend.position="top")+labs(fill="")
ggsave(nameSave, plot = bwplot,width = 297, height = 210, units = "mm" )

#similar idea here for the GFLOPS graph: 
gflops <- subset(dtavg, select=c("hw_platform", "code_version","GFLOPS"))
meltedFlops<- melt(gflops)
fpeaks <- c(606,836,165.7,166,1175,320) # constants for peak GFLOPS
cols <- c("peak"="black","GFLOPS"="darkgreen")
gflops <- transform(gflops, peakF=fpeaks[match(gflops$hw_platform,N_hw)])
nameSave <- paste(saveDir,paste("advancedGflops",volname,".pdf",sep=""),sep="/")
allFlops <- ggplot(meltedFlops, aes(x = code_version, y = value, fill = variable))+theme_grey(base_size=16)  +  geom_bar(stat = 'identity', position = 'stack') + facet_grid(~ hw_platform) + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) + coord_cartesian(ylim = c(0, 1200)) + ylab("Gflops") + xlab("Code Version [ Platform ]")+ geom_hline(aes(yintercept=peakF,colour="peak"),gflops,show_guide=TRUE) + scale_colour_manual(name = "Peak GFLOPS: ",values=cols) + scale_fill_manual(name="Calculated GFLOPS: ", values=cols)+theme(legend.position="top")+labs(fill="")
ggsave(nameSave, plot = allFlops,width = 297, height = 210, units = "mm" )

# nice to have a csv copy of the table of median data for some armchair spreadsheet analysis
write.table(dtavg, paste(saveDir,"dtavg_advanced.csv",sep="/"), sep="\t") 
