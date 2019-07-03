library(data.table)
library(plyr)
library(reshape2)
library(ggplot2)

# script to graph the different memory layouts 

#directory to save pdfs to 
saveDir <- "~/workspace/phd/doc/masters_diss/images/use"

# file already aggregated
file  <- "../data/structCompare.txt"

# read in the data into a table object 
structTable <- read.csv(file,header=FALSE,col.names =c('platform','type','kernel time','total time'),sep=":",stringsAsFactors = FALSE)

#re-cast as a frame
frame = as.data.frame(structTable)
# add in a time leftover value
frame <- transform(frame, timeLeft = (frame$`total.time` - frame$`kernel.time`))

# give the hardware some proper names
frame$platform[like(frame$platform, "nvidia_kepler")] <- "NVIDIA K20"
frame$platform[like(frame$platform, "xeonphi")] <- "Xeon Phi"
frame$platform[like(frame$platform, "amd")] <- "AMD R9 259X2"

#drop everything but the data of interest
keeps <- c("platform", "type","kernel.time","timeLeft")
frameKernel <- frame[keeps]

#rename the columns for time 
names(frameKernel)[names(frameKernel) == 'timeLeft'] <- 'total'
names(frameKernel)[names(frameKernel) == 'kernel.time'] <- 'kernel'

#the usual data meltdown
meltedKernel <- melt(frameKernel)
#nane to save pdf to
nameSave <- paste(saveDir,paste("structComparisonKernel.pdf",sep=""),sep="/")
struct <- ggplot(meltedKernel, aes(x = type, y = value, fill = variable)) +theme_grey(base_size=16)+  geom_bar(stat = 'identity', position = 'stack') + facet_grid(~ platform) + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) + coord_cartesian(ylim = c(0, 200)) + ylab("Time(s)") + xlab("Code Version [ Platform ]")+scale_fill_brewer( type = "div" , palette = "Paired")+theme(legend.position=c(0.05,0.92))+labs(fill="")
#save the ggplot
ggsave(nameSave, plot = struct,width = 297, height = 210, units = "mm" )

write.table(frame, paste(saveDir,"datastruct.csv",sep="/"), sep="\t") 
