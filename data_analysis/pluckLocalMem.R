library(data.table)
library(plyr)
library(reshape2)
library(ggplot2)
library(splitstackshape)

# create graphs comparing local memory versions for both room sizes 

# data already aggregated in this file
dataFile <- "../data/localMemoryData.txt" 

# files to save pdfs to
saveDir <- "~/workspace/phd/doc/masters_diss/images/use"

# read in data to table 
initTable<-read.csv(dataFile,header=FALSE,col.names=c('platform','language','size','type','x','y','z','time'),sep=":",stringsAsFactors = FALSE)

# drop the block sizes 
initTable <- initTable[ , !(names(initTable) %in% c("x","y","z"))]

# rename hardware 
initTable$platform[like(initTable$platform, "AMD")] <- "AMD R9 259X2"
initTable$platform[like(initTable$platform, "GTX")] <- "NVIDIA GTX780"
initTable$platform[like(initTable$platform, "Kepler")] <- "NVIDIA K20"

# there is probably a better way to do this, but hardcoded to separate out smaller and larger rooms
initTable256 <- subset(initTable,size==256)

# reformat as indexed table 
initTable256reshaped <- dcast.data.table(getanID(initTable256,c("platform","language")),platform+language ~ type,value.var="time")

# where data is missing, set to 0
initTable256reshaped[is.na(initTable256reshaped)] <- 0
#reorder the columns
setcolorder(initTable256reshaped,c("platform","language","sharedtex","shared","none"))

# melt the data down for the ggplot 
melted256 <- melt(initTable256reshaped)
#create name string to save file to
nameSave <- paste(saveDir,paste("localMemory256.pdf",sep=""),sep="/")
#create the ggplot for local memory  
melted256plot <- ggplot(melted256, aes(x=language, y=value, fill=variable)) +theme_grey(base_size=16)+ geom_bar(stat = 'identity', position='dodge')+facet_grid(~platform)+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))+scale_fill_brewer( type = "div" , palette = "YlGnBu" )+ xlab("Code Version [ Platform ]")+theme(legend.position=c(0.06,0.92))+labs(fill="")
ggsave(nameSave, plot = melted256plot,width = 297, height = 210, units = "mm" )


# repeat for larger room

initTable512 <- subset(initTable,size==512)
initTable512reshaped <- dcast.data.table(getanID(initTable512,c("platform","language")),platform+language ~ type,value.var="time")
initTable512reshaped[is.na(initTable512reshaped)] <- 0
setcolorder(initTable512reshaped,c("platform","language","sharedtex","shared","none"))
melted512 <- melt(initTable512reshaped)
nameSave <- paste(saveDir,paste("localMemory512.pdf",sep=""),sep="/")
melted512plot <- ggplot(melted512, aes(x=language, y=value, fill=variable)) +theme_grey(base_size=16)+ geom_bar(stat = 'identity', position='dodge')+facet_grid(~platform)+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))+scale_fill_brewer( type = "div" , palette = "YlGnBu" )+ xlab("Code Version [ Platform ]")+theme(legend.position=c(0.06,0.92))+labs(fill="")
ggsave(nameSave, plot = melted512plot,width = 297, height = 210, units = "mm" )

# write out tabled data for easier analysis
write.table(initTable, paste(saveDir,"localMem.csv",sep="/"), sep="\t") 
