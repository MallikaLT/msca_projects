library(anytime)
library(ggplot2)
library(dplyr)
library(lubridate)
library(vioplot)
library(RColorBrewer)
library(formattable)

cdata<-read.csv("cancer_pt_data.csv")
cdata$ID <- seq.int(nrow(cdata))
str(cdata)   

#format dates and get proper NAs
cdata$DEATH_DATE_off[cdata$DEATH_DATE_off == "NULL"]<-NA
cdata[cdata == "#N/A"]<-NA
summary(cdata$BIRTH_DATE_off, 5)
cdata[,c(1,2,10,11,12,13,14)]<-lapply(cdata[,c(1,2,10,11,12,13,14)],
                                      function(z) as.Date(as.character(z), format = "%m/%d/%Y"))
  
#patients with no DX
dx_dates<-cdata[,c(10,11,12,13,14)]
no_dx_ind<-apply(dx_dates, 1, function(z) all(is.na(z)))
length(no_dx_ind[no_dx_ind==TRUE])

#remove those rows - everyone left has a DX date 
cdata<-cdata[no_dx_ind==FALSE,]
dim(cdata) #4153

#remove patients diagnosed after 8/25/16
#first put dummy date in for NA since it won't filter properly otherwise - remove 5 rows
cdata[, c(1,2,10:14)][is.na(cdata[,c(1,2,10:14)])] <- as.Date('1900-01-01')
cdata<-cdata[which(cdata$LUNG.DT < '2016-08-26' & cdata$PANCREAS_DT < '2016-08-26' & cdata$THYROID_DT < '2016-08-26' & cdata$COLORECTAL_DT < '2016-08-26' & cdata$MELANOMA_DT < '2016-08-26'),]
#put NA back
#cdata[,10:14][cdata[,10:14]==as.Date('1900-01-01')] <- NA

#people who have multiple cancers - 29 total
#remove for clarity
cdata$multi<-rowSums(cdata[,5:9])
length(cdata$multi[cdata$multi>1]) 
cdata2<-cdata[cdata$multi==1,]

#add patient age at DX and death age
cdata2$DXage<-ifelse(cdata2$LUNG==1,(cdata2$LUNG.DT-cdata2$BIRTH_DATE_off)/365.25,
                   ifelse(cdata2$PANCREAS==1,(cdata2$PANCREAS_DT-cdata2$BIRTH_DATE_off)/365.25,
                          ifelse(cdata2$THYROID==1,(cdata2$THYROID_DT-cdata2$BIRTH_DATE_off)/365.25,
                                 ifelse(cdata2$COLORECTAL==1,(cdata2$COLORECTAL_DT-cdata2$BIRTH_DATE_off)/365.25,
                                        (cdata2$MELANOMA_DT-cdata2$BIRTH_DATE_off)/365.25))))

cdata2$DeathAge<-ifelse(cdata2$DEATH_DATE_off > '1900-01-01',(cdata2$DEATH_DATE_off-cdata2$BIRTH_DATE_off)/365.25, NA)

#split patients into age deciles, by cancer DX
cdata2$ageCat<-cut(cdata2$DXage, c(10,20,30,40,50,60,70,80,90,100))

#look at death date dist 
#3822 patients don't have a death date & 326 do 
#301 had death date before 8/25/16 
length(cdata$DEATH_DATE_off[cdata$DEATH_DATE_off=='1900-01-01'])
length(cdata$DEATH_DATE_off[cdata$DEATH_DATE_off > '2016-08-25'])
hist(cdata2$DeathAge)

#add cancer type
cdata2$cancer_type<-ifelse(cdata2$LUNG==1, "Lung Cancer", 
                           ifelse(cdata2$PANCREAS==1, "Pancreatic Cancer",
                                  ifelse(cdata2$THYROID==1, "Thyroid Cancer",
                                         ifelse(cdata2$COLORECTAL==1, "Colorectal Cancer", "Melanoma"))))

#get time from DX to death for those who died 
cdata2$dx2death<-ifelse(!is.na(cdata2$DeathAge), (cdata2$DeathAge-cdata2$DXage), NA)
#verify 
cdata2[which(cdata2$dx2death > 20), 15:20] 
hist(cdata2$dx2death)
cdata2[which(cdata2$dx2death < 0),] 
#one person had a negative dx2death due to data error - remove her
cdata2<-cdata2[which(cdata2$dx2death >= 0 | is.na(cdata2$dx2death)),]
hist(cdata2$dx2death, main = "Years to Death After Diagnosis", xlab = "Years",
     ylab = "Probability of Death", prob = TRUE, col = "grey") 


#censor deaths after 8/25/16
#cdata2$censor<-ifelse(cdata2$DEATH_DATE_off > '2016-08-25',1,0)

#add time to event, one DX date and death flag 
cdata2$study_end<-as.Date("2016-08-25")
cdata2$dxDate<-as.Date(apply(cdata2[,10:14], 1, max))
cdata2$ttEvent<-ifelse(!is.na(cdata2$dx2death), cdata2$dx2death*365.25, cdata2$study_end-cdata2$dxDate)
#death flag for people who died on or before 8/25/16
#cdata2$death<-ifelse(!is.na(cdata2$dx2death),1,0)
cdata2$death<-ifelse(cdata2$DEATH_DATE_off > '1900-01-01' & cdata2$DEATH_DATE_off < '2016-08-26',1,0)

#check if any extreme outliers for DX date 
min(cdata2$dxDate)
hist(cdata2$dxDate, breaks = 20)
dim(cdata2[cdata2$dxDate < '2000-01-01',])
cdata2<-cdata2[cdata2$dxDate > '2000-01-01',]

#remove null race (no remaining rows are null/na for gender or race)
cdata2<-cdata2[cdata2$race!="NULL",]

write.csv(cdata2, "cancer_data_supp.csv")

##########################################################################

birth_yr<-cdata[cdata$BIRTH_DATE_off] #get year
hist(cdata$BIRTH_DATE_off) 

#brief exploratory analysis
#freqs for cancer types
ct_freq<-as.data.frame(table(cdata2$cancer_type))
death_freq<-as.data.frame(table(cdata2$cancer_type[cdata2$DEATH_DATE_off > '1900-01-01']))
death_censor<-as.data.frame(table(cdata2$cancer_type[cdata2$DEATH_DATE_off > '1900-01-01' & cdata2$DEATH_DATE_off < '2016-08-25']))
med_ttEvent<-aggregate(cdata2$ttEvent, by = list(cdata2$cancer_type), FUN = median)
perc_death<-round(death_freq[2]/ct_freq[2]*100,2)
perc_death_censor<-round(death_censor[2]/ct_freq[2]*100,2)
summ_table<-cbind(ct_freq, total_deaths = death_freq[2], total_death_perc = perc_death,
                  non_censor_death = death_censor[2], perc_non_censor_death = perc_death_censor,
                  med_ttEvent[2])
colnames(summ_table)<-c("Cancer Type","Total Patients","Patient Total Deaths","Percent Total Death",
"Non-Censored Deaths", "Percent Non-Censored Deaths", "Median Time to Death")
View(summ_table)

#bar chart showing total frequencies and male v female by cancer type
cgender<-as.data.frame(table(cdata2$sex, cdata2$cancer_type))
colnames(cgender)[2]<-"Cancer Type"

ggplot(data = cgender, aes(x=Var1, y=Freq, fill = Var2))+
  geom_bar(stat = "identity") + 
  labs(title = "Cancer Type by Gender", 
       y = "Patient Count", x = "Gender", color = "Cancer Type")+
  scale_y_continuous(labels=function(n){format(n, scientific = FALSE)})+
  scale_fill_brewer(palette="Greens")+
  theme_minimal()

#bar chart showing total frequencies and race by cancer type
#this is useless - find percentage of each cancer from each race and chart 
crace<-as.data.frame(table(cdata2$race, cdata2$cancer_type))

ggplot(data = crace, aes(x=Var1, y=Freq, fill = Var2))+
  geom_bar(stat = "identity") + 
  labs(title = "Cancer Type by Race", 
       y = "Patient Count", x = "Race",
       color = "Cancer Type")+
  scale_y_continuous(labels=function(n){format(n, scientific = FALSE)})+
  scale_color_brewer(palette="Greens")

perc_race<-as.data.frame(unique(cdata2$race))
colnames(perc_race)<-"Race"
perc_race<-perc_race[order(perc_race$`unique(cdata2$race)`),]
target<-c("Black","White","Declined","Pacific Islander","Unknown","Asian","Multiple","Native","Null")
cdata2$cancer_type<-as.factor(cdata2$cancer_type)

for (ct in levels(cdata2$cancer_type)) {
  z<-cdata2[cdata2$cancer_type==get(ct),]
  zdf<-as.data.frame(table(z$race))
  zdf<-zdf[order(zdf$Var1),]
  #row.names(zdf)<-c("Black","White","Declined","Pacific Islander","Unknown","Asian","Multiple","Native","Null")
  perc<-cbind(zdf[1], zdf[2]/dim(z)[1]*100)
  perc_race<-merge(perc_race, perc, by.x = "Race", by.y = "Var1")
  #perc_race<-cbind(perc, perc_race)
}

colnames(perc_race)<-c("Thyroid Cancer", "")

#box and violin plot showing mean age of cancer DX for each cancer type 
boxplot(DXage~cancer_type, data=cdata2, main = "Age at Diagnosis by Cancer Type",
        xlab = "Cancer Type", ylab = "Age at Diagnosis", col = "springgreen")

ggplot(data = cdata2, aes(x=cancer_type, y=DXage))+
  geom_violin(fill='#A4A4A4', color="darkgreen")+
  geom_boxplot(width = 0.1)+
  labs(title = "Mean Age of Diagnosis per Cancer Type", x = "Cancer Type", y = "Diagnosis Age")
  theme_minimal()

#heat map over chart showing race x cancer type with value = mean age of DX 
#2 people with null race 
means_DXage_race<-aggregate(cdata2$DXage, by = list(cdata2$cancer_type, cdata2$race), mean)

means_Dxage_race2<-as.data.frame(with(cdata2, tapply(DXage, list(cancer_type,race), mean)))
means_Dxage_race2<-means_Dxage_race2[-6]
colnames(means_Dxage_race2)<-c("Native","Asian","Black","Multiracial","Pac. Islander","Pt. Declined", "Unknown","White")
means_Dxage_race2<-as.data.frame(t(means_Dxage_race2))

formattable(means_Dxage_race2, list(
  "Colorectal Cancer" = color_tile("red","white"),
  "Lung Cancer" = color_tile("red","white"),
  "Melanoma" = color_tile("red","white"),
  "Pancreatic Cancer" = color_tile("red","white"),
  "Thyroid Cancer" = color_tile("red","white"))
  )
