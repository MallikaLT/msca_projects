library(survival)
library(ggplot2)
library(survminer)

dat<-read.csv("cancer_data_supp.csv")
dat[,c(2,3,11,12,13,14,15)]<-lapply(dat[,c(2,3,11,12,13,14,15)],
                                      function(z) as.Date(as.character(z))) #, format = "%m/%d/%Y"))

#separate into DF for each cancer type
thyroid<-dat[dat$cancer_type=="Thyroid Cancer",]
lung<-dat[dat$cancer_type=="Lung Cancer",]
melanoma<-dat[dat$cancer_type=="Melanoma",]
colorectal<-dat[dat$cancer_type=="Colorectal Cancer",]
pancreatic<-dat[dat$cancer_type=="Pancreatic Cancer",]


##Thyroid Cancer 
#gender KM
thyroid_gender<-survfit(Surv(time = thyroid$ttEvent, event = thyroid$death)~thyroid$sex)
thyroid_gender
summary(thyroid_gender)

thyroid_ggsurv<-ggsurvplot(thyroid_gender, data = thyroid, ylim=c(0.85,1.00),
           risk.table = TRUE, ncensor.plot = TRUE, break.time.by = 1000,
           ggtheme = theme_minimal(), xlim = c(0,6000))
thyroid_ggsurv
thyroid_ggsurv$plot+labs(title = "Thyroid Cancer Survival by Gender")
#log-rank test to see if diff is statistically significant - not
survdiff(Surv(time = thyroid$ttEvent, event = thyroid$death)~thyroid$sex)
#Cox PH for gender (curves don't cross)
thyroid_cox_gender<-coxph(Surv(time = ttEvent, event = death)~sex, data = thyroid)
summary(thyroid_cox_gender)
ggsurvplot(survfit(thyroid_cox_gender), ylim = c(.85, 1.00), theme_minimal())

#race KM
thyroid_race<-survfit(Surv(time = thyroid$ttEvent, event = thyroid$death)~thyroid$race)
thyroid_race
summary(thyroid_race)

thyroid_ggsurv_race<-ggsurvplot(thyroid_race, data = thyroid, ylim=c(0.90,1.00),
                                risk.table = TRUE, ncensor.plot = TRUE, break.time.by = 2000,
                                ggtheme = theme_minimal(), xlim = c(0,6200))
thyroid_ggsurv_race$plot+labs(title = "Thyroid Cancer Survival by Race")
#log-rank test to see if diff is sig - IT IS SIGNIF.
survdiff(Surv(time = thyroid$ttEvent, event = thyroid$death)~thyroid$race)
#survival curves cross 

#age KM
thyroid_age<-survfit(Surv(time = thyroid$ttEvent, event = thyroid$death)~thyroid$ageCat)
thyroid_age
summary(thyroid_age)

thyroid_ggsurv_age<-ggsurvplot(thyroid_age, data = thyroid, ylim=c(0.95,1.00),
                               risk.table = TRUE, ncensor.plot = TRUE, break.time.by = 2000,
                               ggtheme = theme_minimal(), xlim = c(0,6000))
thyroid_ggsurv
thyroid_ggsurv_age$plot+labs(title = "Thyroid Cancer Survival by Age")
#log rank - not signif
survdiff(Surv(time = thyroid$ttEvent, event = thyroid$death)~thyroid$ageCat)
#crossing survival curves

##Lung Cancer 
#gender
lung_gender<-survfit(Surv(time = lung$ttEvent, event = lung$death)~lung$sex)
lung_gender
lung_ggsurv<-ggsurvplot(lung_gender, data = lung, risk.table = TRUE, ncensor.plot = TRUE,
           xlim = c(0,3000), break.time.by = 500, ylim = c(0.70, 1.00))
lung_ggsurv$plot+labs(title = "Lung Cancer Survival by Gender")
#curves cross - no CoxPH 
#log rank
survdiff(Surv(time = lung$ttEvent, event = lung$death)~lung$sex)

#race
lung_race<-survfit(Surv(time = lung$ttEvent, event = lung$death)~lung$race)
lung_race
summary(lung_race)
lung_ggsurv_race<-ggsurvplot(lung_race, data = lung, risk.table = TRUE, ncensor.plot = TRUE,
                        xlim = c(0,4000), break.time.by = 500, ylim = c(0.70, 1.00))
lung_ggsurv_race$plot+labs(title = "Lung Cancer Survival by Race")
#log rank overall 
survdiff(Surv(time = lung$ttEvent, event = lung$death)~lung$race)
#maybe CoxPH between black and white
bwlung<-lung[lung$race=="Black/African-American" | lung$race=="White",]
bwlung$race<-factor(bwlung$race)
bwlung_cox<-coxph(Surv(time = ttEvent, event = death)~race, data = bwlung)
summary(bwlung_cox)
bwlung_plot<-ggsurvplot(survfit(bwlung_cox), ylim = c(.85, 1.00), theme_minimal())
bwlung_plot$plot+labs(title="Cox PH Lung Cancer Survival by Race - Black/White Only")

#age 
lung_age<-survfit(Surv(time = lung$ttEvent, event = lung$death)~lung$ageCat)
lung_age
summary(lung_age)
lung_ggsurv_age<-ggsurvplot(lung_age, data = lung, risk.table = TRUE, ncensor.plot = TRUE,
                             xlim = c(0,4000), break.time.by = 500, ylim = c(0.70, 1.00))
lung_ggsurv_age$plot+labs(title = "Lung Cancer Survival by Age")
#almost all curves cross 
#log rank
survdiff(Surv(time = lung$ttEvent, event = lung$death)~lung$ageCat)

##Melanoma 
#gender
melanoma_gender<-survfit(Surv(time = melanoma$ttEvent, event = melanoma$death)~melanoma$sex)
melanoma_ggsurv1<-ggsurvplot(melanoma_gender, data = melanoma, risk.table = TRUE, ncensor.plot = TRUE,
           xlim = c(0,2000), break.time.by = 400, ylim = c(0.85, 1.00))
melanoma_ggsurv2<-ggsurvplot(melanoma_gender, data = melanoma, risk.table = TRUE, ncensor.plot = TRUE,
                             xlim = c(0,7500), break.time.by = 1000, ylim = c(0.50, 1.00))
melanoma_ggsurv1$plot+labs(title = "Melanoma Survival by Gender - Truncated")
melanoma_ggsurv2$plot+labs(title = "Melanoma Survival by Gender - All")
#log rank - not signif
survdiff(Surv(time = melanoma$ttEvent, event = melanoma$death)~melanoma$sex)

#race
melanoma_race<-survfit(Surv(time = melanoma$ttEvent, event = melanoma$death)~melanoma$race)
melanoma_race
melanoma_ggsurv_race<-ggsurvplot(melanoma_race, data = melanoma, risk.table = TRUE, ncensor.plot = TRUE,
                             xlim = c(0,4800), break.time.by = 400, ylim = c(0.85, 1.00))

melanoma_ggsurv_race$plot+labs(title = "Melanoma Survival by Race")
#log rank - not sig 
survdiff(Surv(time = melanoma$ttEvent, event = melanoma$death)~melanoma$race)

#age
melanoma_ageCat<-survfit(Surv(time = melanoma$ttEvent, event = melanoma$death)~melanoma$ageCat)
melanoma_ageCat
melanoma_ggsurv_ageCat<-ggsurvplot(melanoma_ageCat, data = melanoma, risk.table = TRUE, ncensor.plot = TRUE,
                                 xlim = c(0,4000), break.time.by = 500, ylim = c(0.85, 1.00))

melanoma_ggsurv_ageCat$plot+labs(title = "Melanoma Survival by Age")
#log rank - AGE IS SIGNIF
survdiff(Surv(time = melanoma$ttEvent, event = melanoma$death)~melanoma$ageCat)


##Colorectal Cancer
#gender - signif
colorectal_gender<-survfit(Surv(time = colorectal$ttEvent, event = colorectal$death)~colorectal$sex)
col_ggsurv<-ggsurvplot(colorectal_gender, data = colorectal, risk.table = TRUE, ncensor.plot = TRUE,
           xlim = c(0,6000), break.time.by = 500, ylim = c(0.80, 1.00))
col_ggsurv$plot+labs(title = "Colorectal Cancer Survival by Gender")
#NOTE: no CoxPH here - violates assumption since the survival curves cross 
survdiff(Surv(time = colorectal$ttEvent, event = colorectal$death)~colorectal$race)

#race
colorectal_race<-survfit(Surv(time = colorectal$ttEvent, event = colorectal$death)~colorectal$race)
colorectal_race
colorectal_ggsurv_race<-ggsurvplot(colorectal_race, data = colorectal, risk.table = TRUE, ncensor.plot = TRUE,
                                 xlim = c(0,5000), break.time.by = 400, ylim = c(0.85, 1.00))

colorectal_ggsurv_race$plot+labs(title = "Colorectal Cancer Survival by Race")
#log rank - sig 
survdiff(Surv(time = colorectal$ttEvent, event = colorectal$death)~colorectal$race)
#try Cox PH bt white and black 
bwCR<-colorectal[colorectal$race=="Black/African-American" | colorectal$race=="White",]
bwCR$race<-factor(bwCR$race)
bwCR_cox<-coxph(Surv(time = ttEvent, event = death)~race, data = bwCR)
summary(bwCR_cox)
bwCR_plot<-ggsurvplot(survfit(bwCR_cox), ylim = c(.85, 1.00), theme_minimal())
bwCR_plot$plot+labs(title="Cox PH Colorectal Cancer Survival by Race - Black/White Only")
#Cox PH for race and gender (NOTE: Natives don't have DX)
CR_cox_gender_race <- coxph(Surv(time = ttEvent, event = death) ~ race + sex + race*sex, data =  colorectal)

#age
colorectal_ageCat<-survfit(Surv(time = colorectal$ttEvent, event = colorectal$death)~colorectal$ageCat)
colorectal_ageCat
colorectal_ggsurv_ageCat<-ggsurvplot(colorectal_ageCat, data = colorectal, risk.table = TRUE, ncensor.plot = TRUE,
                                   xlim = c(0,4000), break.time.by = 500, ylim = c(0.85, 1.00))

colorectal_ggsurv_ageCat$plot+labs(title = "Colorectal Cancer Survival by Age")
#log rank - AGE IS NOT SIGNIF
survdiff(Surv(time = colorectal$ttEvent, event = colorectal$death)~colorectal$ageCat)
#try Cox PH

#Pancreatic Cancer
#gender
pancreatic_gender<-survfit(Surv(time = pancreatic$ttEvent, event = pancreatic$death)~pancreatic$sex)
pancreatic_ggsurv_gender<-ggsurvplot(pancreatic_gender, data = pancreatic, risk.table = TRUE, ncensor.plot = TRUE,
           break.time.by = 500, ylim = c(0.9, 1.0))
pancreatic_ggsurv_gender
pancreatic_ggsurv_gender$plot+labs(title = "Pancreatic Cancer Survival by Gender")
#NOTE: try CoxPH here - survival curves cross but barely
pan_cox_gender<-coxph(Surv(time = ttEvent, event = death) ~ sex, data = pancreatic)
#log rank not sig
survdiff(Surv(time = pancreatic$ttEvent, event = pancreatic$death)~pancreatic$sex)


#race
pancreatic_race<-survfit(Surv(time = pancreatic$ttEvent, event = pancreatic$death)~pancreatic$race)
pancreatic_race
pancreatic_ggsurv_race<-ggsurvplot(pancreatic_race, data = pancreatic, risk.table = TRUE, ncensor.plot = TRUE,
                                   xlim = c(0,3000), break.time.by = 400, ylim = c(0.85, 1.00))

pancreatic_ggsurv_race$plot+labs(title = "Pancreatic Cancer Survival by Race")
#log rank - signif
survdiff(Surv(time = pancreatic$ttEvent, event = pancreatic$death)~pancreatic$race)

#age
pancreatic_ageCat<-survfit(Surv(time = pancreatic$ttEvent, event = pancreatic$death)~pancreatic$ageCat)
pancreatic_ageCat
pancreatic_ggsurv_ageCat<-ggsurvplot(pancreatic_ageCat, data = pancreatic, risk.table = TRUE, ncensor.plot = TRUE,
                                     xlim = c(0,3000), break.time.by = 500, ylim = c(0.85, 1.00))

pancreatic_ggsurv_ageCat$plot+labs(title = "Pancreatic Cancer Survival by Age")
#log rank - AGE IS NOT SIGNIF
survdiff(Surv(time = pancreatic$ttEvent, event = pancreatic$death)~pancreatic$ageCat)
#try Cox PH
