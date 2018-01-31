libname proj "/home/mthanky0/Data Mining/Project";

proc import datafile = "/home/mthanky0/Data Mining/Project/thoracic_data.csv"
            dbms = csv out = proj.thoracic_data;
run;

proc contents data=proj.thoracic_data; run;

/*relabel columns with more descriptive labels*/
proc print proj.thoracic_data LABEL;
	label PRE4="forced_vital_cap";
	label PRE5="FEV";
	label PRE6="perf_status";
	label PRE7="pain";
	label PRE8="coughing_blood";
	label PRE9="short_breath";
	label PRE10="cough";
	label PRE11="weakness";
	label PRE14="tumor_size";
	label PRE17="diabetes";
	label PRE19="heart_attack_6mo";
	label PRE25="periph_arterial_dis";
	label PRE30="smoking";
	label PRE32="asthma";
	label AGE="age";
run;

/*create unique patient identifier*/
data proj.thoracic_data;
	set proj.thoracic_data;
	patient_id=_n_;
run;

/*check in Risk1Y=T is rare event*/
proc freq data=proj.thoracic_data nlevels;
	tables risk1Y; 
run;

proc contents data=proj.thoracic_data; run;

proc print data=proj.thoracic_data (obs=100); run;

/*get training and validation partitions*/
proc hpsample data=proj.thoracic_data out=proj.thoracic_part
	seed=20161117 samppct=70 partition;
	class RISK1Y DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
	var patient_id PRE4 PRE5 AGE;
	target RISK1Y;
run;
	
/*split into two separate partitions*/
data proj.thoracic_trng;
	set proj.thoracic_part;
	if _PARTIND_=1;
run;

data proj.thoracic_valid;
	set proj.thoracic_part;
	if _PARTIND_=0;
run;

/*cluster analysis for interval variables*/
filename OutScr '/home/mthanky0/Data Mining/Exam/thoracic_cluster_score_int.sas';
proc hpclus data = proj.thoracic_trng
            maxclusters = 20 seed = 20161117 noc = abc (minclusters = 2) maxiter=100 standardize=std;
   id patient_id;
   input PRE4 PRE5 AGE / level = interval;
   code file = OutScr;
run;

data proj.int_cluster_score;
   set proj.thoracic_valid;
   %include OutScr;  
run;

/*cluster for categorical variables - only this one can include the Risk Variable*/
filename ScrCat '/home/mthanky0/Data Mining/Exam/thoracic_cluster_score_cat.sas';
proc hpclus data = proj.thoracic_trng
            maxclusters = 20 seed = 20161117 maxiter=100;
   id patient_id;
   input RISK1Y DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32/ level = nominal;
   code file = ScrCat;
run;

data proj.cat_cluster_score;
   set proj.thoracic_valid;
   %include ScrCat;  
run;

/*decision tree - play around with tree depth - at depth of 8, more than one node had T, but at 3, only 1.  There may be a middle ground*/

filename TreeScr "/home/mthanky0/Data Mining/Project/_Tree_Scorecode_.sas";
proc hpsplit data = proj.thoracic_trng
             maxbranch = 2
             maxdepth = 3
             minleafsize = 4
             assignmissing = popular
             seed = 20161117
             nodes = detail
             plots=(wholetree(linkstyle=orthogonal linkwidth=constant)
                    zoomedtree(depth = 3 linkstyle=orthogonal linkwidth=constant nolegend));
   class RISK1Y DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
   model RISK1Y(event = "T") = PRE4 PRE5 AGE DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
   grow entropy;
   prune none;
   code file = TreeScr;
run;

filename TreeScr2 "/home/mthanky0/Data Mining/Project/_Tree_Scorecode2_.sas";
proc hpsplit data = proj.thoracic_trng
             maxbranch = 2
             maxdepth = 5
             minleafsize = 4
             assignmissing = popular
             seed = 20161117
             nodes = detail
             plots=(wholetree(linkstyle=orthogonal linkwidth=constant)
                    zoomedtree(depth = 5 linkstyle=orthogonal linkwidth=constant nolegend));
   class RISK1Y DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
   model RISK1Y(event = "T") = PRE4 PRE5 AGE DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
   grow entropy;
   prune none;
   code file = TreeScr2;
run;

filename TreeScr3 "/home/mthanky0/Data Mining/Project/_Tree_Scorecode3_.sas";
proc hpsplit data = proj.thoracic_trng
             maxbranch = 2
             maxdepth = 4
             minleafsize = 4
             assignmissing = popular
             seed = 20161117
             nodes = detail
             plots=(wholetree(linkstyle=orthogonal linkwidth=constant)
                    zoomedtree(depth = 4 linkstyle=orthogonal linkwidth=constant nolegend));
   class RISK1Y DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
   model RISK1Y(event = "T") = PRE4 PRE5 AGE DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
   grow entropy;
   prune none;
   code file = TreeScr3;
run;

filename TreeScr4 "/home/mthanky0/Data Mining/Project/_Tree_Scorecode4_.sas";
proc hpsplit data = proj.thoracic_trng
             maxbranch = 2
             maxdepth = 4
             minleafsize = 3
             assignmissing = popular
             seed = 20161117
             nodes = detail
             plots=(wholetree(linkstyle=orthogonal linkwidth=constant)
                    zoomedtree(depth = 4 linkstyle=orthogonal linkwidth=constant nolegend));
   class RISK1Y DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
   model RISK1Y(event = "T") = PRE4 PRE5 AGE DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
   grow entropy;
   prune none;
   code file = TreeScr4;
run;

/*looking at diff combos of variables*/
/*this takes out the strongest splitting vars AUC 78*/
filename TreeScr5 "/home/mthanky0/Data Mining/Project/_Tree_Scorecode_.sas";
proc hpsplit data = proj.thoracic_trng
             maxbranch = 2
             maxdepth = 6
             minleafsize = 4
             assignmissing = popular
             seed = 20161117
             nodes = detail
             plots=(wholetree(linkstyle=orthogonal linkwidth=constant)
                    zoomedtree(depth = 6 linkstyle=orthogonal linkwidth=constant nolegend));
   class RISK1Y DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
   model RISK1Y(event = "T") = AGE DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32;
   grow entropy;
   prune none;
   code file = TreeScr5;
run;

/*this only uses the strongest splitting vars*/
filename TreeScr6 "/home/mthanky0/Data Mining/Project/_Tree_Scorecode_.sas";
proc hpsplit data = proj.thoracic_trng
             maxbranch = 2
             maxdepth = 5
             minleafsize = 4
             assignmissing = popular
             seed = 20161117
             nodes = detail
             plots=(wholetree(linkstyle=orthogonal linkwidth=constant)
                    zoomedtree(depth = 5 linkstyle=orthogonal linkwidth=constant nolegend));
   class RISK1Y DGN PRE8 PRE9 PRE10 PRE14 PRE17;
   model RISK1Y(event = "T") = PRE4 PRE5 AGE DGN PRE8 PRE9 PRE10 PRE14 PRE17 ;
   grow entropy;
   prune none;
   code file = TreeScr6;
run;

filename TreeScr7 "/home/mthanky0/Data Mining/Project/_Tree_Scorecode_.sas";
proc hpsplit data = proj.thoracic_trng
             maxbranch = 2
             maxdepth = 4
             minleafsize = 4
             assignmissing = popular
             seed = 20161117
             nodes = detail
             plots=(wholetree(linkstyle=orthogonal linkwidth=constant)
                    zoomedtree(depth = 4 linkstyle=orthogonal linkwidth=constant nolegend));
   class RISK1Y DGN PRE6 PRE8 PRE9 PRE14 PRE17 PRE19 PRE30;
   model RISK1Y(event = "T") = AGE DGN PRE6 PRE8 PRE9 PRE14 PRE17 PRE19 PRE30 ;
   grow entropy;
   prune none;
   code file = TreeScr7;
run;

/*using Tree5 score, score validation partition*/
data proj.TreeScoreData;
   set proj.thoracic_valid;
  %include TreeScr5;
run;

data proj.TreeScoreData;
   set proj.TreeScoreData;
  Keep RISK1Y P_RISK1YT;
run;

/*70/470 obs are death = 14.89%*/

%include '/home/mthanky0/Data Mining/Project/compute_model_metric.sas';

%compute_model_metric
(
   ScoreData = TreeScoreData,
   DepVar = RISK1Y,
   EventValue = 1,
   EventPredProb = P_RISK1YT,
   EventProbThreshold = 14.89,
   ModelFit = proj.thoracic_Validation_ModelFit,
   Debug = N
);

/*logistic regression*/
filename LogScr "/home/mthanky0/Data Mining/Project/thoraciclogis.sas";
proc hplogistic data=proj.thoracic_trng;
	class DGN PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32 RISK1Y/param =glm;
	model RISK1Y (event="T")= DGN AGE PRE4 PRE5 PRE6 PRE7 PRE8 PRE9 PRE10 PRE11 PRE14 PRE17 PRE19 PRE25 PRE30 PRE32 /link=logit;
	selection method=none;
	code file = LogScr;
	ods output proj.parameterestimates;
run;

/*score the validation data for logistic regression*/
data proj.LogScoreData;
   set proj.thoracic_valid;
  %include LogScr;
run;


