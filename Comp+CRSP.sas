%let path=P:/My SAS Files/Assignment4.1;
%let datapath=&path/data;
%let comppath=Q:/Data-ReadOnly/COMP;
%let crsppath=Q:/Data-ReadOnly/CRSP;
%let scratchpath=Q:/Scratch;
%let userpath=Q:/Users/ajarrett3;

* Create the libnames for SAS datasets;
libname comp "&comppath";
libname crsp "&crsppath";

* Import Winsorize macro;
%include "&crsppath/winsor.sas";

* Filter by years;
%let startyear=1995;
%let endyear=2008;
%let yearfilter=year(date) >= &startyear and year(date) <= &endyear;

* Balance sheet variables;
%let bsvar=at lt act lct intan;

* Cash flow variables;
%let cfvar=ni ibc xidoc dpc txdc esubc sppiv fopo fsrco exre;

* Variables used for filtering data set;
%let filtervar=indfmt datafmt popsrc consol scf sich compst cik;

* Vars needed from the data set;
%let keepfunda=gvkey cusip fyear tic conm mkvalt &bsvar &cfvar &filtervar;

data funda;
	set comp.funda (
		where = (
			fyear in (&startyear.:&endyear.) and 
			indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C' and
			scf in (1, 2, 3, 7) and compst ne 'AB' and
			not((sich>=6000 and sich<=6999) or (sich>=4900 and sich<=4999)) and
			cik ne ""
		)
		keep=&keepfunda);

	* Recode missing values to 0;
	array change _numeric_;
		do over change;
			if change=. then change=0;
		end;
	
	* Calculate accounting vars here;
	if at ^= 0 then do;
		nita = ni / at;
		tlta = lt / at;
		if mkvalt ^= 0 then do;
			book_market = (at - intan - lt) / mkvalt;
		end;
		else do;
			book_market = .;
		end;
	end;
	else do;
		nita = .;
		tlta = .;
		book_market = .;
	end;

	if lct ^= 0 then cacl = act / lct;
	else cacl = .;

	* Calculate internal cash flow;
	if sich in (1:3) then icf = ibc + xidoc + dpc + txdc + esubc + sppiv + fopo + fsrco;
	else icf = ibc + xidoc + dpc + txdc + esubc + sppiv + fopo + exre;

	* Net assets = total assets - current liabilities;
	na = at - lct;

	* Calculate internal cash flow to net assets;
	if na ^= 0 then icf_na = icf / na;
	else icf_na = .;
	
	label nita="Net Income to Total Assets" tlta="Total Liabilities to Total Assets" 
		  cacl="Current Assets to Current Liabilities" icf_na="Internal Cash Flow to Net Assets";
run;

* Lag the Funda data;
data funda;
	set funda;
	year = fyear + 1;
run;

* We only want to keep these vars from DSF;
%let keepdsf=date cusip permno permco ret prc shrout askhi bidlo vol retx;

* Pull in the DSF;
data dsf;
	informat date yymmdd8.;
	set crsp.dsf (
		where=(
			&yearfilter
		)
		keep=&keepdsf
	);
	format date mmddyy10.;
	prc=abs(prc);
	mkt_val=shrout*prc;
	year=YEAR(date);

	label mkt_val="Market Cap";
run;

* Use permco, date and descending mkt_val to sort by firm id and market cap.
* and then use another sort with nodupkey to only select the highest market 
* value stock for the firm;
proc sort data=dsf;
	by permco date descending mkt_val;
run;
proc sort data=dsf nodupkey;
	by permco date;
run;

%let keepdsi=date vwretd totval;

* Pull in the DSI data to get market value/indexes;
data dsi;
	set crsp.dsi (
		where=( &yearfilter )
		keep=&keepdsi
	);
run;

* Merge DSF and DSI based on date;
proc sql;
	create table crsp as
	select a.*, b.vwretd, b.totval, log(a.mkt_val / b.totval) as rsiz
	from dsf as a, dsi as b
	where a.date = b.date;
quit;

* Delete DSF and DSI data sets;
proc delete data=dsf dsi;
run;

proc sql;
	create table merged as
	select b.cusip, b.permno, a.cik, b.date, a.year, a.fyear, a.tic, a.conm, a.mkvalt, a.at, a.intan, a.lt, a.book_market, a.tlta, a.cacl, a.nita, a.icf_na, b.rsiz, b.mkt_val, b.prc, b.ret, b.askhi, b.bidlo, b.vol, b.retx, b.vwretd, b.totval
	from funda as a, crsp as b
	where substr(a.cusip, 1,6) = substr(b.cusip, 1,6)
		and a.fyear = b.year;
quit;

* Sort merged data by cusip, permno, and then date;
proc sort data=merged;
	by cusip permno date;
run;

%let winsorvars=mkvalt at book_market nita tlta cacl icf_na rsiz prc vwretd totval mkt_val askhi bidlo vol retx;

* Use the winsorize macro to eliminate extreme outliers;
%winsor(dsetin=merged, dsetout=merged_winsorized, vars=&winsorvars, byvar=none, pctl=1 99);

* Clean up unused data sets from winsor;
proc delete data=xtemp xtemp_pctl;
run;

* Write data to disk;
data "&scratchpath./crsp_comp";
	set merged_winsorized;
run;
