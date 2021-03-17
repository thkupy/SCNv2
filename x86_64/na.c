/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__na
#define _nrn_initial _nrn_initial__na
#define nrn_cur _nrn_cur__na
#define _nrn_current _nrn_current__na
#define nrn_jacob _nrn_jacob__na
#define nrn_state _nrn_state__na
#define _net_receive _net_receive__na 
#define _f_trates _f_trates__na 
#define rates rates__na 
#define states states__na 
#define trates trates__na 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gnabar _p[0]
#define ina _p[1]
#define gna _p[2]
#define m _p[3]
#define h _p[4]
#define ena _p[5]
#define Dm _p[6]
#define Dh _p[7]
#define _g _p[8]
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_rates(void);
 static void _hoc_states(void);
 static void _hoc_trates(void);
 static void _hoc_vtrap(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_na", _hoc_setdata,
 "rates_na", _hoc_rates,
 "states_na", _hoc_states,
 "trates_na", _hoc_trates,
 "vtrap_na", _hoc_vtrap,
 0, 0
};
#define vtrap vtrap_na
 extern double vtrap( double , double );
 /* declare global and static user variables */
#define htau htau_na
 double htau = 0;
#define hinf hinf_na
 double hinf = 0;
#define mtau mtau_na
 double mtau = 0;
#define minf minf_na
 double minf = 0;
#define usetable usetable_na
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "gnabar_na", 0, 1e+09,
 "usetable_na", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "mtau_na", "ms",
 "htau_na", "ms",
 "gnabar_na", "mho/cm2",
 "ina_na", "mA/cm2",
 "gna_na", "mho/cm2",
 0,0
};
 static double delta_t = 1;
 static double h0 = 0;
 static double m0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "minf_na", &minf_na,
 "hinf_na", &hinf_na,
 "mtau_na", &mtau_na,
 "htau_na", &htau_na,
 "usetable_na", &usetable_na,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"na",
 "gnabar_na",
 0,
 "ina_na",
 "gna_na",
 0,
 "m_na",
 "h_na",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 9, _prop);
 	/*initialize range parameters*/
 	gnabar = 0.07958;
 	_prop->param = _p;
 	_prop->param_size = 9;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static void _initlists();
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _na_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 9, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 na /home/kuenzel/Dokumente/Python/SCNv2/mech/na.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double _zmexp , _zhexp ;
 static double _zq10 ;
 static double *_t_minf;
 static double *_t__zmexp;
 static double *_t_hinf;
 static double *_t__zhexp;
static int _reset;
static char *modelname = "klt.mod  The low threshold conductance of cochlear nucleus neurons";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_trates(double);
static int rates(double);
static int states();
static int trates(double);
 static void _n_trates(double);
 
static int  states (  ) {
   trates ( _threadargscomma_ v ) ;
   m = m + _zmexp * ( minf - m ) ;
   h = h + _zhexp * ( hinf - h ) ;
   
/*VERBATIM*/
	return 0;
  return 0; }
 
static void _hoc_states(void) {
  double _r;
   _r = 1.;
 states (  );
 hoc_retpushx(_r);
}
 
static int  rates (  double _lv ) {
   _zq10 = pow( 3.0 , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   minf = 1.0 / ( 1.0 + exp ( - ( _lv + 38.0 ) / 7.0 ) ) ;
   hinf = 1.0 / ( 1.0 + exp ( ( _lv + 65.0 ) / 6.0 ) ) ;
   mtau = ( 10.0 / ( 5.0 * exp ( ( _lv + 60.0 ) / 18.0 ) + 36.0 * exp ( - ( _lv + 60.0 ) / 25.0 ) ) ) + 0.04 ;
   htau = ( 100.0 / ( 7.0 * exp ( ( _lv + 60.0 ) / 11.0 ) + 10.0 * exp ( - ( _lv + 60.0 ) / 25.0 ) ) ) + 0.6 ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 static double _mfac_trates, _tmin_trates;
 static void _check_trates();
 static void _check_trates() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_dt;
  static double _sav_celsius;
  if (!usetable) {return;}
  if (_sav_dt != dt) { _maktable = 1;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_trates =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_trates)/300.; _mfac_trates = 1./_dx;
   for (_i=0, _x=_tmin_trates; _i < 301; _x += _dx, _i++) {
    _f_trates(_x);
    _t_minf[_i] = minf;
    _t__zmexp[_i] = _zmexp;
    _t_hinf[_i] = hinf;
    _t__zhexp[_i] = _zhexp;
   }
   _sav_dt = dt;
   _sav_celsius = celsius;
  }
 }

 static int trates(double _lv){ _check_trates();
 _n_trates(_lv);
 return 0;
 }

 static void _n_trates(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_trates(_lv); return; 
}
 _xi = _mfac_trates * (_lv - _tmin_trates);
 if (isnan(_xi)) {
  minf = _xi;
  _zmexp = _xi;
  hinf = _xi;
  _zhexp = _xi;
  return;
 }
 if (_xi <= 0.) {
 minf = _t_minf[0];
 _zmexp = _t__zmexp[0];
 hinf = _t_hinf[0];
 _zhexp = _t__zhexp[0];
 return; }
 if (_xi >= 300.) {
 minf = _t_minf[300];
 _zmexp = _t__zmexp[300];
 hinf = _t_hinf[300];
 _zhexp = _t__zhexp[300];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 minf = _t_minf[_i] + _theta*(_t_minf[_i+1] - _t_minf[_i]);
 _zmexp = _t__zmexp[_i] + _theta*(_t__zmexp[_i+1] - _t__zmexp[_i]);
 hinf = _t_hinf[_i] + _theta*(_t_hinf[_i+1] - _t_hinf[_i]);
 _zhexp = _t__zhexp[_i] + _theta*(_t__zhexp[_i+1] - _t__zhexp[_i]);
 }

 
static int  _f_trates (  double _lv ) {
   double _ltinc ;
 rates ( _threadargscomma_ _lv ) ;
   _ltinc = - dt * _zq10 ;
   _zmexp = 1.0 - exp ( _ltinc / mtau ) ;
   _zhexp = 1.0 - exp ( _ltinc / htau ) ;
    return 0; }
 
static void _hoc_trates(void) {
  double _r;
    _r = 1.;
 trates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double vtrap (  double _lx , double _ly ) {
   double _lvtrap;
 if ( fabs ( _lx / _ly ) < 1e-6 ) {
     _lvtrap = _ly * ( 1.0 - _lx / _ly / 2.0 ) ;
     }
   else {
     _lvtrap = _lx / ( exp ( _lx / _ly ) - 1.0 ) ;
     }
   
return _lvtrap;
 }
 
static void _hoc_vtrap(void) {
  double _r;
   _r =  vtrap (  *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ hoc_execerror("na", "cannot be used with CVODE"); return 0;}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  h = h0;
  m = m0;
 {
   trates ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ena = _ion_ena;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   gna = gnabar * ( pow( m , 3.0 ) ) * h ;
   ina = gna * ( v - ena ) ;
   }
 _current += ina;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ena = _ion_ena;
 _g = _nrn_current(_v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ena = _ion_ena;
 { error =  states();
 if(error){fprintf(stderr,"at line 59 in file na.mod:\n    \n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
   _t_minf = makevector(301*sizeof(double));
   _t__zmexp = makevector(301*sizeof(double));
   _t_hinf = makevector(301*sizeof(double));
   _t__zhexp = makevector(301*sizeof(double));
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/home/kuenzel/Dokumente/Python/SCNv2/mech/na.mod";
static const char* nmodl_file_text = 
  "TITLE klt.mod  The low threshold conductance of cochlear nucleus neurons\n"
  "\n"
  "COMMENT\n"
  "\n"
  "NEURON implementation of Jason Rothman's measurements of VCN conductances.\n"
  "\n"
  "This file implements the average brain sodium current used in the Rothman model.\n"
  "In the absence of direct measurements in the VCN, this is a fair assumption.\n"
  "The model differs from the one used in Rothman et al, (1993) in that the steep\n"
  "voltage dependence of recovery from inactivation in that model is missing. This\n"
  "may affect the refractory period. To use the other model, use najsr.mod instead.\n"
  "\n"
  "Original implementation by Paul B. Manis, April (JHU) and Sept, (UNC)1999.\n"
  "\n"
  "File split implementaiton, April 1, 2004.\n"
  "\n"
  "Contact: pmanis@med.unc.edu\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "UNITS {\n"
  "        (mA) = (milliamp)\n"
  "        (mV) = (millivolt)\n"
  "        (nA) = (nanoamp)\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "        SUFFIX na\n"
  "        USEION na READ ena WRITE ina\n"
  "        RANGE gnabar, gna, ina\n"
  "        GLOBAL hinf, minf, htau, mtau\n"
  "}\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "PARAMETER {\n"
  "        v (mV)\n"
  "        celsius = 22 (degC)  : model is defined on measurements made at room temp in Baltimore\n"
  "        dt (ms)\n"
  "        ena (mV)\n"
  "        gnabar =  0.07958 (mho/cm2) <0,1e9>\n"
  "}\n"
  "\n"
  "STATE {\n"
  "        m h\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    ina (mA/cm2) \n"
  "    gna (mho/cm2)\n"
  "    minf hinf\n"
  "    mtau (ms) htau (ms)\n"
  "    }\n"
  "\n"
  "LOCAL mexp, hexp\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states\n"
  "    \n"
  "    gna = gnabar*(m^3)*h\n"
  "    ina = gna*(v - ena)\n"
  "\n"
  "}\n"
  "\n"
  "UNITSOFF\n"
  "\n"
  "INITIAL {\n"
  "    trates(v)\n"
  "    m = minf\n"
  "    h = hinf\n"
  "}\n"
  "\n"
  "PROCEDURE states() {  :Computes state variables m, h, and n\n"
  "	trates(v)      :             at the current v and dt.\n"
  "	m = m + mexp*(minf-m)\n"
  "	h = h + hexp*(hinf-h)\n"
  "VERBATIM\n"
  "	return 0;\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "LOCAL q10\n"
  "\n"
  "PROCEDURE rates(v) {  :Computes rate and other constants at current v.\n"
  "                      :Call once from HOC to initialize inf at resting v.\n"
  "\n"
  "	q10 = 3^((celsius - 22)/10) : if you don't like room temp, it can be changed!\n"
  "\n"
  ": average sodium channel\n"
  "    minf = 1 / (1+exp(-(v + 38) / 7))\n"
  "    hinf = 1 / (1+exp((v + 65) / 6))\n"
  "\n"
  "    mtau =  (10 / (5*exp((v+60) / 18) + 36*exp(-(v+60) / 25))) + 0.04\n"
  "    htau =  (100 / (7*exp((v+60) / 11) + 10*exp(-(v+60) / 25))) + 0.6\n"
  "}\n"
  "\n"
  "PROCEDURE trates(v) {  :Computes rate and other constants at current v.\n"
  "                      :Call once from HOC to initialize inf at resting v.\n"
  "	LOCAL tinc\n"
  "	TABLE minf, mexp, hinf, hexp\n"
  "	DEPEND dt, celsius FROM -150 TO 150 WITH 300\n"
  "\n"
  "    rates(v)    : not consistently executed from here if usetable_hh == 1\n"
  "        : so don't expect the tau values to be tracking along with\n"
  "        : the inf values in hoc\n"
  "\n"
  "	tinc = -dt * q10\n"
  "	mexp = 1 - exp(tinc/mtau)\n"
  "	hexp = 1 - exp(tinc/htau)\n"
  "	}\n"
  "\n"
  "FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.\n"
  "        if (fabs(x/y) < 1e-6) {\n"
  "                vtrap = y*(1 - x/y/2)\n"
  "        }else{\n"
  "                vtrap = x/(exp(x/y) - 1)\n"
  "        }\n"
  "}\n"
  "\n"
  "UNITSON\n"
  ;
#endif
