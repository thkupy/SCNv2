#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _gclamp_reg(void);
extern void _gfluct2_reg(void);
extern void _kht_reg(void);
extern void _klt_reg(void);
extern void _leak_reg(void);
extern void _na_fast_reg(void);
extern void _na_reg(void);
extern void _synstim_reg(void);
extern void _vecevent_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," \"./mech//gclamp.mod\"");
    fprintf(stderr," \"./mech//gfluct2.mod\"");
    fprintf(stderr," \"./mech//kht.mod\"");
    fprintf(stderr," \"./mech//klt.mod\"");
    fprintf(stderr," \"./mech//leak.mod\"");
    fprintf(stderr," \"./mech//na_fast.mod\"");
    fprintf(stderr," \"./mech//na.mod\"");
    fprintf(stderr," \"./mech//synstim.mod\"");
    fprintf(stderr," \"./mech//vecevent.mod\"");
    fprintf(stderr, "\n");
  }
  _gclamp_reg();
  _gfluct2_reg();
  _kht_reg();
  _klt_reg();
  _leak_reg();
  _na_fast_reg();
  _na_reg();
  _synstim_reg();
  _vecevent_reg();
}
