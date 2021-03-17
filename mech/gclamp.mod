COMMENT
%This is a dynamic clamp (a.k.a conductance clamp). To use it, play a
%stimulus conductance vector into the g range variable. Set the reversal
%potential of the conductance with the e range variable. Record the
%injected current with the i range variable.
ENDCOMMENT

NEURON {
POINT_PROCESS GClamp
RANGE g, i, e
NONSPECIFIC_CURRENT i
}

UNITS {
(mV) = (millivolt)
(uS) = (micromho)
(nA) = (nanoamp)
}

PARAMETER {
g (micromho)
e (millivolt)
v (millivolt)
}
ASSIGNED { i (nanoamp) }

INITIAL {
i = g*(v-e)
}

BREAKPOINT {
i=g*(v-e)
}


