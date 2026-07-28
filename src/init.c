#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

SEXP fastgbm_fit_cpp(SEXP xSEXP, SEXP ySEXP, SEXP objectiveSEXP, SEXP ntreesSEXP,
                     SEXP learning_rateSEXP, SEXP max_depthSEXP, SEXP min_node_sizeSEXP,
                     SEXP max_binsSEXP, SEXP subsampleSEXP, SEXP colsampleSEXP,
                     SEXP lambdaSEXP, SEXP gammaSEXP, SEXP min_child_weightSEXP,
                     SEXP seedSEXP, SEXP verboseSEXP);
SEXP fastgbm_predict_cpp(SEXP modelSEXP, SEXP xSEXP, SEXP typeSEXP);
SEXP fastgbm_importance_cpp(SEXP modelSEXP);

static const R_CallMethodDef CallEntries[] = {
    {"fastgbm_fit_cpp", (DL_FUNC) &fastgbm_fit_cpp, 15},
    {"fastgbm_predict_cpp", (DL_FUNC) &fastgbm_predict_cpp, 3},
    {"fastgbm_importance_cpp", (DL_FUNC) &fastgbm_importance_cpp, 1},
    {NULL, NULL, 0}
};

void R_init_fastgbm(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
