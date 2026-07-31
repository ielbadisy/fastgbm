#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

SEXP fastgbm_fit_cpp(SEXP xSEXP, SEXP timeSEXP, SEXP statusSEXP,
                     SEXP objectiveSEXP, SEXP ntreesSEXP, SEXP learning_rateSEXP,
                     SEXP max_depthSEXP, SEXP min_node_sizeSEXP, SEXP max_binsSEXP,
                     SEXP subsampleSEXP, SEXP colsampleSEXP, SEXP lambdaSEXP,
                     SEXP gammaSEXP, SEXP min_child_weightSEXP, SEXP seedSEXP,
                     SEXP threadsSEXP, SEXP verboseSEXP,
                     SEXP x_validSEXP, SEXP time_validSEXP, SEXP status_validSEXP,
                     SEXP early_stoppingSEXP);
SEXP fastgbm_predict_cpp(SEXP modelSEXP, SEXP xSEXP, SEXP typeSEXP);
SEXP fastgbm_importance_cpp(SEXP modelSEXP);
SEXP fastgbm_grad_hess_cpp(SEXP predSEXP, SEXP timeSEXP, SEXP statusSEXP, SEXP objectiveSEXP);

static const R_CallMethodDef CallEntries[] = {
    {"fastgbm_fit_cpp", (DL_FUNC) &fastgbm_fit_cpp, 21},
    {"fastgbm_predict_cpp", (DL_FUNC) &fastgbm_predict_cpp, 3},
    {"fastgbm_importance_cpp", (DL_FUNC) &fastgbm_importance_cpp, 1},
    {"fastgbm_grad_hess_cpp", (DL_FUNC) &fastgbm_grad_hess_cpp, 4},
    {NULL, NULL, 0}
};

void R_init_fastgbm(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
