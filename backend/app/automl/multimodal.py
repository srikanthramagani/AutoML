"""Validation-only adaptive tabular/text classification with complete export."""
from __future__ import annotations
import time
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
import joblib

_SENTENCE_MODELS = {}
_EMBEDDING_CACHE = OrderedDict()
_EMBEDDING_CACHE_MAX = 25000


class _SentenceEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
        self.model_name, self.batch_size = model_name, batch_size
    def fit_transform(self, texts): return self.transform(texts)
    def transform(self, texts):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Sentence Transformer support needs `pip install sentence-transformers`. TF-IDF works without it.") from exc
        if self.model_name not in _SENTENCE_MODELS:
            _SENTENCE_MODELS[self.model_name] = SentenceTransformer(self.model_name, device="cuda" if self._cuda_available() else "cpu")
        self.model_ = _SENTENCE_MODELS[self.model_name]
        texts=list(texts); missing=list(dict.fromkeys(t for t in texts if (self.model_name,t) not in _EMBEDDING_CACHE))
        if missing:
            vectors=self.model_.encode(missing,batch_size=self.batch_size,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
            for text,vector in zip(missing,vectors):
                _EMBEDDING_CACHE[(self.model_name,text)]=vector
                while len(_EMBEDDING_CACHE)>_EMBEDDING_CACHE_MAX: _EMBEDDING_CACHE.popitem(last=False)
        vectors=[]
        for text in texts:
            key=(self.model_name,text); vector=_EMBEDDING_CACHE[key]; _EMBEDDING_CACHE.move_to_end(key); vectors.append(vector)
        return np.asarray(vectors)
    def __getstate__(self):
        state=self.__dict__.copy(); state.pop("model_",None); return state
    @staticmethod
    def _cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError: return False


def _model(name, seed, params=None):
    params=params or {}
    models = {"Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", C=params.get('C',1.0)),
              "Random Forest": RandomForestClassifier(n_estimators=params.get('n_estimators',160), min_samples_leaf=params.get('min_samples_leaf',2),max_features=params.get('max_features','sqrt'), class_weight="balanced", n_jobs=-1, random_state=seed),
              "Extra Trees": ExtraTreesClassifier(n_estimators=params.get('n_estimators',160),min_samples_leaf=params.get('min_samples_leaf',2),max_features=params.get('max_features','sqrt'),class_weight="balanced", n_jobs=-1, random_state=seed),
              "Linear SVM": CalibratedClassifierCV(LinearSVC(C=params.get('C',1.0),class_weight="balanced", random_state=seed), cv=3)}
    return models[name]


class MultimodalClassifier(BaseEstimator, ClassifierMixin):
    """Pickleable transformer+classifier pipeline; incoming rows are DataFrames."""
    def __init__(self, text_columns, numeric_columns, categorical_columns, representation="tfidf", fusion="early", model_name="Logistic Regression", seed=42, max_features=20000, sentence_model="sentence-transformers/all-MiniLM-L6-v2", fusion_weight=.5, model_params=None, ngram_range=(1,2), min_df=1):
        self.text_columns=text_columns; self.numeric_columns=numeric_columns; self.categorical_columns=categorical_columns
        self.representation=representation; self.fusion=fusion; self.model_name=model_name; self.seed=seed
        self.max_features=max_features; self.sentence_model=sentence_model; self.fusion_weight=fusion_weight
        self.model_params=model_params; self.ngram_range=ngram_range; self.min_df=min_df

    def _texts(self, X): return X.reindex(columns=self.text_columns).fillna("").astype(str).agg(" [SEP] ".join, axis=1).tolist()
    def _build(self, X, y):
        self.feature_columns_ = list(X.columns)
        self.classes_ = np.unique(y)
        cols = self.numeric_columns + self.categorical_columns
        transformers=[]
        if self.numeric_columns: transformers.append(("num", Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())]),self.numeric_columns))
        try:
            categorical_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
        except TypeError:
            categorical_encoder = OneHotEncoder(handle_unknown="ignore")
        if self.categorical_columns: transformers.append(("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",categorical_encoder)]),self.categorical_columns))
        self.tabular_ = ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0) if transformers else None
        if self.tabular_ is not None: self.tabular_.fit(X,y)
        if self.representation == "tfidf":
            self.text_ = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range, min_df=self.min_df, sublinear_tf=True, strip_accents="unicode",stop_words='english')
            self.text_.fit(self._texts(X))
        elif self.representation == "sentence_transformer":
            self.text_ = _SentenceEncoder(self.sentence_model); self.text_.fit_transform(self._texts(X))
        else: raise ValueError("representation must be tfidf or sentence_transformer")
        self.text_model_=_model(self.model_name,self.seed,self.model_params)
        self.tab_model_=_model(self.model_name,self.seed,self.model_params)
        if self.fusion == "early":
            A=self._features(X, fitting=False); self.model_=_model(self.model_name,self.seed,self.model_params); self.model_.fit(A,y)
        elif self.fusion == "late":
            if self.tabular_ is None: raise ValueError("Late fusion requires at least one tabular feature.")
            T=self.text_.transform(self._texts(X))
            A=self.tabular_.transform(X)
            inner=min(3,int(pd.Series(y).value_counts().min()))
            if inner >= 2:
                splitter=StratifiedKFold(inner,shuffle=True,random_state=self.seed)
                oof_text=cross_val_predict(self.text_model_,T,y,cv=splitter,method="predict_proba",n_jobs=1)
                oof_tab=cross_val_predict(self.tab_model_,A,y,cv=splitter,method="predict_proba",n_jobs=1)
                weights=np.linspace(0,1,21)
                scored=[(f1_score(y,self.classes_[np.argmax(w*oof_text+(1-w)*oof_tab,axis=1)],average="macro"),-abs(w-.5),w) for w in weights]
                self.fusion_weight=max(scored)[2]
            self.tab_model_.fit(A,y)
            self.text_model_.fit(T,y)
        else: raise ValueError("fusion must be early or late")
        return self

    def _text_matrix(self, X):
        t=self.text_.transform(self._texts(X))
        return sparse.csr_matrix(t) if isinstance(t,np.ndarray) else t
    def _features(self,X,fitting=False):
        X=X.reindex(columns=self.feature_columns_)
        T=self._text_matrix(X)
        if self.tabular_ is None: return T
        A=self.tabular_.transform(X)
        return sparse.hstack([A,T],format="csr") if sparse.issparse(A) or sparse.issparse(T) else np.hstack([A,T])
    def fit(self,X,y): return self._build(X,pd.Series(y))
    def predict_proba(self,X):
        check_is_fitted(self,"classes_")
        if self.fusion=="early": return self.model_.predict_proba(self._features(X))
        pt=self.text_model_.predict_proba(self._text_matrix(X)); pa=self.tab_model_.predict_proba(self.tabular_.transform(X))
        # Align class columns if an estimator omits a class in a fold (usually prevented by stratification).
        return self.fusion_weight*pt+(1-self.fusion_weight)*pa
    def predict(self,X): return self.classes_[np.argmax(self.predict_proba(X),axis=1)]


def train_multimodal(frame, target, modality, *, representation="tfidf", fusion="early", trials=6, folds=3, seed=42, export_path=None):
    run_started=time.perf_counter()
    if target not in frame: raise ValueError("Target column does not exist.")
    text=list(modality.text); nums=list(modality.numerical)+list(modality.boolean); cats=list(modality.categorical)+list(modality.datetime)
    used=set(text+nums+cats); y=frame[target]; X=frame.drop(columns=[target]+list(modality.identifiers),errors="ignore")
    text=[c for c in text if c in X]; nums=[c for c in nums if c in X]; cats=[c for c in cats if c in X]
    for col in modality.datetime:
        if col in X: X[col]=pd.to_datetime(X[col],errors="coerce").astype("string")
    if not text: raise ValueError("No text columns detected; configure at least one text feature.")
    if y.nunique()<2: raise ValueError("Target must contain at least two classes.")
    counts=y.value_counts(); holdout=max(1, int(round(.2*len(y))))
    if counts.min()<2 or len(y)<12: raise ValueError("At least 12 rows and two examples per class are required.")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=max(holdout, y.nunique()),random_state=seed,stratify=y)
    folds=min(folds,int(ytr.value_counts().min()))
    if folds<2: raise ValueError("Insufficient class examples for cross-validation.")
    configs=[]
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    candidates=["Logistic Regression","Random Forest","Extra Trees","Linear SVM"]
    if representation=="auto":
        reps=["tfidf"]
        try:
            import sentence_transformers  # noqa: F401
            reps.append("sentence_transformer")
        except ImportError: pass
    else: reps=[representation]
    fusions=["early","late"] if fusion=="auto" and (nums or cats) else (["early"] if fusion=="auto" else [fusion])
    def objective(trial):
        name=trial.suggest_categorical("model",candidates)
        rep=trial.suggest_categorical("representation",reps)
        fus=trial.suggest_categorical("fusion",fusions)
        max_features=trial.suggest_int("max_features",2000,20000,step=2000) if rep=="tfidf" else 20000
        ngram_choice=trial.suggest_categorical('ngram_range',['unigram','unigram_bigram']) if rep=='tfidf' else 'unigram_bigram'
        ngram=(1,1) if ngram_choice=='unigram' else (1,2)
        min_df=trial.suggest_int('min_df',1,3) if rep=='tfidf' else 1
        model_params={}
        if name in ('Logistic Regression','Linear SVM'): model_params['C']=trial.suggest_float('C',1e-3,100,log=True)
        else:
            model_params['n_estimators']=trial.suggest_categorical('n_estimators',[100,200,300])
            model_params['min_samples_leaf']=trial.suggest_int('min_samples_leaf',1,5)
            model_params['max_features']=trial.suggest_categorical('tree_max_features',['sqrt',.5])
        pipe=MultimodalClassifier(text,nums,cats,rep,fus,name,seed,max_features=max_features,model_params=model_params,ngram_range=ngram,min_df=min_df)
        if name == "Logistic Regression": pipe.model_name=name
        start=time.perf_counter()
        scores=cross_val_score(pipe,Xtr,ytr,cv=StratifiedKFold(folds,shuffle=True,random_state=seed),scoring="f1_macro",n_jobs=1)
        elapsed=time.perf_counter()-start
        configs.append({"name":name,"score":float(scores.mean()),"std":float(scores.std()),"seconds":elapsed,"scores":scores.tolist(),"trial":trial.number,"representation":rep,"fusion":fus,"max_features":max_features,"model_params":model_params,"ngram_range":ngram,"min_df":min_df})
        return float(scores.mean())
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective,n_trials=trials)
    winner=max(configs,key=lambda c:c["score"])
    final=MultimodalClassifier(text,nums,cats,winner["representation"],winner["fusion"],winner["name"],seed,max_features=winner["max_features"],model_params=winner['model_params'],ngram_range=winner['ngram_range'],min_df=winner['min_df']).fit(Xtr,ytr)
    started=time.perf_counter(); pred=final.predict(Xte); infer=time.perf_counter()-started
    p,r,f,_=precision_recall_fscore_support(yte,pred,average="macro",zero_division=0)
    pw,rw,fw,_=precision_recall_fscore_support(yte,pred,average="weighted",zero_division=0)
    per_class=precision_recall_fscore_support(yte,pred,labels=final.classes_,average=None,zero_division=0)
    metrics={"accuracy":float(accuracy_score(yte,pred)),"precision_macro":float(p),"recall_macro":float(r),"f1_macro":float(f),"precision_weighted":float(pw),"recall_weighted":float(rw),"f1_weighted":float(fw),"confusion_matrix":confusion_matrix(yte,pred,labels=final.classes_).tolist(),"per_class":{str(label):{"precision":float(per_class[0][i]),"recall":float(per_class[1][i]),"f1":float(per_class[2][i]),"support":int(per_class[3][i])} for i,label in enumerate(final.classes_)}}
    try:
        probs=final.predict_proba(Xte)
        metrics["roc_auc"] = float(roc_auc_score(yte,probs[:,1] if probs.shape[1]==2 else probs,multi_class="ovr" if probs.shape[1]>2 else "raise",average="macro"))
    except (ValueError,TypeError): pass
    # Post-selection explanations are measured on the holdout and never feed back into optimization.
    from sklearn.inspection import permutation_importance
    try:
        pi=permutation_importance(final,Xte,yte,n_repeats=2,random_state=seed,scoring="f1_macro",n_jobs=1)
        feature_importance=sorted([{"feature":str(name),"importance_mean":float(value),"importance_std":float(std)} for name,value,std in zip(Xte.columns,pi.importances_mean,pi.importances_std)],key=lambda x:x['importance_mean'],reverse=True)
    except Exception: feature_importance=[]
    branch_delta={}
    try:
        base=f1_score(yte,pred,average='macro')
        if text:
            masked=Xte.copy(); masked[text]=""; branch_delta['text_f1_drop']=float(base-f1_score(yte,final.predict(masked),average='macro'))
        if nums or cats:
            masked=Xte.copy(); masked[nums+cats]=np.nan; branch_delta['tabular_f1_drop']=float(base-f1_score(yte,final.predict(masked),average='macro'))
    except Exception: pass
    if export_path: joblib.dump(final,export_path)
    return {"model":final,"selected_model":winner["name"],"best_params":study.best_params,"cv_score":winner["score"],"cv_std":winner["std"],"trials":configs,"metrics":metrics,"feature_importance":feature_importance,"modality_ablation":branch_delta,"training_seconds":time.perf_counter()-run_started-infer,"inference_seconds":infer,"model_size_bytes":Path(export_path).stat().st_size if export_path and Path(export_path).exists() else None,"n_train":len(Xtr),"n_test":len(Xte),"representation":winner["representation"],"fusion":winner["fusion"],"fusion_weight":float(final.fusion_weight) if winner["fusion"]=="late" else None,"classes":[str(c) for c in final.classes_]}


def explain_records(estimator, records, top_k=8):
    """Example-level confidence perturbation and linear TF-IDF term weights.

    Perturbation is predictive attribution, not a causal effect.
    """
    X=records.copy(); probabilities=estimator.predict_proba(X); labels=estimator.classes_[np.argmax(probabilities,axis=1)]
    result=[]
    for i,(_,row) in enumerate(X.iterrows()):
        predicted=labels[i]; cls=int(np.where(estimator.classes_==predicted)[0][0]); base=float(probabilities[i,cls])
        entry={'prediction':str(predicted),'confidence':base,'fusion':estimator.fusion,'fusion_weight_text':float(estimator.fusion_weight) if estimator.fusion=='late' else None,'attribution_note':'Masking-based predictive attribution; not a causal explanation.'}
        if estimator.fusion=='late':
            entry['text_branch_probability']=float(estimator.text_model_.predict_proba(estimator._text_matrix(X.iloc[[i]]))[0,cls])
            entry['tabular_branch_probability']=float(estimator.tab_model_.predict_proba(estimator.tabular_.transform(X.iloc[[i]]))[0,cls])
        for modality,columns in [('text',estimator.text_columns),('tabular',estimator.numeric_columns+estimator.categorical_columns)]:
            if not columns: continue
            masked=X.iloc[[i]].copy(); masked.loc[:,columns]="" if modality=='text' else np.nan
            entry[f'{modality}_confidence_drop']=base-float(estimator.predict_proba(masked)[0,cls])
        model=estimator.model_ if estimator.fusion=='early' else estimator.text_model_
        if estimator.representation=='tfidf' and hasattr(model,'coef_'):
            vocabulary=estimator.text_.get_feature_names_out(); coef=model.coef_[cls] if model.coef_.shape[0]>1 else model.coef_[0]
            tokens=estimator._texts(X.iloc[[i]])[0].lower().split(); present=set(tokens)
            entry['important_text_terms']=sorted([{'term':str(vocabulary[j]),'weight':float(coef[j])} for j in np.argsort(np.abs(coef))[-min(len(coef),top_k*4):] if vocabulary[j] in present],key=lambda x:abs(x['weight']),reverse=True)[:top_k]
        else: entry['important_text_terms']=[]
        result.append(entry)
    return result
