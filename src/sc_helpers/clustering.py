import scanpy as sc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
import harmonypy as hm
import re
import seaborn as sns
import matplotlib.pyplot as plt 
## Functions for the gridsearch: 

def compute_balance_score(clusters, conditions):
    clusters = np.asarray(clusters)
    conditions = np.asarray(conditions)

    unique_clusters = np.unique(clusters)
    unique_conditions = np.unique(conditions)
    M = len(unique_conditions)
    uniform = np.ones(M) / M

    scores = []
    weights = []

    for cl in unique_clusters:
        idx = clusters == cl
        sub = conditions[idx]

        counts = np.array([(sub == c).sum() for c in unique_conditions])
        p = counts / counts.sum()

        js = jensenshannon(p, uniform, base=2)
        balance = 1 - js

        scores.append(balance)
        weights.append(len(sub))

    return np.average(scores, weights=weights)

def run_normalization(X, method, target_sum=None):
    ad = sc.AnnData(X.copy())

    if method == "sct":
        sc.experimental.pp.normalize_pearson_residuals(ad)
        return ad.X.copy()

    elif method == "libsize":
        sc.pp.normalize_total(ad, target_sum=target_sum)
        return ad.X.copy()

    else:
        raise ValueError("Unknown normalization")
    
def run_log(X, do_log):
    if not do_log:
        return X.copy()
    ad = sc.AnnData(X.copy())
    sc.pp.log1p(ad)
    return ad.X.copy()

def run_scale(X, do_scale):
    if not do_scale:
        return X.copy()
    ad = sc.AnnData(X.copy())
    sc.pp.scale(ad)
    return ad.X.copy()

def run_pca(X, n_pcs,random_s):
    ad = sc.AnnData(X.copy())
    #sc.tl.pca(ad, n_comps=n_pcs, svd_solver="arpack") # maybe change to scanoy.pp.pca ? seems to be the newer version 
    sc.pp.pca(ad,n_comps=n_pcs,key_added="X_pca",random_state=random_s)
    return ad.obsm["X_pca"].copy()

def run_integration(X_pca,theta,batch_values):
    batch_df=pd.DataFrame({"sample":batch_values})
    harmony_out = hm.run_harmony(X_pca, batch_df, "sample",theta=theta,max_iter_harmony=25,verbose=False)
    return harmony_out.Z_corr

def cluster_embedding(X_pca, k, algo, resolution,random_s):
    ad = sc.AnnData(X_pca.copy())
    sc.pp.neighbors(ad, n_neighbors=k,random_state=random_s)

    if algo == "louvain":
        sc.tl.louvain(ad, resolution=resolution,random_state=random_s)
        return ad.obs["louvain"].astype(int).values

    elif algo == "leiden":
        sc.tl.leiden(ad, resolution=resolution,random_state=random_s,flavor="igraph")
        return ad.obs["leiden"].astype(int).values

    else:
        raise ValueError("Unknown clustering algorithm")
    
def evaluate_metrics(X_pca, clusters, conditions, max_sample=3000):
    n = X_pca.shape[0]

    if n > max_sample:
        idx = np.random.choice(n, max_sample, replace=False)
        Xs = X_pca[idx]
        cs = clusters[idx]
    else:
        Xs = X_pca
        cs = clusters

    if len(np.unique(cs)) < 2:
        return dict(sil=np.nan, db=np.nan, ch=np.nan, bal=np.nan)

    return dict(
        n_clusters=len(np.unique(clusters)),
        sil=silhouette_score(Xs, cs),
        db=davies_bouldin_score(Xs, cs),
        ch=calinski_harabasz_score(Xs, cs),
        bal=compute_balance_score(clusters, conditions)
    )


def process_one(scale_key, Xscaled,pcs_list,k_list,algos,resolutions,conditions,n_eval,random_s,obs_names,integration_list):
        out = []
        labels={}
        pca_harmonized={}
        (log_key, scaleflag) = scale_key
        (norm_key,logflag)=log_key
        #(norm_key, logflag, scaleflag) = scale_key
        norm_method, target_sum = norm_key

        for npc in pcs_list:
            X_pca = run_pca(Xscaled, npc,random_s)
            for integration in integration_list:
                #if integration=="harmony":
                if re.match("harmony_[0-9\\.]+",str(integration)):
                    t=float(integration.replace("harmony_",""))
                    X_int=run_integration(X_pca,theta=t,batch_values=conditions)
                elif integration is False:
                    X_int=X_pca
                else:
                    raise ValueError(f"Integration method {integration} not implemented")

                for k in k_list:
                    for algo in algos:
                        for res in resolutions:

                            clusters = cluster_embedding(X_int, k, algo, res,random_s)
                            metrics = evaluate_metrics(X_int, clusters, conditions,n_eval)

                            run_id = (
                            f"{norm_method}_tgt{target_sum}_log{logflag}_scale{scaleflag}_pc{npc}_int{integration}_k{k}_{algo}_res{res}"
                            )
                            out.append(dict(
                                run_id=run_id,
                                normalization=norm_method,
                                libsize_target=target_sum,
                                log=logflag,
                                scale=scaleflag,
                                pcs=npc,
                                k=k,
                                algorithm=algo,
                                resolution=res,
                                integration=integration,
                                **metrics
                            ))

                            labels[run_id] = pd.DataFrame(
                            {"cluster": clusters},
                            index=obs_names
                            ) # the order of the index comes from the the fact that obs_names was defined from the adata object directly in run_full_pipeline function
                            if re.match("harmony_[0-9\\.]+",str(integration)):
                                pca_harmonized[run_id]=X_int
                            else:
                                pca_harmonized[run_id]=None
        return out , labels , pca_harmonized

def run_full_pipeline(adata,condition_col="condition",n_jobs=2,random_s=42,parameter_dict=None):

    X0 = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    X0 = X0.copy() 
    obs_names = adata.obs_names.copy()
    conditions = adata.obs[condition_col].values
    if not parameter_dict:
        #Parameter grid
        normalization_methods = ["libsize"]
        libsize_targets = [100, 1000]
        log_opts = [True, False]
        scale_opts = [True, False]
        pcs_list = [10, 20, 30, 50]
        k_list = [12, 16, 30, 60]
        algos = ["leiden"]
        resolutions = [0.2,0.3,0.4,0.6,0.8]
        integrations=[False]
    else:
        normalization_methods = parameter_dict["normalization_methods"]
        libsize_targets = parameter_dict["libsize_targets"]
        log_opts = parameter_dict["log_opts"]
        scale_opts = parameter_dict["scale_opts"]
        pcs_list = parameter_dict["pcs_list"]
        k_list = parameter_dict["k_list"]
        algos = parameter_dict["algos"]
        resolutions = parameter_dict["resolutions"]
        integrations=parameter_dict["integrations"]



    # normalization_methods = ["libsize"]
    # libsize_targets = [100]
    # log_opts = [True, False]
    # scale_opts = [True, False]
    # pcs_list = [10, 20]
    # k_list = [12, 16,60]
    # algos = ["louvain", "leiden"]
    # resolutions = [0.2, 0.4]

    # === 1. cache normalization ===
    norm_cache = {}
    for nm in normalization_methods:
        if nm == "sct":
            norm_cache[(nm, None)] = run_normalization(X0, "sct")
        else:
            for tgt in libsize_targets:
                norm_cache[(nm, tgt)] = run_normalization(X0, "libsize", tgt)
    # === 2. cache log ===
    log_cache = {}
    for norm_key, Xnorm in norm_cache.items():
        for logflag in log_opts:
            if not (logflag==True and norm_key==('sct', None)):# never combine sct norm with log transfo, not needed as per Sallas 2025 and produce nan because of neg value
                log_cache[(norm_key, logflag)] = run_log(Xnorm, logflag)
    # === 3. cache scaling ===
    scale_cache = {}
    for log_key, Xlog in log_cache.items():
        for scaleflag in scale_opts:
            scale_cache[(log_key, scaleflag)] = run_scale(Xlog, scaleflag)
    # === worker for PCA + neighbors + clustering + metrics ===
    
    # === run workers in parallel ===
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(process_one)(scale_key, Xscaled,pcs_list,k_list,algos,resolutions,conditions,n_eval=X0.shape[0],random_s=random_s,obs_names=obs_names,integration_list=integrations)
        for scale_key, Xscaled in scale_cache.items()
    )

    # flatten list
    flat = []
    all_labels = {}
    all_pca_harmonized={}
    for out, labels, pca in results:
        flat.extend(out)
        all_labels.update(labels)
        all_pca_harmonized.update(pca)
 
    df = pd.DataFrame(flat)
    return df, all_labels , all_pca_harmonized

def run_subcluster_gridsearch(adata,condition_col,parameter_dict=None):
    if parameter_dict:
        clust_params, labels_dict= clustering_gridsearch_helpers.run_full_pipeline(adata, condition_col=condition_col, n_jobs=12,parameter_dict=parameter_dict)
    else :
        clust_params, labels_dict= clustering_gridsearch_helpers.run_full_pipeline(adata, condition_col=condition_col, n_jobs=12)

    clust_params["rank_sil"]=clust_params["sil"].rank(ascending=False)
    clust_params["rank_ch"]=clust_params["ch"].rank(ascending=False)
    clust_params["rank_db"]=clust_params["db"].rank(ascending=True)
    clust_params["sum_rank"]=clust_params["rank_sil"] + clust_params["rank_ch"]+ clust_params["rank_db"]
    clust_params=clust_params.sort_values(axis=0,by="sum_rank",ascending=True)

    return clust_params, labels_dict

## Functions to evaluate the gridsearch: 


# this is the main function that shows for a given clustering result (i.e the best clustering result for a given number of clusters , or a given run if n_clust_rank is a run_id) the umap (in the harmonized space) 
# the dotplos for the clsuter marker and for other gene sets of interest. It also shows the crosstab of these cluster to the cluster obtained on the timepoints separately.
def show_clustering_results(adata,metrics_df,labels_dict,pca_harmonized_dict,n_clust_rank,snap_marker_genes,umap_columns,label_column,crosstab_column,gene_set_1=None,gene_set_2=None,batch_column=None,output="labels",final_labels=None):
    adata=adata.copy()
    # select clustering results
    if isinstance(n_clust_rank,str):
        best_params=metrics_df[metrics_df["run_id"]==n_clust_rank].iloc[0,]
        run_id=n_clust_rank
    else:
        if n_clust_rank==0:
            best_params=metrics_df.iloc[0,:]
            run_id=best_params["run_id"]
        else:
            best_params=metrics_df.iloc[0,:]
            best_n=best_params["n_clusters"]
            best_params=metrics_df[metrics_df["n_clusters"]==best_n+n_clust_rank]
            best_params=best_params.iloc[0,:]
            run_id=best_params["run_id"]
    print (f"run_id: {run_id}")
    # pre-process dataset according to clustering results
    adata.X=adata.layers["raw_counts"].copy()
    sc.pp.normalize_total(adata,target_sum=best_params["libsize_target"])
    if best_params["log"]:
        sc.pp.log1p(adata)
    if best_params["scale"]:
        sc.pp.scale(adata)
    if re.match("harmony_[0-9\\.]+",str(best_params["integration"])):
        #adata.obsm["X_pca_current"]=adata.obsm["X_PCA_harmony"] # change back to X_pca_harmony
        adata.obsm["X_pca_current"]=pca_harmonized_dict[run_id] # change back to X_pca_harmony
    else:
        sc.pp.pca(adata,n_comps=best_params["pcs"],key_added="X_pca_current")
    sc.pp.neighbors(adata, n_neighbors=best_params["k"],use_rep="X_pca_current",key_added="X_neighbors",random_state=42) # this neighbor graph is computed to give a nicer umap than the one using for clustering 
    sc.tl.umap(adata,neighbors_key="X_neighbors",key_added="umap",min_dist=0.1,random_state=42)
    if not best_params["log"]: # still need to log transform for the plots 
        sc.pp.log1p(adata)
        adata.layers["log_counts"]=adata.X.copy() # store log counts for the DEG analysis

    #merge the labels from the labels dict
    adata.obs=adata.obs.join(labels_dict[best_params["run_id"]],how="left")
    if final_labels:
        adata.obs[label_column]= adata.obs["cluster"].map(final_labels).astype("category")
    else:
        adata.obs[label_column]=adata.obs["cluster"].astype(str).astype("category")


    # make diagnostic plots
    if batch_column:
        sc.pl.embedding(adata,"umap",color=umap_columns.extend(batch_column),ncols=3)
    else:
        sc.pl.embedding(adata,"umap",color=umap_columns,ncols=3)

    sc.tl.dendrogram(adata,groupby=label_column,use_rep="umap",key_added="dendrogram_"+label_column)

    sc.tl.rank_genes_groups(
        adata,layer="log_counts",groupby=label_column, method="wilcoxon", key_added="deg_clusters"
    )
    sc.pl.rank_genes_groups_dotplot(adata,layer="log_counts", groupby=label_column,standard_scale="var", n_genes=5, key="deg_clusters")
    sc.pl.rank_genes_groups_dotplot(adata,layer="log_counts", groupby=label_column, n_genes=5, key="deg_clusters")

    markers=sc.get.rank_genes_groups_df(adata,key="deg_clusters",group=None)
    markers = (
    markers.sort_values(["group", "scores"], ascending=[True, False])
      .groupby("group")
      .head(5)
    )

    if snap_marker_genes:
        snap_annotations=snap.annotate_hierarchy(
        adata,
        snap_marker_genes,
        group_name=label_column,
        layer="log_counts",
    )

        print (snap_annotations["assignments"])

    if gene_set_1:
        if isinstance(gene_set_1,list):
            genes_1=[g for g in gene_set_1 if g in adata.var.index]
        elif isinstance(gene_set_1,dict):
            genes_1={cell_type: [gene for gene in gene_list if gene in adata.var.index] for cell_type, gene_list in gene_set_1.items()}
        else:
            raise ValueError("gene_set_1 must be a dict or list")
        sc.pl.dotplot(adata,layer="log_counts",groupby=label_column,var_names=genes_1,standard_scale="var")
        sc.pl.dotplot(adata,layer="log_counts",groupby=label_column,var_names=genes_1)

    if gene_set_2:
        if isinstance(gene_set_2,list):
            genes_2=[g for g in gene_set_2 if g in adata.var.index]
        elif isinstance(gene_set_2,dict):
            genes_2={cell_type: [gene for gene in gene_list if gene in adata.var.index] for cell_type, gene_list in gene_set_2.items()}
        else:
            raise ValueError("gene_set_2 must be a dict or list")
        sc.pl.dotplot(adata,layer="log_counts",groupby=label_column,var_names=genes_2,standard_scale="var")
        sc.pl.dotplot(adata,layer="log_counts",groupby=label_column,var_names=genes_2)

    plt.figure(figsize=(8, 5))
    crosstab_labels=adata.obs.groupby([label_column])[crosstab_column].value_counts().reset_index(name='counts')
    ax1=sns.histplot(
    data=crosstab_labels,
    x=label_column,
    hue=crosstab_column,
    weights="counts",
    multiple="stack",
    shrink=0.8,
    palette='tab20'
    
    )
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.ylabel("Cell Counts")
    plt.tight_layout()
    plt.show()


    if output=="markers":
        return markers
    elif output=="labels":
        return labels_dict[best_params["run_id"]]