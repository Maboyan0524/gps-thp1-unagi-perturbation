# 轨迹对 root 选择有多敏感？
# 用 A 组（复刻原流程、聚类与原文一致）的那套嵌入，只换 root，看：
#   1. 四个亚群的先后顺序会不会变
#   2. 拟时序被细胞质量解释的比例会不会变（原 cds4 是 26%，A 组程序选根是 0.6%）
# root 候选：离每个亚群质心最近的节点 + 图上均匀抽若干节点。

.libPaths(c("~/R/qslib", "~/R/library", .libPaths()))
suppressMessages({library(qs); library(Seurat); library(monocle3); library(dplyr)})
set.seed(42)
OUT <- Sys.getenv("TRAJ_TEST_OUT")
BRANCH1 <- c("Macro-CDKN1C","Macro-CLEC4E","Macro-SEPTIN7","Macro-DEFB1")
NEW_LABELS <- c("0"="Macro-FABP4","1"="Macro-SPP1","2"="Macro-SEPTIN7","3"="Macro-SIGLEC10",
                "4"="Macro-CLEC4E","5"="Macro-CDKN1C","6"="Macro-DEFB1","7"="Macro-LSAMP",
                "8"="Macro-APOC2","9"="Macro-SLC7A11","10"="Macro-GZMH","11"="Macro-PI3",
                "12"="Macro-IL10")

cat(">>> 重建 A 组嵌入（与原流程一致）\n")
mc <- qread(Sys.getenv("SIL_MACRO_QS"))
DefaultAssay(mc) <- "RNA"; mc[["RNA"]] <- JoinLayers(mc[["RNA"]])
mc <- NormalizeData(mc, verbose=FALSE); mc <- FindVariableFeatures(mc, verbose=FALSE)
mc <- ScaleData(mc, verbose=FALSE); mc <- RunPCA(mc, verbose=FALSE)
mc <- FindNeighbors(mc, dims=1:15, verbose=FALSE)
mc <- FindClusters(mc, resolution=0.2, verbose=FALSE)
mc <- RunUMAP(mc, dims=1:15, verbose=FALSE)
Idents(mc) <- mc$seurat_clusters
lv <- levels(Idents(mc)); lab <- NEW_LABELS[lv]
lab[is.na(lab)] <- paste0("Cluster", lv[is.na(lab)]); names(lab) <- lv
mc <- RenameIdents(mc, lab); mc$celltype <- as.character(Idents(mc))
Idents(mc) <- "celltype"
o <- subset(mc, idents=intersect(c("Macro-GZMH","Macro-LSAMP"), unique(mc$celltype)), invert=TRUE)
o <- subset(o, idents=intersect(c("Macro-SLC7A11","Macro-PI3","Macro-IL10"), unique(o$celltype)), invert=TRUE)
uc <- as.data.frame(Embeddings(o,"umap")); colnames(uc) <- c("U1","U2")
o <- subset(o, cells=rownames(uc[uc$U1>=-8 & uc$U1<=12 & uc$U2>=-13 & uc$U2<=12,]))
o <- subset(o, cells=rownames(o@meta.data)[o@meta.data$celltype!="Macro-APOC2"])
uc2 <- as.data.frame(Embeddings(o,"umap")); colnames(uc2) <- c("U1","U2")
o <- subset(o, cells=rownames(uc2[uc2$U1>=-7 & uc2$U1<=7.5 & uc2$U2>=-12 & uc2$U2<=11,]))
cat("   ", ncol(o), "细胞\n")
rm(mc); gc(verbose=FALSE)

cm <- GetAssayData(o, assay="RNA", layer="counts")
cds <- new_cell_data_set(cm, cell_metadata=o@meta.data,
        gene_metadata=data.frame(gene_short_name=rownames(cm), row.names=rownames(cm)))
cds <- preprocess_cds(cds, num_dim=30)
cds <- reduce_dimension(cds, reduction_method="UMAP", verbose=FALSE)
cds@int_colData@listData$reducedDims$UMAP <- Embeddings(o,"umap")
cds <- cluster_cells(cds, resolution=1e-3)
cds <- learn_graph(cds, use_partition=TRUE)

vert <- t(cds@principal_graph_aux$UMAP$dp_mst)
um <- Embeddings(o,"umap")
near <- function(ct) {
  ctr <- colMeans(um[o$celltype==ct,,drop=FALSE])
  rownames(vert)[which.min(sqrt(rowSums((vert - matrix(ctr,nrow(vert),2,byrow=TRUE))^2)))]
}
cands <- list()
for (ct in BRANCH1) cands[[paste0("近_",ct)]] <- near(ct)
set.seed(7)
for (i in 1:8) cands[[paste0("随机_",i)]] <- sample(rownames(vert), 1)
cands <- cands[!duplicated(unlist(cands))]
cat(">>> 共", length(cands), "个候选 root\n\n")

rows <- list()
for (nm in names(cands)) {
  r <- cands[[nm]]
  cdsx <- tryCatch(order_cells(cds, root_pr_nodes=r), error=function(e) NULL)
  if (is.null(cdsx)) { cat(sprintf("  %-22s %s  失败\n", nm, r)); next }
  pt <- pseudotime(cdsx)
  df <- data.frame(pseudotime=as.numeric(pt), celltype=o$celltype[names(pt)],
                   mt=o$percent.mt[names(pt)], nF=o$nFeature_RNA[names(pt)])
  df <- df[is.finite(df$pseudotime),]
  s <- df[df$celltype %in% BRANCH1,] %>% group_by(celltype) %>%
       summarise(m=mean(pseudotime), .groups="drop") %>% arrange(m)
  r2 <- summary(lm(pseudotime ~ nF + mt, data=df))$r.squared
  ord <- paste(sub("Macro-","",s$celltype), collapse=" > ")
  cat(sprintf("  %-22s %-8s  %-34s  质量R2=%5.1f%%\n", nm, r, ord, r2*100))
  rows[[nm]] <- data.frame(root_name=nm, root=r, order=ord, r2=r2)
}
tab <- bind_rows(rows)
write.csv(tab, file.path(OUT,"root_sensitivity.csv"), row.names=FALSE)

cat("\n================ 汇总 ================\n")
cat("已发表顺序: CDKN1C > CLEC4E > SEPTIN7 > DEFB1\n\n")
cat("出现过的顺序及次数:\n"); print(sort(table(tab$order), decreasing=TRUE))
cat(sprintf("\n不同 root 下，质量解释比例从 %.1f%% 到 %.1f%%\n", min(tab$r2)*100, max(tab$r2)*100))
cat(sprintf("命中已发表顺序的 root: %d / %d\n",
            sum(tab$order=="CDKN1C > CLEC4E > SEPTIN7 > DEFB1"), nrow(tab)))
cat("=== ROOT TEST DONE ===\n")
