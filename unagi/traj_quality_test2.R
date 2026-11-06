# 决定性实验 v2：完整复刻用户原流程，只改「ScaleData 要不要回归细胞质量」一个变量。
#
# v1 的错误：直接在已取好子集的 macro_traj3b 上重算 UMAP。
# 但原代码是在**全部巨噬细胞（13 个群）**上算 UMAP，之后才层层取子集，
# cds4 用的是那个全集 UMAP 坐标。所以必须从 macro_cells.qs 走完整条链。
#
# A 对照组：ScaleData()                               ← 原代码
# B 实验组：ScaleData(vars.to.regress=质量)
# 其余（dims=1:15、resolution=0.2、所有筛选框、root 规则）完全一致。
# A 必须复现出 CDKN1C→CLEC4E→SEPTIN7→DEFB1，否则 B 没有参照。

.libPaths(c("~/R/qslib", "~/R/library", .libPaths()))
suppressMessages({library(qs); library(Seurat); library(monocle3)
                  library(igraph); library(dplyr)})
set.seed(42)
OUT <- Sys.getenv("TRAJ_TEST_OUT")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
BRANCH1 <- c("Macro-CDKN1C", "Macro-CLEC4E", "Macro-SEPTIN7", "Macro-DEFB1")

NEW_LABELS <- c("0"="Macro-FABP4","1"="Macro-SPP1","2"="Macro-SEPTIN7","3"="Macro-SIGLEC10",
                "4"="Macro-CLEC4E","5"="Macro-CDKN1C","6"="Macro-DEFB1","7"="Macro-LSAMP",
                "8"="Macro-APOC2","9"="Macro-SLC7A11","10"="Macro-GZMH","11"="Macro-PI3",
                "12"="Macro-IL10")

cat(">>> 读 macro_cells.qs\n")
base_obj <- qread(Sys.getenv("SIL_MACRO_QS"))
cat("   ", ncol(base_obj), "细胞\n")

run_one <- function(regress, tag) {
  cat("\n==========", tag, "==========\n")
  mc <- base_obj
  DefaultAssay(mc) <- "RNA"
  mc[["RNA"]] <- JoinLayers(mc[["RNA"]])
  mc <- NormalizeData(mc, verbose = FALSE)
  mc <- FindVariableFeatures(mc, verbose = FALSE)
  if (is.null(regress)) {
    mc <- ScaleData(mc, verbose = FALSE)
  } else {
    cat("    回归掉:", paste(regress, collapse = ", "), "\n")
    mc <- ScaleData(mc, vars.to.regress = regress, verbose = FALSE)
  }
  mc <- RunPCA(mc, verbose = FALSE)
  mc <- FindNeighbors(mc, dims = 1:15, verbose = FALSE)
  mc <- FindClusters(mc, resolution = 0.2, verbose = FALSE)
  mc <- RunUMAP(mc, dims = 1:15, verbose = FALSE)
  cat("    聚类数:", length(unique(mc$seurat_clusters)), "\n")

  # 标注（簇号→亚群名，同原代码）
  Idents(mc) <- mc$seurat_clusters
  lv <- levels(Idents(mc))
  lab <- NEW_LABELS[lv]
  lab[is.na(lab)] <- paste0("Cluster", lv[is.na(lab)])
  names(lab) <- lv
  mc <- RenameIdents(mc, lab)
  mc$celltype <- as.character(Idents(mc))
  print(table(mc$celltype))

  # 层层取子集（同原代码）
  Idents(mc) <- "celltype"
  drop1 <- intersect(c("Macro-GZMH","Macro-LSAMP"), unique(mc$celltype))
  o <- if (length(drop1)) subset(mc, idents = drop1, invert = TRUE) else mc
  drop2 <- intersect(c("Macro-SLC7A11","Macro-PI3","Macro-IL10"), unique(o$celltype))
  if (length(drop2)) o <- subset(o, idents = drop2, invert = TRUE)

  uc <- as.data.frame(Embeddings(o, "umap")); colnames(uc) <- c("UMAP1","UMAP2")
  keep <- rownames(uc[uc$UMAP1 >= -8 & uc$UMAP1 <= 12 & uc$UMAP2 >= -13 & uc$UMAP2 <= 12, ])
  o <- subset(o, cells = keep)
  o <- subset(o, cells = rownames(o@meta.data)[o@meta.data$celltype != "Macro-APOC2"])
  uc2 <- as.data.frame(Embeddings(o, "umap")); colnames(uc2) <- c("UMAP1","UMAP2")
  keep2 <- rownames(uc2[uc2$UMAP1 >= -7 & uc2$UMAP1 <= 7.5 & uc2$UMAP2 >= -12 & uc2$UMAP2 <= 11, ])
  o <- subset(o, cells = keep2)
  cat("    筛选后:", ncol(o), "细胞\n"); print(table(o$celltype))

  cm <- GetAssayData(o, assay = "RNA", layer = "counts")
  cds <- new_cell_data_set(cm, cell_metadata = o@meta.data,
                           gene_metadata = data.frame(gene_short_name = rownames(cm),
                                                      row.names = rownames(cm)))
  cds <- preprocess_cds(cds, num_dim = 30)
  cds <- reduce_dimension(cds, reduction_method = "UMAP", verbose = FALSE)
  cds@int_colData@listData$reducedDims$UMAP <- Embeddings(o, "umap")   # 覆盖成 Seurat UMAP，同原代码
  cds <- cluster_cells(cds, resolution = 1e-3)
  cds <- learn_graph(cds, use_partition = TRUE)

  vert <- t(cds@principal_graph_aux$UMAP$dp_mst)
  um <- Embeddings(o, "umap")
  ctr <- colMeans(um[o$celltype == "Macro-CDKN1C", , drop = FALSE])
  root <- rownames(vert)[which.min(sqrt(rowSums((vert - matrix(ctr, nrow(vert), 2, byrow = TRUE))^2)))]
  cat("    root:", root, "\n")
  cds <- order_cells(cds, root_pr_nodes = root)

  pt <- pseudotime(cds)
  df <- data.frame(cell = names(pt), pseudotime = as.numeric(pt),
                   celltype = o$celltype[names(pt)], group = o$group[names(pt)],
                   nFeature = o$nFeature_RNA[names(pt)], mt = o$percent.mt[names(pt)])
  df <- df[is.finite(df$pseudotime), ]
  write.csv(df, file.path(OUT, paste0("v2_pseudotime_", tag, ".csv")), row.names = FALSE)

  b1 <- df[df$celltype %in% BRANCH1, ]
  s <- b1 %>% group_by(celltype) %>%
    summarise(n = n(), 拟时序均值 = mean(pseudotime), 检出基因 = mean(nFeature),
              线粒体 = mean(mt), 矽肺占比 = mean(group == "SIL"), .groups = "drop") %>%
    arrange(拟时序均值)
  cat("\n    branch1 按拟时序排序：\n"); print(as.data.frame(s))
  cat("    顺序 =", paste(s$celltype, collapse = " -> "), "\n")
  r2 <- summary(lm(pseudotime ~ nFeature + mt, data = df))$r.squared
  r2mt <- summary(lm(pseudotime ~ mt, data = df))$r.squared
  cat(sprintf("    拟时序 ~ 质量  R2 = %.4f (仅线粒体 %.4f)   Spearman(mt) = %+.3f\n",
              r2, r2mt, cor(df$pseudotime, df$mt, method = "spearman")))
  rm(mc); gc(verbose = FALSE)
  list(order = s$celltype, summary = s, df = df, r2 = r2, r2mt = r2mt)
}

A <- run_one(NULL, "A")
B <- run_one(c("percent.mt", "nFeature_RNA"), "B")

cat("\n\n================ 结论 ================\n")
cat("已发表顺序   : ", paste(BRANCH1, collapse = " -> "), "\n")
cat("A 原样       : ", paste(A$order, collapse = " -> "), "\n")
cat("B 回归质量后 : ", paste(B$order, collapse = " -> "), "\n\n")
cat("A 复现原顺序 :", identical(as.character(A$order), BRANCH1), "\n")
cat("B 保持原顺序 :", identical(as.character(B$order), BRANCH1), "\n")
cat(sprintf("\n拟时序被质量解释: A = %.1f%%  ->  B = %.1f%%\n", A$r2*100, B$r2*100))
saveRDS(list(A = A[c("order","summary","r2","r2mt")], B = B[c("order","summary","r2","r2mt")]),
        file.path(OUT, "v2_result.rds"))
cat("\n>>> 已写", OUT, "\n=== TRAJ TEST2 DONE ===\n")
